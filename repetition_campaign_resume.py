"""Resume verified Nemo work in a new campaign on the assigned GPU subset."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
from statistics import median
import subprocess
import sys
import time


DEADLINE = 1795568400
HOST = "nemo-rl-env-0-0"
ARCHIVE_CONFIG = "87ce495c2de41c1734e6f80fd04021e88499e672453ffd55437d86c3694aeeee"
AGGREGATE = "6648362bef446be81e1d10c0a48adc98e1f3aa73c48be60c48fe325ddce92240"
FOLLOWUP_AGGREGATE = "23a14f368f54cf00e5cb0d1090a84a303db7ffcc324659254b37b0cfa305e88e"


def sha(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def load(path):
    return json.loads(path.read_text())


def save(path, data):
    if path.exists():
        if load(path) == data:
            return
        raise FileExistsError(path)
    temp = path.with_suffix(".pending")
    with temp.open("x") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
    temp.replace(path)


def smi(field):
    return [line.split(", ") for line in subprocess.check_output(
        ["nvidia-smi", field, "--format=csv,noheader,nounits"], text=True).splitlines() if line.strip()]


def archive_state(root):
    config_path = root / "config.json"
    if socket.gethostname() != HOST or sha(config_path) != ARCHIVE_CONFIG:
        raise ValueError("Wrong host or archive configuration digest")
    cfg = load(config_path)
    status = load(root / "status.json")
    plans = {}
    for path in sorted((root / "plans").glob("plan-*.json")):
        if sha(path) != status["manifest_shas"].get(path.name):
            raise ValueError(f"Manifest digest changed for {path.name}")
        plans[path.name] = load(path)
    return cfg, status, plans


def verify_completed(root, name, entry):
    path = Path(entry["output"])
    fingerprint = entry["sha256"]
    receipt = load(root / "jobs" / f"{name}.receipt.json")
    if (path.parent != root / "jobs" or receipt["output"] != str(path)
            or receipt["sha256"] != fingerprint or receipt["stage"] != "completed"
            or sha(path) != fingerprint):
        raise ValueError(f"Completed artifact or receipt differs for {name}")
    return fingerprint


def inspect(args):
    cfg, status, plans = archive_state(args.archive)
    print(json.dumps({
        "python": sys.executable, "host": cfg["host"], "status": status["state"],
        "plans": {
            p: {"counts": dict(Counter((j["kind"] + ":" + status["jobs"][j["id"]]["stage"])
                                      for j in plan["jobs"])),
                "examples": [j for j in plan["jobs"][:2]], "worker_pins": plan["pins"]}
            for p, plan in plans.items()
        },
        "old_pids_active": [pid for pid in (197376, 197864)
                            if (Path("/proc") / str(pid) / "cmdline").is_file()
                            and os.fsencode(args.archive) in (Path("/proc") / str(pid) / "cmdline").read_bytes()],
    }, indent=2))


def copy_pinned(original, dest, expected):
    if sha(original) != expected:
        raise ValueError(f"Source digest differs for {original.name}")
    if not dest.exists():
        with dest.open("xb") as output, original.open("rb") as source:
            shutil.copyfileobj(source, output)
    if sha(dest) != expected:
        raise ValueError(f"Staged copy differs for {dest.name}")


def prepare(args, cfg, status, plans):
    ref = status["jobs"]
    root, archive, source = args.root, args.archive, args.source
    completed = {name for name, entry in ref.items() if entry["stage"] == "completed"}
    times = defaultdict(list)
    for entry in ref.values():
        if entry["stage"] == "completed" and entry["kind"] == "gpu" and entry.get("wall_seconds"):
            times[entry["lane"]].append(entry["wall_seconds"])

    pins = {}
    verified = {}
    jobs = []
    old_ids = set()

    def reuse(name):
        if name not in verified:
            verified[name] = verify_completed(archive, name, ref[name])
        return verified[name]

    def add(job, old_pins, priority):
        name = job["id"]
        if name in completed:
            return
        old_worker = Path(job["worker"])
        new_worker = source / old_worker.name
        fingerprint = old_pins[job["worker"]]
        if str(new_worker) not in pins:
            copy_pinned(old_worker, new_worker, fingerprint)
            pins[str(new_worker)] = fingerprint
        elif pins[str(new_worker)] != fingerprint:
            raise ValueError(f"Conflicting worker digest for {new_worker.name}")
        deps = []
        substitutions = {}
        for dep in job["deps"]:
            if dep in completed:
                substitutions[f"{{sha:{dep}}}"] = reuse(dep)
                substitutions[ref[dep]["output"]] = ref[dep]["output"]
            else:
                deps.append(dep)
        output = str(archive / "jobs" / (name + job.get("extension", ".jsonl")))
        if output not in job["argv"]:
            raise ValueError(f"Missing archived output path for {name}")
        output_ref = {value: value.replace(str(archive / "jobs") + "/", str(root / "jobs") + "/", 1)
                      for value in job["argv"] if value.startswith(str(archive / "jobs") + "/")}
        output_ref.update(substitutions)
        argv = [output_ref.get(value, value) for value in job["argv"]]
        if "--deadline" in argv:
            argv[argv.index("--deadline") + 1] = str(DEADLINE)
        past = times[job["lane"]]
        eta = job["eta"]
        if job["kind"] == "gpu" and len(past) >= 3:
            eta = max(30, min(eta, int(median(past) * 1.6)))
        jobs.append(job | {"worker": str(new_worker), "argv": argv, "deps": deps,
                           "preferred": job.get("preferred", 0) % len(cfg["devices"]),
                           "priority": priority, "eta": eta})
        old_ids.add(name)

    selected_plans = (("plan-03-motif.json", 10), ("plan-01-information-depth.json", 20),
                      ("plan-00-amc-archived-depth.json", 30), ("plan-04-4k-nested-review.json", 31),
                      ("plan-02-seed-power-reserve.json", 90))
    for filename, rank in selected_plans:
        plan = plans[filename]
        for job in plan["jobs"]:
            add(job, plan["pins"], rank if job["kind"] == "gpu" else rank - 1)
    if any(not set(j["deps"]).issubset(old_ids) for j in jobs):
        raise ValueError("Missing dependency in resumed campaign")
    return {"pins": pins, "jobs": jobs}, verified, times


def launch(args):
    cfg, status, plans = archive_state(args.archive)
    if time.time() >= DEADLINE - 1800:
        raise TimeoutError("Insufficient authorized time for admission")
    cards = smi("--query-gpu=index,uuid,memory.used,utilization.gpu")
    processes = smi("--query-compute-apps=gpu_uuid,pid,used_memory")
    if [row[0] for row in cards] != list(map(str, range(8))) or [row[1] for row in cards] != cfg["devices"]:
        raise ValueError("Nemo host roster differs from the archive")
    assigned = cards[2:]
    devices = [row[1] for row in assigned]
    if any(int(row[2]) > 5 or row[3] != "0" for row in assigned) or any(row[0] in devices for row in processes):
        raise ValueError("The six selected devices are not all idle")
    if (args.root / "launch.json").exists():
        raise FileExistsError("Campaign was already launched")
    source = args.source
    queue = source / "repetition_followup_queue.py"
    if sha(queue) != args.queue_sha256:
        raise ValueError("Scoped scheduler source digest differs")
    env = {**os.environ, "PYTHONPATH": f"{source}:{cfg['runtime']}", "CUDA_VISIBLE_DEVICES": "",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    probe = subprocess.run([sys.executable, "-B", "-c",
                            "import torch,transformers,tokenizers;"
                            "from repetition_followup_baseline import runtime_for;"
                            "print(torch.__version__,transformers.__version__,tokenizers.__version__)"],
                           check=True, env=env, stdout=subprocess.PIPE, text=True)
    versions = probe.stdout.strip().splitlines()[-1]
    if not versions.startswith("2.10.0+cu129 5.3.0 0.22.2"):
        raise ValueError(f"Unexpected runtime versions: {versions}")
    full = cfg["devices"]
    cfg.update(devices=devices, host_devices=full, deadline=DEADLINE, access=DEADLINE - 1800,
               queue_sha256=args.queue_sha256, source_root=str(source), runtime=env["PYTHONPATH"])
    plan, verified, times = prepare(args, cfg, status, plans)
    root = args.root
    (root / "plans").mkdir(exist_ok=True)
    (root / "jobs").mkdir(exist_ok=True)
    plan_file = root / "plans" / "plan-00-resumed.json"
    save(plan_file, plan)
    save(root / "config.json", cfg)
    provenance = {"archive": str(args.archive), "archive_config_sha256": ARCHIVE_CONFIG,
                  "archive_status_sha256": sha(args.archive / "status.json"),
                  "verified_completed_dependencies": verified, "prelaunch_devices": assigned,
                  "versions": versions, "host": HOST, "deadline": DEADLINE,
                  "jobs": dict(Counter(j["kind"] + ":" + j["lane"] for j in plan["jobs"])),
                  "estimated_gpu_hours": round(sum(j["eta"] for j in plan["jobs"] if j["kind"] == "gpu") / 3600, 2),
                  "prior_median_gpu_seconds": {name: round(median(series), 2)
                                               for name, series in times.items() if series}}
    save(root / "resume-provenance.json", provenance)
    subprocess.run([sys.executable, "-B", str(queue), "launch", "--root", str(root),
                    "--config-sha256", sha(root / "config.json")], check=True, env=env)
    print(json.dumps(provenance), flush=True)


def snapshot(root):
    current = root / "status.json"
    state = load(current)
    jobs = state["jobs"]
    owned = {str(mark["pid"]): name for name, mark in jobs.items()
             if mark["stage"] == "running" and mark["kind"] == "gpu"}
    live = [{"id": name, "pid": j["pid"], "device": j["device"], "started": j["started"]}
            for name, j in jobs.items() if j["stage"] == "running" and j["kind"] == "gpu"]
    devices = load(root / "config.json")["devices"]
    apps = [row + [owned.get(row[1])] for row in smi("--query-compute-apps=gpu_uuid,pid,used_memory")
            if row[0] in devices]
    print(json.dumps({"observed": time.time(), "updated": state["updated"], "state": state["state"],
                      "gpu_workers": live, "compute": apps, "manifest_shas": state["manifest_shas"],
                      "by_lane": {lane: dict(Counter(j["stage"] for j in jobs.values() if j["lane"] == lane))
                                  for lane in sorted({j["lane"] for j in jobs.values()})},
                      "failures": {name: j.get("reason") for name, j in jobs.items() if j["stage"] == "failed"},
                      "status_sha256": sha(current)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inspect", "launch", "snapshot"))
    for flag in ("archive", "root", "source"):
        parser.add_argument("--" + flag, type=Path)
    parser.add_argument("--queue-sha256")
    args = parser.parse_args()
    if args.action == "snapshot":
        snapshot(args.root)
    elif args.action == "inspect":
        inspect(args)
    else:
        launch(args)


if __name__ == "__main__":
    main()
