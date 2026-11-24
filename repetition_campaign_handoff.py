"""Record a sealed campaign and defer a Huy follow-on until verified release."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

from repetition_campaign_resume import (
    AGGREGATE, DEADLINE, FOLLOWUP_AGGREGATE, copy_pinned, load, save, sha, smi,
)


HUY = "huyhq21-membedding-v2-1-0-2"
HUY_ARCHIVE_CONFIG = "eac998dffafa2f40c6a3412571efa23c3b75db6d77f441e927d775080dea0433"
HUY_ARCHIVE_CUTOFF = 1795566300
GUARD_PINS = {
    "repetition_motif_guard.py": "edf8457087a1a88b4ed781768ac58aba0fb95b48e4ade41bc28076e2e3727bd8",
    "repetition_mediation.py": "3159907371a3539b03f7c78c54e4bc6faaa9f6e82c6b6943e534def0505653f7",
}
ENGAGEMENT = "58f25d4262f0fe00db924fee7e28fb89d9295384ee43e48d88b8a6a2f5e0f88f"


def configure_env(config):
    return {**os.environ, "PYTHONPATH": config["runtime"], "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}


def start_report(root, source):
    cfg = load(root / "config.json")
    output = root / "campaign-summary.json"
    report = source / "repetition_depth_aggregate.py"
    if sha(report) != AGGREGATE or cfg["host"] != socket.gethostname() or cfg["deadline"] != DEADLINE:
        raise ValueError("Unexpected report host, cutoff or source")
    if (root / "aggregate-launch.json").exists():
        raise FileExistsError("Reporter already launched")
    with (root / "aggregate.log").open("x") as log:
        proc = subprocess.Popen([sys.executable, "-B", str(report), "--root", str(root),
                                 "--output", str(output), "--watch"], env=configure_env(cfg),
                                start_new_session=True, stdout=log, stderr=subprocess.STDOUT)
    receipt = {"pid": proc.pid, "worker_sha256": AGGREGATE, "output": str(output), "deadline": DEADLINE}
    save(root / "aggregate-launch.json", receipt)
    return receipt


def report_sources(source, reference):
    for name, fingerprint in (("repetition_depth_aggregate.py", AGGREGATE),
                              ("repetition_followup_aggregate.py", FOLLOWUP_AGGREGATE)):
        copy_pinned(reference / name, source / name, fingerprint)


def seal(root):
    deadline = load(root / "config.json")["deadline"]
    manifests = {p.name: sha(p) for p in sorted((root / "plans").glob("plan-*.json"))}
    if not manifests or deadline != DEADLINE:
        raise ValueError("No approved manifests to seal")
    pin = {"manifest_shas": manifests, "deadline": deadline}
    save(root / "sealed.json", pin)
    return pin | {"seal_sha256": sha(root / "sealed.json")}


def guard_nemo(args):
    source, root, archive = args.source, args.root, args.archive
    conf = load(root / "config.json")
    if (conf["host"] != "nemo-rl-env-0-0" or conf["host"] != socket.gethostname()
            or len(conf["devices"]) != 6 or conf["deadline"] != DEADLINE):
        raise ValueError("Guard requires the Nemo successor on six assigned devices")
    if (root / "sealed.json").exists():
        raise FileExistsError("Campaign is already sealed")
    old_path = archive / "plans" / "plan-03-motif.json"
    if sha(old_path) != load(archive / "status.json")["manifest_shas"][old_path.name]:
        raise ValueError("Archived motif plan digest differs")
    templates = load(old_path)["jobs"]
    guard = source / "guard"
    guard.mkdir(exist_ok=True)
    for name, fingerprint in GUARD_PINS.items():
        copy_pinned(args.reference / "guard" / name, guard / name, fingerprint)
    gate = source / "check_guard_engagement.py"
    copy_pinned(args.reference / gate.name, gate, ENGAGEMENT)
    worker = str(guard / "repetition_motif_guard.py")
    pins = {str(guard / name): fingerprint for name, fingerprint in GUARD_PINS.items()}
    pins[str(gate)] = ENGAGEMENT
    jobs = []
    canary = "GUARD-canary"
    review = canary + "-review"
    engagement = canary + "-engaged"
    for job in templates:
        is_canary = "canary" in job["id"]
        name = job["id"].replace("MOTIF-", "GUARD-", 1)
        score = job["kind"] == "cpu"
        deps = [d.replace("MOTIF-", "GUARD-", 1) for d in job["deps"]]
        argv = [x.replace("{sha:MOTIF-", "{sha:GUARD-", 1).replace(
            str(archive / "jobs") + "/MOTIF-", str(root / "jobs") + "/GUARD-", 1) for x in job["argv"]]
        if not score:
            argv[argv.index("--deadline") + 1] = str(DEADLINE)
            if not is_canary:
                argv[argv.index("--cap") + 1] = "2048"
                if "--draw-start" in argv:
                    flag(argv, "--draw-start", 60)
                else:
                    argv.extend(("--draw-start", "60"))
                deps = [engagement]
        jobs.append(job | {"id": name, "lane": "GUARD", "worker": worker, "argv": argv,
                           "deps": deps, "preferred": job.get("preferred", 0) % 6,
                           "eta": 110 if is_canary else job["eta"],
                           "priority": 22 if is_canary else 25})
    jobs.append({"id": engagement, "lane": "GUARD", "kind": "cpu", "worker": str(gate),
                 "extension": ".json", "argv": ["--score", str(root / "jobs" / f"{review}.json"),
                  "--fingerprint", f"{{sha:{review}}}", "--output", str(root / "jobs" / f"{engagement}.json")],
                 "deps": [review], "priority": 22, "eta": 30, "preferred": 0})
    plan = root / "plans" / "plan-01-guard-amc.json"
    save(plan, {"pins": pins, "jobs": jobs})
    report_sources(source, args.reference)
    print(json.dumps({"seal": seal(root), "report": start_report(root, source),
                      "guard_jobs": len(jobs)}), flush=True)


def flag(argv, name, new):
    argv[argv.index(name) + 1] = str(new)


def huy_preflight(cfg):
    cards = smi("--query-gpu=index,uuid,memory.used,utilization.gpu")
    if socket.gethostname() != HUY or [r[1] for r in cards] != cfg["devices"] or len(cards) != 4:
        raise ValueError("Huy host or full device order changed")
    devices = set(cfg["devices"])
    busy = [r for r in smi("--query-compute-apps=gpu_uuid,pid,used_memory") if r[0] in devices]
    return cards, busy


def read_huy(args):
    archive = args.archive
    if sha(archive / "config.json") != HUY_ARCHIVE_CONFIG:
        raise ValueError("Unexpected Huy predecessor configuration digest")
    old = load(archive / "config.json")
    state = load(archive / "status.json")
    cards, busy = huy_preflight(old)
    print(json.dumps({
        "host": old["host"], "state": state["state"], "updated": state["updated"],
        "status_sha256": sha(archive / "status.json"), "gpu": cards, "compute": busy,
        "counts": {lane: dict(Counter(j["stage"] for j in state["jobs"].values() if j["lane"] == lane))
                   for lane in sorted({j["lane"] for j in state["jobs"].values()})},
        "release_present": (archive / "release.json").is_file(),
        "manifest_shas": state["manifest_shas"],
    }), flush=True)
    return old, state


def defer_huy(args):
    old, state = read_huy(args)
    root, archive, source = args.root, args.archive, args.source
    original = Path(old["source_root"])
    if old["deadline"] != HUY_ARCHIVE_CUTOFF or time.time() >= DEADLINE - 1800:
        raise ValueError("Huy predecessor deadline or extension window differs")
    queue = source / "repetition_followup_queue.py"
    copy_pinned(original / queue.name, queue, old["queue_sha256"])
    report_sources(source, original)
    cfg = old | {"access": DEADLINE - 1800, "deadline": DEADLINE,
                 "source_root": str(source), "runtime": f"{source}:{old['runtime']}"}
    jobs, pins = [], {}

    def add(job, new_ids, plan, rank, new_draw=None):
        name = new_ids[job["id"]]
        worker = source / Path(job["worker"]).name
        fingerprint = plan["pins"][job["worker"]]
        copy_pinned(Path(job["worker"]), worker, fingerprint)
        pins[str(worker)] = fingerprint
        argv = []
        for item in job["argv"]:
            if item.startswith(f"{archive}/jobs/"):
                path = Path(item)
                item = str(root / "jobs" / (new_ids[path.stem] + path.suffix))
            match = re.fullmatch(r"\{sha:([^}]+)\}", item)
            if match:
                item = f"{{sha:{new_ids[match[1]]}}}"
            argv.append(item)
        eta = job["eta"]
        if job["kind"] == "gpu":
            flag(argv, "--deadline", DEADLINE)
            if new_draw is None:
                flag(argv, "--draw-start", 80)
                flag(argv, "--draws", 1)
                flag(argv, "--cap", 2048)
                eta = min(eta, 900)
            else:
                flag(argv, "--draw", new_draw)
        deps = [new_ids[d] for d in job["deps"] if d in new_ids]
        jobs.append(job | {"id": name, "worker": str(worker), "argv": argv, "deps": deps,
                           "priority": rank, "eta": eta})

    def previous(name):
        path = archive / "plans" / name
        if sha(path) != state["manifest_shas"][name]:
            raise ValueError(f"Huy predecessor manifest differs: {name}")
        return load(path)

    motif = previous("plan-02-amc-extended-motif.json")
    canary = state["jobs"]["MOTIF-amc-ext-canary-review"]
    if canary["stage"] != "completed" or sha(Path(canary["output"])) != canary["sha256"]:
        raise ValueError("Verified Huy AMC replay canary is missing")
    effect = [j for j in motif["jobs"] if "canary" not in j["id"]]
    effect_ids = {j["id"]: j["id"].replace("MOTIF-", "MOTIF-r80-", 1) for j in effect}
    for job in effect:
        add(job, effect_ids, motif, 10)
    standby = previous("plan-00-independent-heldout-draws.json")
    baseline = [(j, int(j["argv"][j["argv"].index("--draw") + 1])) for j in standby["jobs"]
                if j["kind"] == "gpu"]
    existing = {j["id"]: j for j in standby["jobs"]}
    for round_index, offset in enumerate((600, 700, 800, 900)):
        ids = {}
        draws = {}
        for job, previous_draw in baseline:
            name = job["id"]
            draw = previous_draw + offset
            ids[name] = name.replace(f"-d{previous_draw}-", f"-d{draw}-", 1)
            ids[name + "-score"] = ids[name] + "-score"
            draws[name] = draw
        for name, renamed in ids.items():
            job = existing[name]
            add(job, ids, standby, 30 + round_index, draws.get(name) if job["kind"] == "gpu" else None)
    if len({j["id"] for j in jobs}) != len(jobs):
        raise ValueError("Huy successor job names collide")

    save(root / "config.json", cfg)
    (root / "plans").mkdir()
    save(root / "plans" / "plan-00-huy-followon.json", {"pins": pins, "jobs": jobs})
    manifest = seal(root)
    cmd = [sys.executable, "-B", str(Path(__file__)), "wait-huy", "--root", str(root),
           "--source", str(source), "--archive", str(archive)]
    with (root / "defer.log").open("x") as log:
        proc = subprocess.Popen(cmd, start_new_session=True, stdout=log, stderr=subprocess.STDOUT)
    receipt = {"gate_pid": proc.pid, "deadline": DEADLINE, "host": HUY, "devices": cfg["devices"],
               "archive": str(archive), "requires_verified_release": True, "seal": manifest,
               "jobs": dict(Counter(j["lane"] + ":" + j["kind"] for j in jobs)),
               "nominal_gpu_hours": round(sum(j["eta"] for j in jobs if j["kind"] == "gpu") / 3600, 2)}
    save(root / "defer-launch.json", receipt)
    print(json.dumps(receipt), flush=True)


def wait_huy(args):
    root, archive = args.root, args.archive
    cfg = load(root / "config.json")
    release = archive / "release.json"
    while time.time() < cfg["access"]:
        cards, busy = huy_preflight(cfg)
        if release.is_file() and not busy and all(int(row[2]) <= 5 and row[3] == "0" for row in cards):
            evidence = load(release)
            status = archive / "status.json"
            if (evidence["status_sha256"] != sha(status) or evidence["host"] != HUY
                    or evidence["cutoff_epoch"] != HUY_ARCHIVE_CUTOFF or evidence["campaign_compute"]
                    or [row[1] for row in evidence["allocation"]] != cfg["devices"]):
                raise ValueError("Previous campaign did not provide a valid release receipt")
            break
        time.sleep(15)
    else:
        save(root / "gate-result.json", {"state": "expired", "deadline": DEADLINE})
        return
    registration = load(root / "sealed.json")
    for name, fingerprint in registration["manifest_shas"].items():
        if sha(root / "plans" / name) != fingerprint:
            raise ValueError("Successor manifest seal changed")
    queue = args.source / "repetition_followup_queue.py"
    if sha(queue) != cfg["queue_sha256"]:
        raise ValueError("Huy successor queue digest changed")
    subprocess.run([sys.executable, "-B", str(queue), "launch", "--root", str(root),
                    "--config-sha256", sha(root / "config.json")], check=True, env=configure_env(cfg))
    while not (root / "status.json").is_file() and time.time() < cfg["access"]:
        time.sleep(2)
    report = start_report(root, args.source)
    save(root / "gate-result.json", {"state": "launched", "launch": load(root / "launch.json"),
                                     "archive_release_sha256": sha(release), "report": report})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = {"finalize-nemo": guard_nemo, "read-huy": read_huy,
               "defer-huy": defer_huy, "wait-huy": wait_huy}
    parser.add_argument("action", choices=actions)
    for name in ("root", "archive", "source", "reference"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    try:
        actions[args.action](args)
    except Exception as error:
        if args.action == "wait-huy":
            save(args.root / "gate-result.json", {"state": "failed", "error": str(error)})
        raise


if __name__ == "__main__":
    main()
