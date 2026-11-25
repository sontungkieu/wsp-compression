"""Pin, run, and audit the delayed-query experiment on the allotted B200s."""
import argparse
from collections import Counter
import hashlib
import importlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


HOST = "general-embedding-training-4-0-1"
DEVICES = [
    "GPU-9679efe6-954f-ae1d-ed1f-1544cc67114e",
    "GPU-30b4be09-0d9f-2649-9cfa-6a356742a2fe",
    "GPU-505ae3dd-e8f2-814e-4402-81835980c007",
    "GPU-52333868-e7f4-2ca4-0737-63189b81ebad",
]
PREDECESSORS = (
    "repetition-huy-guard-20261124-v1",
    "repetition-huy-extension-20261125-v1",
    "repetition-nemo-resume-20261125-v1",
)
DURATION = 3 * 3600


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(path.read_text())


def save(path, data):
    pending = path.with_suffix(".incoming")
    with pending.open("x") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
    if path.exists():
        raise FileExistsError(path)
    pending.replace(path)


def smi(selector):
    return [line.split(", ") for line in subprocess.check_output(
        ["nvidia-smi", selector, "--format=csv,noheader,nounits"],
        text=True, timeout=10).splitlines() if line.strip()]


def preflight():
    cards = smi("--query-gpu=index,uuid,memory.used,utilization.gpu")
    compute = smi("--query-compute-apps=gpu_uuid,pid,used_memory")
    if socket.gethostname() != HOST or [row[1] for row in cards] != DEVICES:
        raise ValueError("Host or ordered B200 roster differs from the allocation")
    busy = {row[0] for row in compute}
    return {"host": HOST, "cards": cards, "compute": compute,
            "idle": [int(i) for i, uuid, mem, util in cards if uuid not in busy and int(mem) <= 5 and util == "0"]}


def paths(base):
    artifact = base / "tmp" / "astra-qwen35-gsm8k-20261118-v1"
    return artifact, artifact / "model", artifact / "test-00000-of-00001.parquet"


def env_vars(source, base):
    predecessor = base / "runs" / PREDECESSORS[1]
    cfg = load(predecessor / "config.json")
    return {**os.environ, "PYTHONPATH": f"{source}:{cfg['runtime']}",
            "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "2"}


def abi(base):
    from importlib import metadata
    import torch
    import tokenizers
    import transformers
    from transformers.models.qwen3_5 import modeling_qwen3_5

    major = (transformers.__version__, tokenizers.__version__)
    if major != ("5.3.0", "0.22.2"):
        raise ValueError(f"Unexpected Transformers/tokenizers versions: {major}")
    artifact, model, data = paths(base)
    expected = load(artifact / "manifest.json")["artifacts"]
    model_paths = sorted((key for key in expected if key.startswith("model/"))) + [data.name]
    observed = {key: sha(artifact / key) for key in model_paths}
    if any(digest != expected[key]["sha256"] for key, digest in observed.items()):
        raise ValueError("A checkpoint/tokenizer/data artifact disagrees with its prior manifest")
    modules = ["repetition_state_value", "repetition_cache_swap", "repetition_sampler_reach",
               "repetition_followup_queue", "repetition_followup_baseline"]
    harness = {}
    for name in modules:
        injected = Path(importlib.import_module(name).__file__)
        harness[name] = {"path": str(injected), "sha256": sha(injected)}
    extras = {}
    for package in ("flash-linear-attention", "causal-conv1d"):
        try:
            extras[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            extras[package] = None
    print(json.dumps({"torch": torch.__version__, "torch_cuda": torch.version.cuda,
                      "transformers": major[0], "tokenizers": major[1], "packages": extras,
                      "qwen_fast_path": modeling_qwen3_5.is_fast_path_available,
                      "checkpoint": str(model), "artifacts": observed, "harness": harness}), flush=True)


def audit(base):
    outputs = {}
    for run in PREDECESSORS:
        root = base / "runs" / run
        state_path = root / "status.json"
        release_path = root / "release.json"
        score_path = root / "campaign-summary.json"
        if not all(p.is_file() for p in (state_path, release_path, score_path)):
            outputs[run] = {"available": False}
            continue
        state, release, scores = map(load, (state_path, release_path, score_path))
        if release["status_sha256"] != sha(state_path) or release["campaign_compute"]:
            raise ValueError(f"Archive release evidence differs for {run}")
        motif = {}
        for group, data in scores.get("motif", {}).items():
            cap = group.split("/")[2]
            window = data["windows"][cap]
            motif[group] = {"eligible": data["eligible"], "total": data["total"], "arms": {
                arm: {key: counts.get(key) for key in ("n", "correct_eos", "wrong_eos", "cap",
                                                       "terminal_exact", "original_motif_tail")}
                for arm, counts in window["arms"].items()}}
        info = {}
        for group, windows in scores.get("information_nested", {}).items():
            budget = str(max(map(int, windows)))
            info[group] = {"budget": budget, "eos_paired_both_values": windows[budget]["eos_paired_both_values"]}
        outputs[run] = {"available": True, "state": release["state"], "released_at": release["observed_ict"],
                        "status_sha256": release["status_sha256"], "summary_sha256": sha(score_path),
                        "release_sha256": sha(release_path), "host": release["host"],
                        "jobs": dict(Counter(mark["stage"] for mark in state["jobs"].values())),
                        "motif": motif, "information": info}
    print(json.dumps(outputs), flush=True)


def compile_jobs(worker, model, panel, data, root, deadline):
    jobs = []
    out = root / "jobs"
    common = ["--panel", str(panel), "--model", str(model)]

    def add(name, lane, kind, args, deps, extent, eta, preferred=0):
        target = str(out / (name + extent))
        jobs.append({"id": name, "lane": lane, "kind": kind, "worker": str(worker),
                     "extension": extent, "argv": [*args, "--output", target],
                     "deps": deps, "priority": {"canary": 0, "pilot": 10, "eval": 20}[lane],
                     "eta": eta, "preferred": preferred})

    def shard(split, start, count, draws, needs, preferred):
        name = f"DQ-{split}-{start:03d}"
        args = ["run", *common, "--data", str(data), "--split", split, "--start", str(start),
                "--count", str(count), "--draws", str(draws), "--cap", "64" if split == "canary" else "512",
                "--deadline", str(deadline)]
        add(name, split, "gpu", args, needs, ".jsonl", 240 if split == "canary" else 1000, preferred)
        review = name + "-score"
        add(review, split, "cpu", ["score", *common, "--raw", str(out / (name + ".jsonl")),
                                    "--fingerprint", f"{{sha:{name}}}"], [name], ".json", 90)
        return review

    def summary(split, parents):
        args = ["summarize", "--panel", str(panel), "--split", split,
                "--reports", *(str(out / (n + ".json")) for n in parents),
                "--fingerprints", *(f"{{sha:{n}}}" for n in parents)]
        name = f"DQ-{split}-summary"
        add(name, split, "cpu", args, parents, ".json", 120)
        return name

    canaries = [shard("canary", j, 1, 1, [], device) for j, device in enumerate((1, 3))]
    pilot = summary("pilot", [shard("pilot", j * 4, 4, 1, canaries, (1, 3, 0, 2)[j])
                              for j in range(4)])
    summary("eval", [shard("eval", j * 8, 8, 2, [pilot], (1, 3, 0, 2)[j % 4])
                     for j in range(8)])
    return jobs


def launch(args):
    observed = preflight()
    if not observed["idle"]:
        raise RuntimeError("No assigned GPUs are currently idle")
    base, root, source = args.base, args.root, args.source
    if ((root / "config.json").exists() or root.parent != base / "runs" or source.parent != base / "tmp"):
        raise ValueError("Expected a fresh WSP run and source root")
    worker = source / "repetition_delayed_query.py"
    queue = source / "repetition_followup_queue.py"
    if sha(worker) != args.worker_sha256 or sha(queue) != args.queue_sha256:
        raise ValueError("Worker or queue checksum differs")
    receipt = load(root / "preflight.json")
    _, model, data = paths(base)
    if receipt["abi"]["checkpoint"] != str(model) or receipt["inventory"]["host"] != HOST:
        raise ValueError("Preflight receipt is from a different host or checkpoint")
    for pinned in receipt["abi"]["harness"].values():
        if sha(Path(pinned["path"])) != pinned["sha256"]:
            raise ValueError("Pinned harness bytes changed after preflight")
    env = env_vars(source, base)
    panel = root / "panel.json"
    subprocess.run([sys.executable, "-B", str(worker), "freeze", "--model", str(model),
                    "--panel", str(panel)], check=True, env=env)
    deadline = int(time.time()) + DURATION
    cfg = {"host": HOST, "devices": DEVICES, "deadline": deadline, "access": deadline - 900,
           "source_root": str(source), "runtime": env["PYTHONPATH"], "queue_sha256": args.queue_sha256}
    jobs = compile_jobs(worker, model, panel, data, root, deadline)
    (root / "plans").mkdir()
    plan = root / "plans" / "plan-00-delayed-query.json"
    save(plan, {"jobs": jobs, "pins": {str(worker): args.worker_sha256}})
    save(root / "config.json", cfg)
    admission = {"deadline": deadline, "access": cfg["access"], "night_window_seconds": DURATION,
                 "panel_sha256": sha(panel), "preflight_sha256": sha(root / "preflight.json"),
                 "inventory_at_launch": observed, "manifest_shas": {plan.name: sha(plan)},
                 "jobs": dict(Counter(j["lane"] + ":" + j["kind"] for j in jobs)),
                 "estimated_gpu_hours": round(sum(j["eta"] for j in jobs if j["kind"] == "gpu") / 3600, 2)}
    save(root / "sealed.json", admission)
    subprocess.run([sys.executable, "-B", str(queue), "launch", "--root", str(root),
                    "--config-sha256", sha(root / "config.json")], check=True, env=env)
    print(json.dumps(admission | {"seal_sha256": sha(root / "sealed.json")}), flush=True)


def probe(args):
    observed = preflight()
    env = env_vars(args.source, args.base)
    proc = subprocess.run([sys.executable, "-B", __file__, "abi", "--base", str(args.base)],
                          check=True, env=env, text=True, stdout=subprocess.PIPE)
    resolved = json.loads(proc.stdout.strip().splitlines()[-1])
    value = {"observed": time.time(), "inventory": observed, "abi": resolved}
    save(args.root / "preflight.json", value)
    print(json.dumps(value), flush=True)


def snapshot(root):
    status = load(root / "status.json")
    print(json.dumps({"state": status["state"], "updated": status["updated"],
                      "deadline": status["deadline"], "inventory": preflight(),
                      "jobs": {key: {
                          k: v for k, v in value.items()
                          if k in {"stage", "pid", "device", "output", "sha256", "wall_seconds", "reason", "exit_code"}
                      } for key, value in status["jobs"].items() if value["stage"] != "queued"},
                      "manifests": status["manifest_shas"], "status_sha256": sha(root / "status.json"),
                      "release": load(root / "release.json") if (root / "release.json").is_file() else None}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("probe", "abi", "audit", "launch", "snapshot"))
    for opt in ("base", "root", "source"):
        parser.add_argument("--" + opt, type=Path)
    for opt in ("worker", "queue"):
        parser.add_argument("--" + opt + "-sha256")
    args = parser.parse_args()
    if args.action == "abi":
        abi(args.base)
    elif args.action == "audit":
        audit(args.base)
    elif args.action == "probe":
        probe(args)
    elif args.action == "snapshot":
        snapshot(args.root)
    else:
        launch(args)


if __name__ == "__main__":
    main()
