"""Run the fixed-query cache-restore probe once the timing GPUs have released."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from repetition_delayed_campaign import DEVICES, HOST, load, paths, preflight, save, sha
from repetition_delayed_timing_campaign import DEADLINE, PANEL, QUEUE, SELECTED


TIMING_WORKER = "6e82ced4020abdcd9c2900ae36097a8374216dd8815747f9ef344371c88bb9ee"


def compile_jobs(root, worker, panel, model, data):
    jobs = []
    out = root / "jobs"

    def add(name, lane, kind, argv, deps, eta, preferred=0):
        jobs.append({"id": name, "lane": lane, "kind": kind, "worker": str(worker),
                     "extension": ".json", "argv": [*argv, "--output", str(out / f"{name}.json")],
                     "deps": deps, "priority": 0 if lane == "canary" else 10,
                     "eta": eta, "preferred": preferred})

    def shard(split, start, count, needs, preferred):
        name = f"DQR-{split}-{start:03d}"
        add(name, split, "gpu", ["run", "--panel", str(panel), "--panel-sha256", PANEL,
            "--model", str(model), "--data", str(data), "--split", split, "--start", str(start),
            "--count", str(count), "--deadline", str(DEADLINE)], needs, 120 if split == "canary" else 300, preferred)
        return name

    canaries = [shard("canary", index, 1, [], index) for index in range(2)]
    scores = [shard("eval", start, 8, canaries, index % 2) for index, start in enumerate(range(0, 64, 8))]
    add("DQR-eval-summary", "eval", "cpu",
        ["summarize", "--reports", *(str(out / f"{name}.json") for name in scores),
         "--fingerprints", *(f"{{sha:{name}}}" for name in scores)], scores, 30)
    return jobs


def launch(args):
    live = preflight()
    if not set(SELECTED) <= set(live["idle"]):
        raise ValueError("Expected both assigned GPUs to be idle after timing")
    if time.time() >= DEADLINE - 700:
        raise TimeoutError("Insufficient time for the canary and restoration probe")
    root, source, base = args.root, args.source, args.base
    if root.parent != base / "runs" or source.parent != base / "tmp" or (root / "launch.json").exists():
        raise ValueError("Expected fresh scoped WSP run and source roots")
    timing = base / "runs" / "repetition-delayed-query-reset-offset-20261126-v1"
    panel = timing / "panel.json"
    release_path = timing / "release.json"
    release, prior = load(release_path), load(timing / "config.json")
    devices = [DEVICES[i] for i in SELECTED]
    if (release["status_sha256"] != sha(timing / "status.json") or release["campaign_compute"]
            or release["host"] != HOST or release["state"] != "completed" or prior["devices"] != devices):
        raise ValueError("Expected a completed timing campaign and a verified release")
    worker = source / "repetition_delayed_transport.py"
    queue = source / "repetition_followup_queue.py"
    for path, expected in ((worker, args.worker_sha256), (queue, QUEUE), (panel, PANEL),
                           (source / "repetition_delayed_timing.py", TIMING_WORKER)):
        if sha(path) != expected:
            raise ValueError(f"Pinned bytes differ for {path.name}")
    _, model, data = paths(base)
    cfg = prior | {"source_root": str(source), "runtime": f"{source}:{prior['runtime']}",
                   "deadline": DEADLINE, "access": DEADLINE - 120, "queue_sha256": QUEUE,
                   "admission_floor_seconds": 45, "admission_buffer_seconds": 70}
    jobs = compile_jobs(root, worker, panel, model, data)
    (root / "plans").mkdir()
    plan = root / "plans" / "plan-00-restore-factorial.json"
    save(plan, {"jobs": jobs, "pins": {str(worker): args.worker_sha256,
                                      str(source / "repetition_delayed_timing.py"): TIMING_WORKER}})
    save(root / "config.json", cfg)
    seal = {"deadline": DEADLINE, "bridge_cutoff": 1795642200, "devices": SELECTED, "panel_sha256": PANEL,
            "timing_release_sha256": sha(release_path), "timing_status_sha256": release["status_sha256"],
            "manifest_shas": {plan.name: sha(plan)}, "jobs": dict(Counter(j["lane"] + ":" + j["kind"] for j in jobs))}
    save(root / "sealed.json", seal)
    env = {**os.environ, "PYTHONPATH": cfg["runtime"], "CUDA_VISIBLE_DEVICES": "",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    subprocess.run([sys.executable, "-B", str(queue), "launch", "--root", str(root),
                    "--config-sha256", sha(root / "config.json")], check=True, env=env)
    print(json.dumps(seal | {"seal_sha256": sha(root / "sealed.json")}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "root", "source"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--worker-sha256", required=True)
    launch(p.parse_args())


if __name__ == "__main__":
    main()
