"""Run an independent delayed-READ holdout after the cache probe sequence."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from repetition_delayed_campaign import DEVICES, HOST, load, paths, preflight, save, sha
from repetition_delayed_timing_campaign import DEADLINE, HELPER, PANEL, QUEUE, SELECTED, compile_jobs as timing_jobs


TIMING = "6e82ced4020abdcd9c2900ae36097a8374216dd8815747f9ef344371c88bb9ee"
TRANSPORT = "ea70326d0c2247718b57792783d90e6c84be9370cda0159260254aa93463e51d"
RESTORE = "c4b1512260b12dffa63ce9daedf4e9a428a344618af66a6eeb342075cc7e8ec2"


def restore_jobs(root, worker, panel, fingerprint, model, data, count):
    out = root / "jobs"
    jobs = []

    def add(name, lane, kind, arguments, deps, eta, preferred=0):
        jobs.append({"id": name, "lane": lane, "kind": kind, "worker": str(worker),
                     "extension": ".json", "argv": [*arguments, "--output", str(out / f"{name}.json")],
                     "deps": deps, "priority": 0 if lane == "canary" else 10,
                     "eta": eta, "preferred": preferred})
        return name

    def shard(split, start, size, dependencies, preferred):
        canary = split == "canary"
        return add(f"DQP-{split}-{start:03d}", split, "gpu", [
            "run", "--panel", str(panel), "--panel-sha256", fingerprint, "--model", str(model),
            "--data", str(data), "--split", split, "--anchor", "7", "--start", str(start),
            "--count", str(size), "--draws", "1" if canary else "2", "--cap", "128" if canary else "512",
            "--deadline", str(DEADLINE),
        ], dependencies, 120 if canary else 470, preferred)

    canaries = [shard("canary", i, 1, [], i) for i in range(2)]
    results = [shard("eval", start, 8, canaries, i % 2) for i, start in enumerate(range(0, count, 8))]
    add("DQP-eval-summary", "eval", "cpu", [
        "summarize", "--reports", *(str(out / f"{name}.json") for name in results),
        "--fingerprints", *(f"{{sha:{name}}}" for name in results),
    ], results, 35)
    return jobs


def launch(args):
    live = preflight()
    if not set(SELECTED) <= set(live["idle"]):
        raise ValueError("Expected GPU 1 and 3 idle after the cache probes")
    minimum = 1740 if args.mode == "timing" else 2050 if args.count == 64 else 1220
    if time.time() >= DEADLINE - minimum:
        raise TimeoutError("Insufficient time for the selected independent holdout")
    base, root, source = args.base, args.root, args.source
    if root.parent != base / "runs" or source.parent != base / "tmp" or (root / "launch.json").exists():
        raise ValueError("Expected a fresh scoped WSP run and source")
    previous_slug = ("repetition-delayed-query-value-cross-20261126-v1" if args.mode == "timing"
                     else "repetition-delayed-query-replication-20261126-v1")
    previous = base / "runs" / previous_slug
    status_path, release_path = previous / "status.json", previous / "release.json"
    release, prior = load(release_path), load(previous / "config.json")
    devices = [DEVICES[i] for i in SELECTED]
    if (release["host"] != HOST or release["state"] != "completed" or release["campaign_compute"]
            or release["status_sha256"] != sha(status_path) or prior["devices"] != devices
            or [row[1] for row in release["allocation"]] != devices):
        raise ValueError("Expected verified predecessor completion and exact device release")
    panel, queue = root / "panel.json", source / "repetition_followup_queue.py"
    if sha(panel) != args.panel_sha256 or load(panel)["source_panel_sha256"] != PANEL:
        raise ValueError("Independent panel byte digest or archive lineage differs")
    _, model, data = paths(base)
    timing = source / "repetition_delayed_timing.py"
    pins = {str(timing): TIMING}
    if args.mode == "timing":
        if args.count != 64:
            raise ValueError("Timing replication uses the complete balanced holdout")
        jobs = timing_jobs(root, timing, panel, model, data)
        for job in jobs:
            job["argv"] = [args.panel_sha256 if word == PANEL else word for word in job["argv"]]
    else:
        worker = source / "repetition_delayed_restore_rollout.py"
        pins.update({str(worker): RESTORE, str(source / "repetition_delayed_transport.py"): TRANSPORT})
        jobs = restore_jobs(root, worker, panel, args.panel_sha256, model, data, args.count)
    support = {str(queue): QUEUE, str(source / "repetition_delayed_campaign.py"): HELPER}
    for path, digest in (pins | support).items():
        if sha(Path(path)) != digest:
            raise ValueError(f"Pinned bytes differ for {Path(path).name}")
    cfg = prior | {"source_root": str(source), "runtime": f"{source}:{prior['runtime']}",
                   "deadline": DEADLINE, "access": DEADLINE - 120, "queue_sha256": QUEUE,
                   "admission_floor_seconds": 45, "admission_buffer_seconds": 90}
    (root / "plans").mkdir()
    plan = root / "plans" / f"plan-00-independent-{args.mode}.json"
    save(plan, {"jobs": jobs, "pins": pins})
    save(root / "config.json", cfg)
    seal = {"deadline": DEADLINE, "bridge_cutoff": 1795642200, "devices": SELECTED,
            "panel_sha256": args.panel_sha256, "count": args.count, "mode": args.mode,
            "predecessor_release_sha256": sha(release_path), "manifest_shas": {plan.name: sha(plan)},
            "jobs": dict(Counter(j["lane"] + ":" + j["kind"] for j in jobs))}
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
    p.add_argument("--panel-sha256", required=True)
    p.add_argument("--mode", choices=("timing", "restore"), required=True)
    p.add_argument("--count", choices=(32, 64), type=int, required=True)
    launch(p.parse_args())


if __name__ == "__main__":
    main()
