"""Run a short matrix-only restoration control after the cache probes release."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from repetition_delayed_campaign import load, paths, preflight, save, sha
from repetition_delayed_replication_campaign import TIMING, TRANSPORT, restore_jobs
from repetition_delayed_timing_campaign import DEADLINE, QUEUE, SELECTED
from repetition_delayed_transfer_campaign import completed


PANEL = "490603ed38a679f4a1220147ac5abee2e0fa580b25646d29ff04e990024bbf53"
WORKER = "a6f0f08f2074ddda59535150176b62a63fa118fb7a4f7db0234060ea0392d72c"


def launch(args):
    if not set(SELECTED) <= set(preflight()["idle"]):
        raise ValueError("Expected GPU 1 and 3 idle before matrix control")
    if time.time() >= DEADLINE - (480 if args.second_block else 600):
        raise TimeoutError("Insufficient time for the selected matrix-control block")
    base, root, source = args.base, args.root, args.source
    if root.parent != base / "runs" or source.parent != base / "tmp" or (root / "launch.json").exists():
        raise ValueError("Expected fresh scoped WSP run and source")
    old = base / "runs" / "repetition-delayed-query-restore-holdout-20261126-v1"
    old_release, _ = completed(old)
    previous = next((base / "runs" / f"repetition-delayed-query-{suffix}-20261126-v1" for suffix in
                     ("pulse7", "cross7", "value-cross-offset17") if
                     (base / "runs" / f"repetition-delayed-query-{suffix}-20261126-v1" / "launch.json").exists()), None)
    if previous is None:
        raise ValueError("Missing late-read campaign release")
    last_release, _ = completed(previous)
    first_block = base / "runs" / "repetition-delayed-query-matrix-control-20261126-v1"
    prior = load(old / "config.json")
    panel, queue = old / "panel.json", source / "repetition_followup_queue.py"
    worker = source / "repetition_delayed_restore_rollout.py"
    pins = {str(worker): WORKER, str(source / "repetition_delayed_timing.py"): TIMING,
            str(source / "repetition_delayed_transport.py"): TRANSPORT}
    for path, digest in (pins | {str(queue): QUEUE, str(panel): PANEL}).items():
        if sha(Path(path)) != digest:
            raise ValueError(f"Pinned bytes differ for {Path(path).name}")
    _, model, data = paths(base)
    jobs = restore_jobs(root, worker, panel, PANEL, model, data, count=16)
    canary_pins = {}
    if args.second_block:
        last_release, first_status = completed(first_block)
        for index in range(2):
            name = f"DQP-canary-{index:03d}"
            path = first_block / "jobs" / f"{name}.json"
            receipt, canary = first_status["jobs"][name], load(path)
            if (receipt["stage"] != "completed" or receipt["output"] != str(path)
                    or receipt["sha256"] != sha(path) or canary["worker_sha"] != WORKER
                    or not canary["matrix_control"] or canary["panel_sha256"] != PANEL or canary["anchor"] != 7):
                raise ValueError(f"Unverified matrix-control canary {name}")
            canary_pins[name] = receipt["sha256"]
        jobs = jobs[2:]
        names = {"DQP-eval-000": "DQP-eval-016", "DQP-eval-008": "DQP-eval-024"}
        def rename(value):
            for old_name, new_name in names.items():
                value = value.replace(old_name, new_name)
            return value
        for job in jobs:
            job["id"] = rename(job["id"])
            job["argv"] = list(map(rename, job["argv"]))
            job["deps"] = list(map(rename, job["deps"]))
            if job["kind"] == "gpu":
                start = job["argv"].index("--start") + 1
                job["argv"][start] = str(int(job["argv"][start]) + 16)
                job["deps"] = []
                job["eta"] = 390
    for job in jobs:
        if job["kind"] == "gpu":
            job["argv"].append("--matrix-control")
    cfg = prior | {"source_root": str(source), "runtime": f"{source}:{prior['runtime']}", "deadline": DEADLINE,
                   "access": DEADLINE - 120, "queue_sha256": QUEUE, "admission_buffer_seconds": 65}
    (root / "plans").mkdir()
    plan = root / "plans" / "plan-00-matrix-control.json"
    save(plan, {"jobs": jobs, "pins": pins})
    save(root / "config.json", cfg)
    seal = {"mode": "matrix-only-control", "deadline": DEADLINE, "panel_sha256": PANEL, "count": 16,
            "start": 16 if args.second_block else 0, "reused_canary_shas": canary_pins,
            "restore_release_sha256": old_release, "predecessor_release_sha256": last_release,
            "manifest_shas": {plan.name: sha(plan)}}
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
    p.add_argument("--second-block", action="store_true")
    launch(p.parse_args())


if __name__ == "__main__":
    main()
