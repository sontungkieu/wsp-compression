"""Queue the paired value-state factorial after the numeric probe releases."""
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
from repetition_delayed_transport_campaign import TIMING_WORKER, compile_jobs


SUPPORT = {
    "repetition_delayed_numeric.py": "d2dde8c0e8b76992654c085945e03c858c7ed03165bf5ca10d65453fbdc3abe3",
    "repetition_delayed_transport.py": "ea70326d0c2247718b57792783d90e6c84be9370cda0159260254aa93463e51d",
    "repetition_delayed_timing.py": TIMING_WORKER,
}


def launch(args):
    live = preflight()
    if not set(SELECTED) <= set(live["idle"]):
        raise ValueError("Expected both assigned GPUs available after numeric scoring")
    if time.time() >= DEADLINE - 1200:
        raise TimeoutError("Insufficient time for the full value-state factorial")
    root, source, base = args.root, args.source, args.base
    if root.parent != base / "runs" or source.parent != base / "tmp" or (root / "launch.json").exists():
        raise ValueError("Expected fresh scoped WSP run and source roots")
    predecessor = base / "runs" / "repetition-delayed-query-numeric-20261126-v1"
    release_path = predecessor / "release.json"
    release, prior = load(release_path), load(predecessor / "config.json")
    if (release["status_sha256"] != sha(predecessor / "status.json") or release["campaign_compute"]
            or release["host"] != HOST or release["state"] != "completed"
            or prior["devices"] != [DEVICES[i] for i in SELECTED]):
        raise ValueError("Expected the numeric probe to complete and release both GPUs")
    panel = base / "runs" / "repetition-delayed-query-pad-split-20261126-v1" / "panel.json"
    worker = source / "repetition_delayed_value_cross.py"
    queue = source / "repetition_followup_queue.py"
    pins = {str(worker): args.worker_sha256} | {str(source / name): pin for name, pin in SUPPORT.items()}
    for path, pin in pins.items() | {(str(panel), PANEL), (str(queue), QUEUE)}:
        if sha(Path(path)) != pin:
            raise ValueError(f"Pinned bytes differ for {Path(path).name}")
    _, model, data = paths(base)
    cfg = prior | {"source_root": str(source), "runtime": f"{source}:{prior['runtime']}",
                   "deadline": DEADLINE, "access": DEADLINE - 120, "queue_sha256": QUEUE,
                   "admission_floor_seconds": 45, "admission_buffer_seconds": 100}
    jobs = compile_jobs(root, worker, panel, model, data, label="DQX")
    (root / "plans").mkdir()
    plan = root / "plans" / "plan-00-value-state-factorial.json"
    save(plan, {"jobs": jobs, "pins": pins})
    save(root / "config.json", cfg)
    seal = {"deadline": DEADLINE, "bridge_cutoff": 1795642200, "devices": SELECTED, "panel_sha256": PANEL,
            "numeric_release_sha256": sha(release_path), "numeric_status_sha256": release["status_sha256"],
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
