"""Queue one bounded READ-boundary check after the late cache crossing."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from repetition_delayed_campaign import load, paths, preflight, save, sha
from repetition_delayed_timing_campaign import DEADLINE, PANEL, QUEUE, SELECTED
from repetition_delayed_transfer_campaign import LATE_WORKER, SUPPORT, completed
from repetition_delayed_transport_campaign import TIMING_WORKER, compile_jobs


PULSE_WORKER = "13e117b7a894ea30e8ef2111a0b87f58bb2c9586ec9bdeb927b913528561b982"


def launch(args):
    if not set(SELECTED) <= set(preflight()["idle"]):
        raise ValueError("Expected GPU 1 and 3 idle after the late crossing")
    if time.time() >= DEADLINE - (620 if args.count == 32 else 420):
        raise TimeoutError("Insufficient time for the boundary follow-up")
    base, root, source = args.base, args.root, args.source
    if root.parent != base / "runs" or source.parent != base / "tmp" or (root / "launch.json").exists():
        raise ValueError("Expected a fresh scoped WSP run and source")
    late = base / "runs" / "repetition-delayed-query-value-cross-offset17-20261126-v1"
    predecessor_sha, _ = completed(late)
    prior = load(late / "config.json")
    panel = base / "runs" / "repetition-delayed-query-pad-split-20261126-v1" / "panel.json"
    queue = source / "repetition_followup_queue.py"
    _, model, data = paths(base)
    pulse = args.mode == "pulse7"
    if pulse:
        label = "DQP7"
        worker = source / "repetition_delayed_transport.py"
        pins = {str(worker): PULSE_WORKER, str(source / "repetition_delayed_timing.py"): TIMING_WORKER}
        mode_args = ["--pulse-offset", "7"]
    else:
        label = "DQL7"
        worker = source / "repetition_delayed_value_cross.py"
        pins = {str(source / name): pin for name, pin in SUPPORT.items()} | {str(worker): LATE_WORKER}
        mode_args = ["--offset", "7"]
    for path, digest in (pins | {str(queue): QUEUE, str(panel): PANEL}).items():
        if sha(Path(path)) != digest:
            raise ValueError(f"Pinned bytes differ for {Path(path).name}")
    jobs = compile_jobs(root, worker, panel, model, data, label=label,
                        panel_args=["--panel", str(panel), "--panel-sha256", PANEL, *mode_args])
    summary = jobs.pop()
    eval_names = [f"{label}-eval-{start:03d}" for start in range(0, args.count, 8)]
    jobs = [job for job in jobs if job["lane"] == "canary" or job["id"] in eval_names]
    for job in jobs:
        job["eta"] = 90 if job["lane"] == "canary" else 180
    summary["deps"] = eval_names
    summary["argv"] = ["summarize", "--reports", *(str(root / "jobs" / f"{name}.json") for name in eval_names),
                       "--fingerprints", *(f"{{sha:{name}}}" for name in eval_names)]
    reference_shas = []
    if not pulse:
        early = base / "runs" / "repetition-delayed-query-value-cross-20261126-v1"
        _, status = completed(early)
        references = []
        for start in range(0, args.count, 8):
            name = f"DQX-eval-{start:03d}"
            job = status["jobs"][name]
            path = early / "jobs" / f"{name}.json"
            if job["stage"] != "completed" or job["sha256"] != sha(path):
                raise ValueError(f"Unverified paired early shard {name}")
            references.append(str(path))
            reference_shas.append(job["sha256"])
        summary["argv"].extend(["--reference-reports", *references, "--reference-fingerprints", *reference_shas])
    summary["argv"].extend(["--output", str(root / "jobs" / f"{label}-eval-summary.json")])
    jobs.append(summary)
    cfg = prior | {"source_root": str(source), "runtime": f"{source}:{prior['runtime']}", "deadline": DEADLINE,
                   "access": DEADLINE - 120, "queue_sha256": QUEUE, "admission_buffer_seconds": 65}
    (root / "plans").mkdir()
    plan = root / "plans" / f"plan-00-{args.mode}.json"
    save(plan, {"jobs": jobs, "pins": pins})
    save(root / "config.json", cfg)
    seal = {"mode": args.mode, "panel_sha256": PANEL, "count": args.count, "deadline": DEADLINE,
            "predecessor_release_sha256": predecessor_sha, "reference_shas": reference_shas,
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
    p.add_argument("--mode", choices=("pulse7", "cross7"), required=True)
    p.add_argument("--count", choices=(16, 32), type=int, required=True)
    launch(p.parse_args())


if __name__ == "__main__":
    main()
