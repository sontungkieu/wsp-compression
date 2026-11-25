"""Compare crossed record values across checkpoint or query offset on paired IDs."""

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


MODEL = Path("/workspace/storage-shared/models/Qwen3.5-2B")
TOKENIZER = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
CHECKPOINT = "0f6dac51a0539b59f4aa15f7fc9d0d1ea4e776cc0a676338186ff0fdc2ffd2fc"
LATE_WORKER = "253bfba24d023dea78d76c81a97be50190b611b4ce5cb5007bc48fd6c12459a1"
SUPPORT = {
    "repetition_delayed_value_cross.py": "8aed350297b386e0a3d8705a3f9b6afa5f280f6363b1609091cc54d27b33a029",
    "repetition_delayed_numeric.py": "d2dde8c0e8b76992654c085945e03c858c7ed03165bf5ca10d65453fbdc3abe3",
    "repetition_delayed_transport.py": "ea70326d0c2247718b57792783d90e6c84be9370cda0159260254aa93463e51d",
    "repetition_delayed_timing.py": TIMING_WORKER,
}


def completed(root):
    release_path = root / "release.json"
    release, status = load(release_path), load(root / "status.json")
    devices = [DEVICES[i] for i in SELECTED]
    if (release["state"] != "completed" or status["state"] != "completed" or release["host"] != HOST
            or release["status_sha256"] != sha(root / "status.json") or release["campaign_compute"]
            or [row[1] for row in release["allocation"]] != devices):
        raise ValueError(f"Invalid completed predecessor release for {root.name}")
    return sha(release_path), status


def launch(args):
    if not set(SELECTED) <= set(preflight()["idle"]):
        raise ValueError("Expected both allotted GPUs to be idle")
    late = args.offset == 17
    minimum = (780 if args.count == 32 else 560) if late else (1000 if args.count == 32 else 670)
    if time.time() >= DEADLINE - minimum:
        raise TimeoutError("Insufficient time for the paired cache comparison")
    base, root, source = args.base, args.root, args.source
    if (root.parent != base / "runs" or source.parent != base / "tmp" or (root / "launch.json").exists()):
        raise ValueError("Expected fresh scoped WSP roots")
    runs = base / "runs"
    base_run = runs / "repetition-delayed-query-value-cross-20261126-v1"
    rep = runs / "repetition-delayed-query-replication-20261126-v1"
    restore = runs / "repetition-delayed-query-restore-holdout-20261126-v1"
    transfer = runs / "repetition-delayed-query-value-cross-2b-20261126-v1"
    previous_candidates = (transfer, restore, rep) if late else (restore, rep)
    predecessor = next(path for path in previous_candidates if (path / "launch.json").is_file())
    previous_sha, _ = completed(predecessor)
    base_release, old = completed(base_run)
    prior = load(predecessor / "config.json")
    checkpoint = runs / "repetition-model-sizes-20261122-v1" / "Qwen3.5-2B-checkpoint.json"
    panel = runs / "repetition-delayed-query-pad-split-20261126-v1" / "panel.json"
    queue = source / "repetition_followup_queue.py"
    _, base_model, data = paths(base)
    model = base_model if late else MODEL
    label = "DQLX" if late else "DQY"
    worker = source / "repetition_delayed_value_cross.py"
    pins = {str(source / name): pin for name, pin in SUPPORT.items()}
    if late:
        pins[str(worker)] = LATE_WORKER
    inputs = {str(queue): QUEUE, str(model / "tokenizer.json"): TOKENIZER, str(panel): PANEL}
    if not late:
        inputs[str(checkpoint)] = CHECKPOINT
    for path, digest in (pins | inputs).items():
        if sha(Path(path)) != digest:
            raise ValueError(f"Pinned bytes differ for {Path(path).name}")
    mode_arguments = ["--offset", "17"] if late else ["--checkpoint", str(checkpoint)]
    if not late:
        weights = load(checkpoint)["expected_public_snapshot"]["weights"]
        if any(sha(model / name) != fingerprint for name, fingerprint in weights.items()):
            raise ValueError("Current 2B weight bytes differ from the verified checkpoint")
    jobs = compile_jobs(root, worker, panel, model, data, label=label, panel_args=[
        "--panel", str(panel), "--panel-sha256", PANEL, *mode_arguments,
    ])
    summary = jobs.pop()
    eval_names = [f"{label}-eval-{start:03d}" for start in range(0, args.count, 8)]
    jobs = [job for job in jobs if job["lane"] == "canary" or job["id"] in eval_names]
    references, reference_pins = [], []
    for start in range(0, args.count, 8):
        name = f"DQX-eval-{start:03d}"
        row, artifact = old["jobs"][name], base_run / "jobs" / f"{name}.json"
        if row["stage"] != "completed" or row["sha256"] != sha(artifact):
            raise ValueError(f"Missing verified matched 0.8B shard {name}")
        references.append(str(artifact))
        reference_pins.append(row["sha256"])
    summary["deps"] = eval_names
    summary["argv"] = ["summarize", "--reports", *(str(root / "jobs" / f"{name}.json") for name in eval_names),
                       "--fingerprints", *(f"{{sha:{name}}}" for name in eval_names),
                       "--reference-reports", *references, "--reference-fingerprints", *reference_pins,
                       "--output", str(root / "jobs" / f"{label}-eval-summary.json")]
    jobs.append(summary)
    cfg = prior | {"source_root": str(source), "runtime": f"{source}:{prior['runtime']}",
                   "deadline": DEADLINE, "access": DEADLINE - 120, "queue_sha256": QUEUE,
                   "admission_floor_seconds": 45, "admission_buffer_seconds": 90}
    (root / "plans").mkdir()
    mode = "offset17" if late else "2b"
    plan = root / "plans" / f"plan-00-value-cross-{mode}.json"
    save(plan, {"jobs": jobs, "pins": pins})
    save(root / "config.json", cfg)
    seal = {"deadline": DEADLINE, "bridge_cutoff": 1795642200, "devices": SELECTED, "count": args.count,
            "panel_sha256": PANEL, "mode": mode, "checkpoint_sha256": None if late else CHECKPOINT,
            "base_release_sha256": base_release,
            "reference_shas": reference_pins, "predecessor_release_sha256": previous_sha,
            "manifest_shas": {plan.name: sha(plan)}, "jobs": dict(Counter(j["lane"] for j in jobs))}
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
    p.add_argument("--count", choices=(16, 32), type=int, required=True)
    p.add_argument("--offset", choices=(0, 17), type=int, default=0)
    launch(p.parse_args())


if __name__ == "__main__":
    main()
