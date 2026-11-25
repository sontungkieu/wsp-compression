"""Check restoration raw receipts and rescore exact response parity on CPU."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

from repetition_delayed_campaign import load, sha


def audit(root, worker, fingerprint):
    status_path = root / "status.json"
    status, release = load(status_path), load(root / "release.json")
    if status["state"] != "completed" or sha(status_path) != release["status_sha256"]:
        raise ValueError("Restoration campaign has no verified completed release")
    summary_path = root / "jobs" / "DQP-eval-summary.json"
    job = status["jobs"]["DQP-eval-summary"]
    if job["stage"] != "completed" or sha(summary_path) != job["sha256"] or sha(worker) != fingerprint:
        raise ValueError("Summary or audit worker differs from the pinned bytes")
    summary = load(summary_path)
    reports = []
    for start, digest in enumerate(summary["score_shas"]):
        name = f"DQP-eval-{start * 8:03d}"
        row = status["jobs"][name]
        path = root / "jobs" / f"{name}.json"
        if row["stage"] != "completed" or row["sha256"] != digest or sha(path) != digest:
            raise ValueError(f"Unverified restoration shard {name}")
        reports.append(str(path))
    output = root / "restore-rollout-parity-summary.json"
    runtime = load(root / "config.json")["runtime"]
    env = {**os.environ, "PYTHONPATH": runtime, "CUDA_VISIBLE_DEVICES": "",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    subprocess.run([sys.executable, "-B", str(worker), "summarize", "--reports", *reports,
                    "--fingerprints", *summary["score_shas"], "--output", str(output)],
                   check=True, env=env)
    print(sha(output), output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--worker", type=Path, required=True)
    p.add_argument("--worker-sha256", required=True)
    a = p.parse_args()
    audit(a.root, a.worker, a.worker_sha256)


if __name__ == "__main__":
    main()
