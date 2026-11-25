"""Check restoration raw receipts and rescore exact response parity on CPU."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys

from repetition_delayed_campaign import load, sha


def settled(root):
    status_path = root / "status.json"
    status, release = load(status_path), load(root / "release.json")
    if status["state"] != "completed" or sha(status_path) != release["status_sha256"]:
        raise ValueError("Restoration campaign has no verified completed release")
    summary_path = root / "jobs" / "DQP-eval-summary.json"
    job = status["jobs"]["DQP-eval-summary"]
    if job["stage"] != "completed" or sha(summary_path) != job["sha256"]:
        raise ValueError("Summary differs from the pinned bytes")
    return status, load(summary_path)


def checked_reports(root, status, fingerprints, first_shard=0):
    reports = []
    for start, digest in enumerate(fingerprints, start=first_shard):
        name = f"DQP-eval-{start * 8:03d}"
        row = status["jobs"][name]
        path = root / "jobs" / f"{name}.json"
        if row["stage"] != "completed" or row["sha256"] != digest or sha(path) != digest:
            raise ValueError(f"Unverified restoration shard {name}")
        reports.append(path)
    return reports


def audit(root, worker, fingerprint):
    if sha(worker) != fingerprint:
        raise ValueError("Audit worker differs from the pinned bytes")
    status, summary = settled(root)
    reports = checked_reports(root, status, summary["score_shas"])
    output = root / "restore-rollout-parity-summary.json"
    runtime = load(root / "config.json")["runtime"]
    env = {**os.environ, "PYTHONPATH": runtime, "CUDA_VISIBLE_DEVICES": "",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    subprocess.run([sys.executable, "-B", str(worker), "summarize", "--reports", *map(str, reports),
                    "--fingerprints", *summary["score_shas"], "--output", str(output)],
                   check=True, env=env)
    print(sha(output), output)


def compare_controls(root, previous):
    now, followup = settled(root)
    old, baseline = settled(previous)
    identity = ("anchor", "cap", "draws", "panel_sha256", "tokenizer_sha256", "checkpoint", "runtime", "sampling")
    if not followup.get("matrix_control") or any(followup[field] != baseline[field] for field in identity):
        raise ValueError("Matrix control and earlier restoration identities differ")
    fingerprints = followup["score_shas"]
    start = load(root / "sealed.json").get("start", 0)
    if start < 0 or start % 8:
        raise ValueError("Unexpected unaligned starting point for paired restoration")
    first_shard = start // 8
    reference_shas = baseline["score_shas"][first_shard:first_shard + len(fingerprints)]
    current = checked_reports(root, now, fingerprints, first_shard)
    reference = checked_reports(previous, old, reference_shas, first_shard)
    arms = {"keep", "pulse_0", "conv_matrix"}

    def rows(paths):
        points = [(r["id"], r["draw"], r["variant"], r["arm"], r) for path in paths
                  for r in load(path)["rows"] if r["arm"] in arms]
        values = {tuple(p[:4]): p[-1] for p in points}
        if len(values) != len(points):
            raise ValueError("Duplicated control replay identity")
        return values

    controls, source = rows(current), rows(reference)
    if controls.keys() != source.keys() or len(controls) != followup["prompts"] * followup["draws"] * 2 * len(arms):
        raise ValueError("Matrix control and restoration rows do not pair exactly")
    exact, matched = Counter(), Counter()
    for key, cell in controls.items():
        ref = source[key]
        if (cell["seed"], cell["post_seed"]) != (ref["seed"], ref["post_seed"]):
            raise ValueError("Control replay changed its paired seed")
        matched[key[-1]] += 1
        exact[key[-1]] += int((cell["tokens"], cell["eos_token"]) == (ref["tokens"], ref["eos_token"]))
    result = {"prompts": followup["prompts"], "start": start, "paired_controls": matched,
              "exact_tokens_and_eos": exact, "matrix_shas": fingerprints, "reference_shas": reference_shas}
    print(json.dumps(result))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--worker", type=Path)
    p.add_argument("--worker-sha256")
    p.add_argument("--compare-controls", type=Path)
    a = p.parse_args()
    if a.compare_controls:
        compare_controls(a.root, a.compare_controls)
    elif a.worker and a.worker_sha256:
        audit(a.root, a.worker, a.worker_sha256)
    else:
        p.error("provide --compare-controls or a pinned --worker and --worker-sha256")


if __name__ == "__main__":
    main()
