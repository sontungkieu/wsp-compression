"""Schedule fixed-context cache-timing replays on the two assigned GE4 GPUs."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from repetition_delayed_campaign import DEVICES, HOST, load, paths, preflight, save, sha


PANEL = "35e2e6209896e66410120aa481a7a2b49351483fd6b9e2a08935bb81e5873d79"
ORIGINAL_RELEASE = "b7841a71742e7ab5f3783ab150e1a4ace5553764bee4c0fb2e3fc9df7ce9b8c3"
TAIL_RELEASE = "71b782a39c6c61c887dad06e0f45354e28f11138a0f3980a790c77e21b82bbcb"
MERGED = "867aa335e98ccdb91e0aec485b51c37a3be4e44f4abb4f0ebd1fde7c5795f06a"
QUEUE = "2aefdca6391630f8d9c95fdd91ada9d872d69e76a7a6a6e3b1e688ef601dc858"
HELPER = "d87750404144dea62a981ebf4e350ab2a7ebd1ef59a4d851b1bd444aa5ede598"
SELECTED = (1, 3)
DEADLINE = 1795641900


def compile_jobs(root, worker, panel, model, data):
    jobs = []
    out = root / "jobs"

    def add(name, lane, kind, argv, deps, extension, eta, preferred=0):
        jobs.append({"id": name, "lane": lane, "kind": kind, "worker": str(worker),
                     "extension": extension, "argv": [*argv, "--output", str(out / (name + extension))],
                     "deps": deps, "priority": 0 if lane == "canary" else 10,
                     "eta": eta, "preferred": preferred})

    common = ["--panel", str(panel), "--panel-sha256", PANEL, "--model", str(model)]

    def shard(split, start, count, draws, needs, preferred):
        name = f"DQT-{split}-{start:03d}"
        cap = 128 if split == "canary" else 512
        add(name, split, "gpu", ["run", *common, "--data", str(data), "--split", split,
            "--start", str(start), "--count", str(count), "--draws", str(draws),
            "--cap", str(cap), "--deadline", str(DEADLINE)], needs, ".jsonl",
            180 if split == "canary" else 620, preferred)
        review = name + "-score"
        add(review, split, "cpu", ["score", *common, "--raw", str(out / f"{name}.jsonl"),
            "--fingerprint", f"{{sha:{name}}}"], [name], ".json", 65)
        return review

    canaries = [shard("canary", index, 1, 1, [], index) for index in range(2)]
    scores = [shard("eval", start, 8, 2, canaries, index % 2)
              for index, start in enumerate(range(0, 64, 8))]
    add("DQT-eval-summary", "eval", "cpu",
        ["summarize", "--reports", *(str(out / f"{name}.json") for name in scores),
         "--fingerprints", *(f"{{sha:{name}}}" for name in scores)], scores, ".json", 25)
    return jobs


def validate_release(root, fingerprint):
    path = root / "release.json"
    if sha(path) != fingerprint:
        raise ValueError(f"Unexpected release digest for {root.name}")
    release = load(path)
    if (release["host"] != HOST or release["status_sha256"] != sha(root / "status.json")
            or release["campaign_compute"] or [r[1] for r in release["allocation"]] != [DEVICES[i] for i in SELECTED]):
        raise ValueError(f"Invalid release/roster evidence for {root.name}")


def launch(args):
    live = preflight()
    if not set(SELECTED) <= set(live["idle"]):
        raise ValueError("Expected GPU indexes 1 and 3 idle")
    if time.time() >= DEADLINE - 1800:
        raise TimeoutError("Insufficient allocation time to start an eight-shard timing run")
    root, source, base = args.root, args.source, args.base
    archive = base / "runs" / "repetition-delayed-query-pad-split-20261126-v1"
    tail = base / "runs" / "repetition-delayed-query-pad-split-tail-20261126-v1"
    if root.parent != base / "runs" or source.parent != base / "tmp" or (root / "launch.json").exists():
        raise ValueError("Expected fresh scoped WSP run and source roots")
    validate_release(archive, ORIGINAL_RELEASE)
    validate_release(tail, TAIL_RELEASE)
    report = tail / "DQL-three-position-summary.json"
    if sha(report) != MERGED or not load(report)["complete"] or load(report)["prompts"] != 64:
        raise ValueError("Expected all 64 archived split-padding cases")
    prior = load(archive / "config.json")
    panel = archive / "panel.json"
    worker, queue, helper = [source / filename for filename in
                            ("repetition_delayed_timing.py", "repetition_followup_queue.py",
                             "repetition_delayed_campaign.py")]
    for path, fingerprint in ((panel, PANEL), (worker, args.worker_sha256), (queue, QUEUE), (helper, HELPER)):
        if sha(path) != fingerprint:
            raise ValueError(f"Unexpected pinned bytes for {path.name}")
    _, model, data = paths(base)
    abi = load(archive / "preflight.json")["abi"]
    if abi["checkpoint"] != str(model):
        raise ValueError("Archived runtime receipt identifies a different model")
    for pin in abi["harness"].values():
        if sha(Path(pin["path"])) != pin["sha256"]:
            raise ValueError("Archived harness bytes differ")
    cfg = prior | {"host": HOST, "host_devices": DEVICES, "devices": [DEVICES[i] for i in SELECTED],
                   "deadline": DEADLINE, "access": DEADLINE - 180, "source_root": str(source),
                   "runtime": f"{source}:{prior['runtime']}", "queue_sha256": QUEUE,
                   "admission_floor_seconds": 45, "admission_buffer_seconds": 120}
    jobs = compile_jobs(root, worker, panel, model, data)
    (root / "plans").mkdir()
    path = root / "plans" / "plan-00-reset-offsets.json"
    save(path, {"jobs": jobs, "pins": {str(worker): args.worker_sha256}})
    save(root / "config.json", cfg)
    seal = {"deadline": DEADLINE, "bridge_cutoff": 1795642200, "devices": SELECTED, "panel_sha256": PANEL,
            "archived_split_release": ORIGINAL_RELEASE, "archived_tail_release": TAIL_RELEASE,
            "archive_summary_sha256": MERGED, "manifest_shas": {path.name: sha(path)},
            "jobs": dict(Counter(j["lane"] + ":" + j["kind"] for j in jobs))}
    save(root / "sealed.json", seal)
    env = {**os.environ, "PYTHONPATH": cfg["runtime"], "CUDA_VISIBLE_DEVICES": "",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    subprocess.run([sys.executable, "-B", str(queue), "launch", "--root", str(root),
                    "--config-sha256", sha(root / "config.json")], env=env, check=True)
    print(json.dumps(seal | {"seal_sha256": sha(root / "sealed.json")}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "root", "source"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--worker-sha256", required=True)
    launch(p.parse_args())


if __name__ == "__main__":
    main()
