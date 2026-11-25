"""Finish the eight missing split-padding cases on the two authorized GPUs."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from repetition_delayed_campaign import DEVICES, HOST, load, preflight, save, sha


PANEL = "35e2e6209896e66410120aa481a7a2b49351483fd6b9e2a08935bb81e5873d79"
WORKER = "975a7fb4348a3f18c38b604a75e90d7f1154dccc0465e4fda6b6d38f5c14407b"
QUEUE = "2aefdca6391630f8d9c95fdd91ada9d872d69e76a7a6a6e3b1e688ef601dc858"
ARCHIVE_NAME = "repetition-delayed-query-pad-split-20261126-v1"
SELECTED = (1, 3)


def verified_archive(archive, inputs, release_pin, status_pin):
    config = load(archive / "config.json")
    status_path, release_path = archive / "status.json", archive / "release.json"
    if sha(status_path) != status_pin or sha(release_path) != release_pin or sha(archive / "panel.json") != PANEL:
        raise ValueError("Expected the verified archived panel, status and release")
    status, release = load(status_path), load(release_path)
    scope = [DEVICES[i] for i in SELECTED]
    if (release["status_sha256"] != status_pin or release["campaign_compute"]
            or config["host"] != HOST or config["devices"] != scope or config["source_root"] != str(inputs)
            or config["deadline"] != 1795631904 or release["state"] != "partial"):
        raise ValueError("Predecessor release or scope differs")
    manifest = archive / "plans" / "plan-00-delayed-query-length.json"
    key, fingerprint = manifest.name, sha(manifest)
    if (load(archive / "sealed.json")["manifest_shas"][key] != fingerprint
            or status["manifest_shas"][key] != fingerprint):
        raise ValueError("Original campaign plan differs from the seal")
    marks = status["jobs"]
    for start in range(0, 56, 8):
        name = f"DQL-eval-{start:03d}"
        parent, score = marks[name], marks[name + "-score"]
        for ident, mark in ((name, parent), (name + "-score", score)):
            if (mark["stage"] != "completed" or load(archive / "jobs" / f"{ident}.receipt.json") != mark
                    or sha(Path(mark["output"])) != mark["sha256"]):
                raise ValueError(f"Original completed receipt differs for {ident}")
        if load(Path(score["output"]))["raw_sha256"] != parent["sha256"]:
            raise ValueError(f"Original raw/score mismatch for {name}")
    if marks["DQL-eval-056"]["stage"] not in {"skipped_deadline", "failed"} or "DQL-eval-060" in marks:
        raise ValueError("Archived result does not have exactly the expected missing range")
    return config, load(manifest)


def tail_jobs(root, source, template, deadline):
    jobs = []
    old = {job["id"]: job for job in template["jobs"]}
    worker = str(source / "repetition_delayed_query.py")
    out = root / "jobs"
    for preferred, start in enumerate((56, 60)):
        name = f"DQL-eval-{start:03d}"
        gpu = old["DQL-eval-056"]
        if gpu["kind"] != "gpu":
            raise ValueError("Unexpected archived job kind")
        args = gpu["argv"].copy()
        for key, value in {"--start": str(start), "--count": "4", "--deadline": str(deadline),
                           "--output": str(out / f"{name}.jsonl")}.items():
            args[args.index(key) + 1] = value
        jobs.append(gpu | {"id": name, "worker": worker, "argv": args, "preferred": preferred, "eta": 210})
        score = old["DQL-eval-056-score"]
        args = score["argv"].copy()
        for key, value in {"--raw": str(out / f"{name}.jsonl"), "--fingerprint": f"{{sha:{name}}}",
                           "--output": str(out / f"{name}-score.json")}.items():
            args[args.index(key) + 1] = value
        jobs.append(score | {"id": name + "-score", "worker": worker, "argv": args, "deps": [name],
                             "preferred": preferred, "eta": 45})
    return jobs


def launch(args):
    observed = preflight()
    if not set(SELECTED) <= set(observed["idle"]):
        raise ValueError("GPU indexes 1 and 3 must both be idle")
    base, root, source = args.base, args.root, args.source
    archive, inputs = base / "runs" / ARCHIVE_NAME, base / "tmp" / ARCHIVE_NAME
    if root.parent != base / "runs" or source.parent != base / "tmp" or (root / "launch.json").exists():
        raise ValueError("Expected a fresh scoped tail run")
    original, template = verified_archive(archive, inputs, args.release_sha256, args.status_sha256)
    for name, pin in (("repetition_delayed_query.py", WORKER), ("repetition_followup_queue.py", QUEUE)):
        src, dest = inputs / name, source / name
        if sha(src) != pin:
            raise ValueError(f"Archived digest differs for {name}")
        with src.open("rb") as incoming, dest.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing)
        if sha(dest) != pin:
            raise ValueError(f"Copied digest differs for {name}")
    deadline = int(time.time()) + 900
    cfg = original | {"deadline": deadline, "access": deadline - 120, "source_root": str(source),
                      "runtime": f"{source}:{original['runtime']}",
                      "admission_floor_seconds": 45, "admission_buffer_seconds": 30}
    jobs = tail_jobs(root, source, template, deadline)
    (root / "plans").mkdir()
    plan = root / "plans" / "plan-00-split-tail.json"
    save(plan, {"jobs": jobs, "pins": {str(source / "repetition_delayed_query.py"): WORKER}})
    save(root / "config.json", cfg)
    seal = {"deadline": deadline, "archive": str(archive), "archive_status_sha256": args.status_sha256,
            "archive_release_sha256": args.release_sha256, "panel_sha256": PANEL, "starts": [56, 60],
            "manifest_shas": {plan.name: sha(plan)}, "jobs": dict(Counter(j["kind"] for j in jobs))}
    save(root / "sealed.json", seal)
    subprocess.run([sys.executable, "-B", str(source / "repetition_followup_queue.py"), "launch", "--root",
                    str(root), "--config-sha256", sha(root / "config.json")],
                   check=True, env={**os.environ, "PYTHONPATH": cfg["runtime"], "CUDA_VISIBLE_DEVICES": "",
                                    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    print(json.dumps(seal | {"seal_sha256": sha(root / "sealed.json")}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "root", "source"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--release-sha256", required=True)
    p.add_argument("--status-sha256", required=True)
    launch(p.parse_args())


if __name__ == "__main__":
    main()
