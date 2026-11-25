"""Schedule the numeric-content probe after the cache-restoration release."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from repetition_delayed_campaign import DEVICES, HOST, load, paths, preflight, save, sha
from repetition_delayed_timing_campaign import DEADLINE, PANEL, QUEUE, SELECTED, validate_release
from repetition_delayed_transport_campaign import TIMING_WORKER, compile_jobs


PANEL_PATHS = (
    ("repetition-delayed-query-length-20261126-v2",
     "979a96af65143cda4908f4b1b27adff4b84be033b1a1b95ec9e2ba9e698ffad6",
     "115c4c77f017622e489a39f191e98fa108c48a9280b8d28d56e78e93c94ab1e0"),
    ("repetition-delayed-query-pad-split-20261126-v1", PANEL,
     "b7841a71742e7ab5f3783ab150e1a4ace5553764bee4c0fb2e3fc9df7ce9b8c3"),
    ("repetition-delayed-query-pad-position-20261126-v1",
     "13eb607ba1dd105294b2988bf813137b18bdba972aa1bb09808adb392e351a4c",
     "ed1ae68e06294ddfa1b6ed3241040c651ecff162a204695fe189611052406e32"),
)


def launch(args):
    live = preflight()
    if not set(SELECTED) <= set(live["idle"]):
        raise ValueError("Expected both assigned GPUs idle after restoration")
    if time.time() >= DEADLINE - 700:
        raise TimeoutError("Insufficient time for the numeric-content probe")
    root, source, base = args.root, args.source, args.base
    if root.parent != base / "runs" or source.parent != base / "tmp" or (root / "launch.json").exists():
        raise ValueError("Expected fresh scoped WSP run and source roots")
    transport = base / "runs" / "repetition-delayed-query-transport-20261126-v1"
    release_path = transport / "release.json"
    release, prior = load(release_path), load(transport / "config.json")
    devices = [DEVICES[i] for i in SELECTED]
    if (release["status_sha256"] != sha(transport / "status.json") or release["campaign_compute"]
            or release["host"] != HOST or release["state"] != "completed" or prior["devices"] != devices):
        raise ValueError("Expected completed restoration and a verified release")
    panels, pins = [], []
    for slug, pin, release_pin in PANEL_PATHS:
        archive = base / "runs" / slug
        validate_release(archive, release_pin)
        path = archive / "panel.json"
        if sha(path) != pin:
            raise ValueError(f"Unexpected panel bytes for {slug}")
        panels.append(str(path))
        pins.append(pin)
    worker = source / "repetition_delayed_numeric.py"
    queue = source / "repetition_followup_queue.py"
    timing = source / "repetition_delayed_timing.py"
    for path, expected in ((worker, args.worker_sha256), (queue, QUEUE), (timing, TIMING_WORKER)):
        if sha(path) != expected:
            raise ValueError(f"Pinned bytes differ for {path.name}")
    _, model, data = paths(base)
    cfg = prior | {"source_root": str(source), "runtime": f"{source}:{prior['runtime']}",
                   "deadline": DEADLINE, "access": DEADLINE - 120, "queue_sha256": QUEUE,
                   "admission_floor_seconds": 45, "admission_buffer_seconds": 70}
    jobs = compile_jobs(root, worker, None, model, data, label="DQN",
                        panel_args=["--panels", *panels, "--panel-sha256", *pins])
    (root / "plans").mkdir()
    plan = root / "plans" / "plan-00-numeric-odds.json"
    save(plan, {"jobs": jobs, "pins": {str(worker): args.worker_sha256, str(timing): TIMING_WORKER}})
    save(root / "config.json", cfg)
    seal = {"deadline": DEADLINE, "bridge_cutoff": 1795642200, "devices": SELECTED, "panel_sha256": pins,
            "transport_release_sha256": sha(release_path), "transport_status_sha256": release["status_sha256"],
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
