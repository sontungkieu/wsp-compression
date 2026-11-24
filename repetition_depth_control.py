"""Detach reporting and defer the Huy successor until the original release."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from repetition_motif_plan import load, save, sha

HARD_STOP = {"nemo-rl-env-0-0": 1795567320, "huyhq21-membedding-v2-1-0-2": 1795566300}
BASELINE_EXTENDED = "5757db8ab6eb1fa44962b29297315469455b91f13b6bb289378bd2d181b456f3"


def pinned(path, expected):
    if sha(path) != expected:
        raise ValueError(f"source checksum differs for {path.name}")


def env_for(cfg):
    return {**os.environ, "PYTHONPATH": cfg["runtime"], "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "2"}


def launch_report(root, src, cfg, expected):
    report = src / "repetition_depth_aggregate.py"
    pinned(report, expected)
    receipt = root / "aggregate-launch.json"
    if receipt.exists() or not (root / "status.json").is_file():
        raise ValueError("an active campaign without an existing reporter is required")
    with (root / "aggregate.log").open("x") as log:
        output = root / "campaign-summary.json"
        proc = subprocess.Popen([sys.executable, "-B", str(report), "--root", str(root),
                                 "--output", str(output), "--watch"], env=env_for(cfg),
                                start_new_session=True, stdout=log, stderr=subprocess.STDOUT)
    result = {"pid": proc.pid, "worker_sha256": expected, "output": str(output), "deadline": cfg["deadline"]}
    save(receipt, result)
    print(json.dumps(result), flush=True)


def verify_inventory(cfg):
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,memory.used,utilization.gpu",
                                    "--format=csv,noheader,nounits"], text=True).splitlines()
    cards = [line.split(", ") for line in rows]
    if [x[0] for x in cards] != cfg["devices"]:
        raise ValueError("exact GPU roster changed")
    pids = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                    "--format=csv,noheader,nounits"], text=True).strip()
    return not pids and all(int(n) <= 5 and util == "0" for _, n, util in cards)


def attest_host(cfg):
    if socket.gethostname() != cfg["host"] or HARD_STOP.get(cfg["host"]) != cfg["deadline"]:
        raise ValueError("campaign does not match an authorized host and cutoff")


def defer(root, old, src, args):
    cfg = load(old / "config.json")
    attest_host(cfg)
    if cfg["host"] != "huyhq21-membedding-v2-1-0-2" or time.time() >= cfg["deadline"] - 1800:
        raise ValueError("successor applies to Huy before the admission buffer")
    verify_inventory(cfg)
    receipt = root / "defer-launch.json"
    if receipt.exists() or (root / "config.json").exists():
        raise FileExistsError(receipt)
    original_sources = Path(cfg["source_root"])
    for name, digest in (("repetition_followup_queue.py", cfg["queue_sha256"]),
                         ("repetition_followup_baseline_extended.py", BASELINE_EXTENDED)):
        data = (original_sources / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"old worker pin changed for {name}")
        with (src / name).open("xb") as stream:
            stream.write(data)
    pinned(src / "repetition_motif_plan.py", args.planner_sha)
    pinned(src / "repetition_motif_intervention.py", args.motif_sha)
    pinned(src / "repetition_depth_aggregate.py", args.aggregate_sha)
    successor = cfg | {"source_root": str(src), "runtime": str(src) + ":" + cfg["runtime"],
                       "access": cfg["deadline"] - 1800}
    save(root / "config.json", successor)
    with (root / "defer.log").open("x") as log:
        cmd = [sys.executable, "-B", str(Path(__file__).resolve()), "wait",
               "--root", str(root), "--archive", str(old), "--source", str(src),
               "--planner-sha", args.planner_sha, "--motif-sha", args.motif_sha, "--aggregate-sha", args.aggregate_sha]
        proc = subprocess.Popen(cmd, env=env_for(successor), start_new_session=True,
                                stdout=log, stderr=subprocess.STDOUT)
    result = {"gate_pid": proc.pid, "host": cfg["host"], "archive": str(old), "deadline": cfg["deadline"],
              "requires": ["archive release SHA agrees with status", "all four exact UUIDs are idle"]}
    save(receipt, result)
    print(json.dumps(result), flush=True)


def wait_for_huy(root, old, src, args):
    cfg = load(root / "config.json")
    attest_host(cfg)
    release = old / "release.json"
    while time.time() < cfg["access"]:
        # An empty queue and an idle device roster are independent admission checks.
        if release.is_file() and verify_inventory(cfg):
            data = load(release)
            if (data["status_sha256"] != sha(old / "status.json") or data["cutoff_epoch"] != cfg["deadline"]
                    or data["host"] != cfg["host"] or data["campaign_compute"]):
                raise ValueError("archived release receipt is invalid for successor")
            break
        time.sleep(30)
    else:
        save(root / "gate-result.json", {"state": "expired", "deadline": cfg["deadline"], "observed": time.time()})
        return
    planner = src / "repetition_motif_plan.py"
    queue = src / "repetition_followup_queue.py"
    pinned(planner, args.planner_sha)
    pinned(src / "repetition_motif_intervention.py", args.motif_sha)
    pinned(queue, cfg["queue_sha256"])
    env = env_for(cfg)
    subprocess.run([sys.executable, "-B", str(planner), "--root", str(root), "--source", str(src),
                    "--archive", str(old), "--node", "huy"], env=env, check=True)
    while time.time() < cfg["access"] and not verify_inventory(cfg):
        time.sleep(30)
    if time.time() >= cfg["access"]:
        save(root / "gate-result.json", {"state": "expired_after_validation", "deadline": cfg["deadline"]})
        return
    hold = {"archive_release_sha256": sha(release), "deadline": cfg["deadline"], "verified_idle": time.time(),
            "manifest_shas": {p.name: sha(p) for p in (root / "plans").glob("plan-*.json")}}
    save(root / "sealed.json", hold)
    subprocess.run([sys.executable, "-B", str(queue), "launch", "--root", str(root), "--config-sha256",
                    sha(root / "config.json")], env=env, check=True)
    while not (root / "status.json").is_file() and time.time() < cfg["access"]:
        time.sleep(2)
    launch_report(root, src, cfg, args.aggregate_sha)
    save(root / "gate-result.json", {"state": "launched", "verified": hold, "queue": load(root / "launch.json")})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("report", "defer", "wait"))
    for name in ("root", "archive", "source"):
        p.add_argument("--" + name, type=Path, required=name != "archive")
    p.add_argument("--planner-sha")
    p.add_argument("--motif-sha")
    p.add_argument("--aggregate-sha", required=True)
    a = p.parse_args()
    if a.action == "report":
        cfg = load(a.root / "config.json")
        attest_host(cfg)
        launch_report(a.root, a.source, cfg, a.aggregate_sha)
    elif a.action == "defer":
        defer(a.root, a.archive, a.source, a)
    else:
        try:
            wait_for_huy(a.root, a.archive, a.source, a)
        except Exception as error:
            save(a.root / "gate-result.json", {"state": "failed", "error": str(error)})
            raise


if __name__ == "__main__":
    main()
