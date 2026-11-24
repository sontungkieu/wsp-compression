"""Requeue Nemo motif shards displaced by another workload, under fresh output names."""
import argparse
import json
from pathlib import Path
import socket
import time

from repetition_campaign_resume import DEADLINE, HOST, load, save, sha


def retry_manifest(root):
    cfg = load(root / "config.json")
    if (socket.gethostname() != HOST or cfg["host"] != HOST or cfg["deadline"] != DEADLINE
            or cfg["host_devices"][2:] != cfg["devices"] or time.time() >= cfg["access"]):
        raise ValueError("Nemo successor admission identity or cutoff differs")
    path = root / "plans" / "plan-00-resumed.json"
    status = load(root / "status.json")
    base_sha = sha(path)
    if (load(root / "sealed.json")["manifest_shas"].get(path.name) != base_sha
            or status["manifest_shas"].get(path.name) != base_sha):
        raise ValueError("Resumed manifest differs from the admitted seal")
    resume = load(path)
    marks = status["jobs"]
    interrupted = {j["id"] for j in resume["jobs"] if j["lane"] == "MOTIF" and j["kind"] == "gpu"
                   and marks[j["id"]]["stage"] == "failed"
                   and marks[j["id"]].get("reason") == "additional_compute_process"}
    if not interrupted:
        raise ValueError("No displaced motif shards require retry")
    for name in interrupted:
        mark = marks[name]
        if (load(root / "jobs" / f"{name}.receipt.json") != mark or Path(mark["output"]).exists()
                or marks[name + "-review"]["stage"] != "blocked"):
            raise ValueError(f"Displaced shard state differs: {name}")
    renames = {n: n.replace("MOTIF-", "MOTIF-retry-", 1)
               for name in sorted(interrupted) for n in (name, name + "-review")}
    jobs, pins = [], {}
    for job in resume["jobs"]:
        old = job["id"]
        if old not in renames:
            continue
        for arg in job["argv"]:
            if arg.startswith(str(root / "jobs") + "/"):
                source = Path(arg)
                if source.stem not in renames:
                    raise ValueError(f"Unknown retry input or output for {old}")
        argv = [value.replace(f"{root}/jobs/MOTIF-", f"{root}/jobs/MOTIF-retry-", 1)
                .replace("{sha:MOTIF-", "{sha:MOTIF-retry-", 1) for value in job["argv"]]
        deps = [renames[dep] for dep in job["deps"]]
        jobs.append(job | {"id": renames[old], "argv": argv, "deps": deps, "priority": 11})
        pins[job["worker"]] = resume["pins"][job["worker"]]
    if len(jobs) != 2 * len(interrupted):
        raise ValueError("Every displaced shard needs one verified replay and score task")
    plan = root / "plans" / "plan-02-displaced-retry.json"
    save(plan, {"pins": pins, "jobs": jobs})
    receipt = {"parent_sha256": base_sha, "plan": plan.name, "plan_sha256": sha(plan),
               "deadline": DEADLINE, "replays": sorted(interrupted), "reason": "additional_compute_process"}
    save(root / "sealed-retry-addendum.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(retry_manifest(args.root)))


if __name__ == "__main__":
    main()
