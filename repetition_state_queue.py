"""Supervise the native-state experiment and its gated continuation."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

from repetition_mediation_queue import Supervisor, check_devices, digest, load, save

HOST = "model-training-1-0-1"


class StateQueue(Supervisor):
    def __init__(self, args):
        args.role = "queue"
        super().__init__(args)
        self.src = Path(__file__).resolve().parent
        self.env["PYTHONPATH"] = f"{self.src}:{args.canonical}:{args.runtime}:{args.stable}:{args.mediation}:{args.overlay}"
        self.base = self.src / "repetition_state_value.py"
        self.rank = self.src / "repetition_rank_one.py"

    def cpu(self, script, action, *params):
        env = {**self.env, "CUDA_VISIBLE_DEVICES": ""}
        subprocess.run([sys.executable, "-B", str(script), action, *map(str, params)], env=env, check=True, timeout=210)

    def stage(self, phase, *, worker=None, max_reserve=300, summarize=True):
        if not self.bounded(max_reserve):
            self.status(skipped=self.state.get("skipped", []) + [phase])
            return None
        script = worker or self.base
        plans = []
        for gpu in range(2):
            out = self.lane / f"{phase}-{gpu:02d}.jsonl"
            argv = [script, "run", "--phase", phase, "--part", gpu, "--partitions", 2,
                    "--root", self.root, "--model", self.args.model, "--data", self.args.data,
                    "--output", out, "--deadline", self.args.deadline]
            plans.append((gpu, out, argv))
        outputs = self.run_batch(phase, plans)
        receipt = {"outputs": [[str(file), checksum] for file, checksum in outputs]}
        save(self.lane / f"{phase}-done.json", receipt)
        if not summarize:
            return receipt
        self.cpu(script, "summarize", "--phase", phase, "--root", self.root)
        return load(self.root / f"{phase}-summary.json")

    def run(self):
        try:
            self.status("admitted", cohort_sha256=digest(self.root / "cohort.json"))
            copy = self.stage("copy-dev")
            if copy and copy.get("copy_selection"):
                self.stage("copy-test")
            if self.args.copy_only:
                test = (self.root / "copy-test-summary.json").is_file()
                self.status("completed" if test else "copy_dev_only")
                return
            if self.args.eval_and_copy:
                evaluated = self.stage("labels-eval", max_reserve=900)
                self.status("completed" if evaluated else "partial_deadline")
                return
            self.stage("labels-dev", max_reserve=900)
            marker = self.src / "rank-ready.json"
            wait_until = min(self.args.deadline - 900, time.time() + 180)
            while not marker.is_file() and time.time() < wait_until:
                self.status("waiting_for_rank_release")
                time.sleep(8)
            if marker.is_file():
                contracts = load(marker)["sources"]
                if self.rank.name not in contracts or any(digest(self.src / name) != sha for name, sha in contracts.items()):
                    raise ValueError("rank continuation source mismatch")
                from repetition_rank_one import continue_run
                continue_run(self)
            else:
                self.stage("labels-eval", max_reserve=900)
                self.status(rank_release="not_available")
            complete = (self.root / "labels-eval-summary.json").exists() or (self.root / "rank-eval-summary.json").exists()
            self.status("completed" if complete else "partial_deadline")
        except Exception as e:
            self.status("failed", error=str(e))
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("launch", "supervise"))
    for name in ("output", "model", "data", "canonical", "stable", "mediation", "overlay", "runtime"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--deadline", type=int, required=True)
    p.add_argument("--devices", required=True)
    p.add_argument("--cohort-sha256", required=True)
    p.add_argument("--worker-sha256", required=True)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--copy-only", action="store_true")
    mode.add_argument("--eval-and-copy", action="store_true")
    args = p.parse_args()
    q = StateQueue(args)
    if (socket.gethostname() != HOST or len(set(q.devices)) != 2
            or not 120 < args.deadline - time.time() <= 10800):
        raise ValueError("incorrect node, device assignment or deadline")
    if digest(q.root / "cohort.json") != args.cohort_sha256 or digest(q.base) != args.worker_sha256:
        raise ValueError("unverified state cohort or worker")
    if args.action == "launch":
        check_devices(q.devices)
        if not args.copy_only:
            with (q.root / "timing-smoke.jsonl").open() as s:
                h, *rows, end = map(json.loads, s)
            if (h["worker_sha256"] != args.worker_sha256 or end["kind"] != "complete" or len(rows) != 6
                    or any(r["arm"] == "e0" and r["think_step"] != 0 for r in rows)
                    or any(r["arm"] in ("e0", "e32") and (r["after_think_tokens"] or 0) > 8 for r in rows)):
                raise ValueError("timing smoke did not verify forcing and matched post-tag budgets")
        q.lane.mkdir()
        with (q.lane / "supervisor.log").open("x") as log:
            child = subprocess.Popen([sys.executable, "-B", __file__, "supervise", *sys.argv[2:]], env=q.env,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"supervisor": child.pid, "root": str(q.root), "deadline": args.deadline,
                          "devices": q.devices, "cohort_sha256": args.cohort_sha256}))
    else:
        q.run()


if __name__ == "__main__":
    main()
