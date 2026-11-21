"""Run a cache-swap canary, then paired development draws before a hard deadline."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys

from repetition_cache_swap import BATCH
from repetition_mediation_queue import Supervisor, check_devices, digest, load, save


class CacheSwapQueue(Supervisor):
    def __init__(self, args):
        args.role, args.output = "queue", args.root
        super().__init__(args)
        self.env["PYTHONPATH"] = args.runtime_path
        self.source = Path(__file__).resolve().parent

    def plan(self, lane, phase, draw, cap):
        out = self.lane/f"{phase}-{draw:02d}.jsonl"
        argv = [self.source/"repetition_cache_swap.py", "--phase", phase, "--draw", draw, "--cap", cap,
                "--root", self.root, "--model", self.args.model, "--data", self.args.data, "--output", out,
                "--deadline", self.args.deadline]
        return lane, out, argv

    def verify_smoke(self, path):
        with path.open() as stream:
            header, *rows, footer = map(json.loads, stream)
        probes = [a for a in rows if a["kind"] == "audit"]
        if (header["cohort_sha256"] != self.args.cohort_sha256 or header["phase"] != "smoke"
                or header["worker_sha256_cache"] != digest(self.source/"repetition_cache_swap.py")
                or header["cap"] != 64 or len(header["ids"]) != BATCH or footer["kind"] != "complete"
                or len(probes) != 1 or not probes[0]["donor_sha256"]
                or min(probes[0]["max_logit_change"]) <= 0):
            raise ValueError("cache-swap canary audit failed")

    def supervise(self):
        try:
            self.status(cohort_sha256=self.args.cohort_sha256)
            (canary, sha), = self.run_batch("canary", [self.plan(0, "smoke", 0, 64)])
            self.verify_smoke(canary)
            self.status("canary_passed", canary_sha256=sha)
            if not self.bounded(420):
                self.status("budget_complete")
                return
            if len(self.devices) == 2:
                checks = self.run_batch("dev", [self.plan(i, "dev", i, 512) for i in range(2)])
            else:
                checks = []
                for i in range(2):
                    checks.extend(self.run_batch(f"dev{i}", [self.plan(0, "dev", i, 512)]))
            save(self.lane/"draws-done.json", {"outputs": [[str(p), sha] for p, sha in checks]})
            summary = self.root/"dev-summary.json"
            argv = [sys.executable, "-B", self.source/"repetition_cache_swap_score.py", "--root", self.root,
                    "--model", self.args.model, "--output", summary]
            subprocess.run(argv, env={**self.env, "CUDA_VISIBLE_DEVICES": ""}, check=True, timeout=100)
            self.status("completed", summary_sha256=digest(summary))
        except Exception as exc:
            self.status("failed", error=str(exc))
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("launch", "supervise"))
    for name in ("root", "model", "data", "overlay", "stable", "source"):
        p.add_argument("--"+name, type=Path, required=True)
    for name in ("runtime-path", "devices", "host", "cohort-sha256"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--deadline", type=int, required=True)
    a = p.parse_args()
    q = CacheSwapQueue(a)
    if socket.gethostname() != a.host or not 1 <= len(set(q.devices)) == len(q.devices) <= 2 or not q.bounded(420):
        raise ValueError("wrong cache experiment host, devices or deadline")
    if digest(a.root/"cohort.json") != a.cohort_sha256:
        raise ValueError("cache-swap cohort hash changed")
    if any(digest(q.source/name) != sha for name, sha in load(a.source).items()):
        raise ValueError("staged cache-swap source hash changed")
    if a.action == "supervise":
        q.supervise()
        return
    check_devices(q.devices)
    q.lane.mkdir()
    with (q.lane/"supervisor.log").open("x") as log:
        child = subprocess.Popen([sys.executable, "-B", __file__, "supervise", *sys.argv[2:]], env=q.env,
                                 start_new_session=True, stdout=log, stderr=subprocess.STDOUT)
    print(json.dumps({"supervisor":child.pid, "root":str(q.root), "host":a.host, "deadline":a.deadline,
                      "devices":q.devices, "source_sha256":digest(a.source)}))


if __name__ == "__main__":
    main()
