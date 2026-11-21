"""Run the frozen same-budget restart pilot and evaluation with bounded workers."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

from repetition_mediation_queue import Supervisor, check_devices, digest, load, save
from repetition_restart import ARMS, tokens


def verify_smoke(path, source, cohort):
    with path.open() as stream:
        header, *records, footer = map(json.loads, stream)
    plan = load(cohort)
    ids = [r["row"]["id"] for r in plan["pilot"][:4]]
    if (header["phase"] != "smoke" or header["ids"] != ids or tuple(header["arms"]) != ARMS
            or header["cohort_sha256"] != digest(cohort)
            or header["worker_sha256_policy"] != digest(source) or footer["kind"] != "complete"):
        raise ValueError("canary does not match the frozen worker and cohort")
    raw = {(row["id"], row["arm"]): row for row in records if row["kind"] == "rollout"}
    if set(raw) != {(p, arm) for p in ids for arm in ARMS}:
        raise ValueError("canary lost a prompt or arm")
    if any(not all(row["short_trunk_exact"]) for row in records if row["kind"] == "batch"):
        raise ValueError("canary native trunk differs at the fixed switch")
    switched = [row for row in raw.values() if row["arm"] == "switch_restart" and row["switched"]]
    if not switched or not all(row["nested_parity"] for row in switched):
        raise ValueError("restart equivariance remains unverified")
    for row in raw.values():
        length = len(tokens(row))
        if length > plan["budget"] or (row["eos_token"] is None and length != plan["budget"]):
            raise ValueError("canary violates the total token ledger")
        if row["eos_token"] is not None and row["eos_token"] not in header["eos_ids"]:
            raise ValueError("canary captured an unrecognized EOS ID")
    return len(switched)


class RestartQueue(Supervisor):
    worker_args = ()

    def __init__(self, args):
        args.role, args.output = "queue", args.root
        super().__init__(args)
        self.source = Path(__file__).resolve().parent
        self.worker = self.source/"repetition_restart.py"
        self.env["PYTHONPATH"] = args.runtime_path

    def wave(self, phase, draw):
        label = f"{phase}{draw}"
        plans = []
        for i in range(len(self.devices)):
            out = self.lane/f"{label}-{i:02d}.jsonl"
            argv = [self.worker, *self.worker_args, "--phase", phase, "--root", self.root, "--model", self.args.model,
                    "--data", self.args.data, "--output", out, "--deadline", self.args.deadline,
                    "--part", i, "--partitions", len(self.devices), "--draw", draw]
            plans.append((i, out, argv))
        start = time.monotonic()
        outputs = self.run_batch(label, plans)
        save(self.lane/f"{label}-done.json", {"outputs": [[str(p), sha] for p, sha in outputs]})
        return time.monotonic()-start

    def score(self, phase):
        path = self.root/f"{phase}-summary.json"
        argv = [sys.executable, "-B", self.source/"repetition_restart_score.py", "--phase", phase,
                "--root", self.root, "--model", self.args.model, "--output", path]
        subprocess.run(argv, env={**self.env, "CUDA_VISIBLE_DEVICES": ""}, check=True, timeout=100)
        self.status(f"{phase}_scored", **{f"{phase}_summary_sha256": digest(path)})
        return load(path)

    def supervise(self):
        try:
            self.status(cohort_sha256=self.args.cohort_sha256)
            elapsed = self.wave("pilot", 0)
            pilot = self.score("pilot")
            parity = pilot["pilot_explicit_restart_parity"]
            if parity["same"] != parity["total"] or parity["total"] == 0:
                raise ValueError("explicit reset and same-template nested cut disagree")
            trunk = pilot["pilot_short_trunk_parity"]
            if trunk["same"] != trunk["total"]:
                raise ValueError("pilot fixed-length trunk disagrees with native")
            if self.args.evaluate:
                plan = load(self.root/"cohort.json")
                reserve = max(420, elapsed*len(plan["eval"])/len(plan["pilot"])*1.35+120)
                completed = 0
                for draw in range(self.args.eval_draws):
                    if not self.bounded(reserve):
                        break
                    elapsed = self.wave("eval", draw)
                    completed += 1
                    reserve = 1.35*elapsed+120
                if completed:
                    self.score("eval")
                self.status("completed" if completed == self.args.eval_draws else "budget_complete",
                            completed_eval_draws=completed)
            else:
                self.status("completed", completed_eval_draws=0)
        except Exception as exc:
            self.status("failed", error=str(exc))
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("launch", "supervise"))
    for name in ("root", "model", "data", "overlay", "stable", "source", "canary-report"):
        p.add_argument("--"+name, type=Path, required=True)
    for name in ("runtime-path", "devices", "host", "cohort-sha256"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--deadline", type=int, required=True)
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--eval-draws", type=int, choices=(1, 2), default=1)
    a = p.parse_args()
    q = RestartQueue(a)
    if socket.gethostname() != a.host or not 1 <= len(set(q.devices)) == len(q.devices) <= 4 or not q.bounded(420):
        raise ValueError("wrong assigned host, devices or deadline")
    if digest(a.root/"cohort.json") != a.cohort_sha256:
        raise ValueError("cohort hash changed")
    if any(digest(q.source/name) != sha for name, sha in load(a.source).items()):
        raise ValueError("staged worker checksum mismatch")
    if a.action == "supervise":
        q.supervise()
        return
    check_devices(q.devices)
    checked = verify_smoke(a.canary_report, q.worker, a.root/"cohort.json")
    q.lane.mkdir()
    with (q.lane/"supervisor.log").open("x") as log:
        child = subprocess.Popen([sys.executable, "-B", __file__, "supervise", *sys.argv[2:]], env=q.env,
                                 start_new_session=True, stdout=log, stderr=subprocess.STDOUT)
    print(json.dumps({"supervisor": child.pid, "root": str(q.root), "host": a.host, "deadline": a.deadline,
                      "devices": q.devices, "source_sha256": digest(a.source), "canary_switches": checked}))


if __name__ == "__main__":
    main()
