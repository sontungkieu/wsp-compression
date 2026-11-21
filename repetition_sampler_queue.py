"""Run the sampler reachability probes and dev-gated replay on the assigned pair."""
import argparse
import json
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time

from repetition_mediation_queue import check_devices, digest, load, save
from repetition_sampler_reach import choose, iter_probe
from repetition_state_queue import HOST, StateQueue


class ReachQueue(StateQueue):
    def __init__(self, args):
        super().__init__(args)
        self.src = Path(__file__).resolve().parent
        self.worker = self.src / "repetition_sampler_reach.py"
        self.env["PYTHONPATH"] = f"{args.runtime}:{self.src}:{args.base_source}:{args.canonical}:{args.stable}:{args.mediation}:{args.overlay}"

    def probe(self, phase, reserve=600):
        return self.stage(phase, worker=self.worker, max_reserve=reserve)

    def run(self):
        try:
            self.status("admitted", source_sha256=digest(self.args.source), cohort_sha256=self.args.cohort_sha256)
            if self.args.norm_only:
                result = self.probe("norm-eval")
                self.status("completed" if result else "partial_deadline")
                return
            if not self.probe("probe-dev"):
                self.status("partial_deadline")
                return
            dev = list(iter_probe(self.root, "probe-dev"))
            rankings, diagnostics = choose(dev)
            positive = sum(r["patches"]["3:left_to_blank"]["q"] > 0 for r in dev)
            used_higher, used_oracle = False, False
            plan = {}
            if not rankings and self.probe("higher-dev"):
                used_higher = True
                dev += list(iter_probe(self.root, "higher-dev"))
                rankings, diagnostics = choose(dev)
            if not rankings and self.bounded(1200):
                bound = statistics.median(r["last_native_donor_norm"] for r in dev if r["patches"].get("3:old_fixed"))
                plan["oracle_norm"] = bound
                save(self.root / "oracle-budget.json", plan)
                if self.probe("oracle-dev"):
                    used_oracle = True
                    dev += list(iter_probe(self.root, "oracle-dev"))
                    rankings, diagnostics = choose(dev)
            picked, used = [], set()
            for name in rankings:
                family = "oracle" if "oracle_" in name else name
                if family not in used and len(picked) < 2:
                    picked.append(name)
                    used.add(family)
            plan.update(picked=picked, higher=used_higher, oracle=used_oracle, development=diagnostics,
                        positive_control_support=positive)
            save(self.root / "decision.json", plan)
            self.status("decision_frozen", decision_sha256=digest(self.root / "decision.json"), picked=picked)
            if not self.probe("probe-eval"):
                self.status("partial_deadline")
                return
            self.status("prefill_complete", picked=picked)
            replay = self.probe("replay", reserve=1900)
            self.status("completed" if replay else "prefill_only_deadline")
        except Exception as e:
            self.status("failed", error=str(e))
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("launch", "supervise"))
    for name in ("output", "model", "data", "canonical", "stable", "mediation", "overlay", "runtime", "source", "base-source"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--deadline", type=int, required=True)
    p.add_argument("--devices", required=True)
    p.add_argument("--cohort-sha256", required=True)
    p.add_argument("--norm-only", action="store_true")
    args = p.parse_args()
    q = ReachQueue(args)
    if (socket.gethostname() != HOST or len(set(q.devices)) != 2
            or not 120 < args.deadline - time.time() <= 10800):
        raise ValueError("incorrect node, GPU assignment or deadline")
    if digest(q.root / "cohort.json") != args.cohort_sha256:
        raise ValueError("unverified cohort")
    source = load(args.source)
    for name, fingerprint in source["files"].items():
        if digest(q.src / name) != fingerprint:
            raise ValueError(f"source digest mismatch at {name}")
    if digest(q.root / "old-direction.json") != source["old_direction_sha256"]:
        raise ValueError("wrong fixed direction source")
    if args.action == "supervise":
        q.run()
        return
    check_devices(q.devices)
    with (q.root / "prefill-smoke.jsonl").open() as stream:
        head, *rows, end = map(json.loads, stream)
    phase = "norm-eval" if args.norm_only else "probe-dev"
    if (head["reach_sha256"] != source["files"][q.worker.name] or head["phase"] != phase or end["kind"] != "complete"
            or len(rows) != 3 or head["cohort_sha256"] != args.cohort_sha256
            or max(r["sampler_parity_max"] for r in rows) > 2e-7):
        raise ValueError("prefill smoke did not verify shipped source and sampler parity")
    q.lane.mkdir()
    with (q.lane / "supervisor.log").open("x") as log:
        child = subprocess.Popen([sys.executable, "-B", __file__, "supervise", *sys.argv[2:]],
                                 env=q.env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({"supervisor": child.pid, "deadline": args.deadline, "devices": q.devices,
                      "root": str(q.root), "cohort_sha256": args.cohort_sha256,
                      "source_sha256": digest(args.source)}))


if __name__ == "__main__":
    main()
