"""Schedule two-node controls for premature reasoning termination."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys

from repetition_mediation_queue import Supervisor, check_devices, digest, load, save
from repetition_replay import ASTRA_REVISION, terminal_cycle

HOSTS = {"collection": "model-training-1-0-1", "mechanism": "textsum-vdt21-0-1"}
TARGET_LOOP, TARGET_CLEAN = 128, 64
THINK_END = 248069


def discover_cases(path, checked_sha):
    if digest(path) != checked_sha:
        raise ValueError(f"baseline digest mismatch: {path}")
    with path.open() as stream:
        head = json.loads(next(stream))
        if (head["model_revision"] != ASTRA_REVISION or head["cap"] != 8192 or head["shard"] < 12
                or head["modes"] != ["think"] or head["decoders"] != ["astra"] or len(head["panel"]) != 32):
            raise ValueError(f"baseline manifest mismatch: {path}")
        panel = {r["id"]: r for r in head["panel"]}
        loops, clean, seen = [], [], set()
        completed = False
        for line in stream:
            record = json.loads(line)
            if record["record"] == "completed":
                completed = True
                continue
            if record["record"] != "rollout" or record["draw"] != 0 or record["id"] in seen:
                raise ValueError("unexpected or duplicate baseline record")
            seen.add(record["id"])
            tokens = record["token_ids"]
            cycle = terminal_cycle(tokens)
            row = panel[record["id"]]
            common = {"row": row, "origin": path.name, "budget": 41}
            if cycle:
                period, end = cycle
                prefix = tokens[:end]
                if THINK_END not in prefix:
                    loops.append({**common, "completion_prefix": prefix, "period": period,
                                  "cycle": tokens[end - period:end], "stratum": "loop"})
            elif record["outcomes"]["8192"]["eos"] and THINK_END in tokens:
                end = tokens.index(THINK_END)
                prefix = tokens[:max(0, end - 128)]
                if not terminal_cycle(prefix):
                    clean.append({**common, "completion_prefix": prefix, "period": 0, "cycle": [],
                                  "stratum": "clean", "reference_correct": record["outcomes"]["8192"]["correct"]})
        if not completed or seen != panel.keys():
            raise ValueError("incomplete discovery shard")
    return loops, clean


class ExitSupervisor(Supervisor):
    def __init__(self, args):
        super().__init__(args)
        self.env["PYTHONPATH"] = f"{args.runtime}:{args.stable}:{args.mediation}:{args.overlay}"
        self.worker = self.src / "repetition_exit.py"

    def collect(self):
        loop, clean, shards = {}, {}, []
        for wave in range(11):
            self.status("assembling_cohort", loops=len(loop), clean=len(clean))
            # Keep the later intervention phases inside the same wall-clock budget.
            if (len(loop) >= TARGET_LOOP and len(clean) >= TARGET_CLEAN) or not self.bounded(8 * 3600):
                break
            plans = []
            for gpu in range(2):
                shard = 12 + 2 * wave + gpu
                output = self.lane / f"baseline-{shard:02d}.jsonl"
                cmd = [self.args.stable / "repetition_baseline.py", "--astra-math", "--mode", "think",
                       "--decoder", "astra", "--model", self.args.model, "--data", self.args.data,
                       "--output", output, "--deadline", self.args.deadline, "--shard", shard,
                       "--count", 32, "--batch", 8, "--cap", 8192]
                plans.append((gpu, output, cmd))
            done = self.run_batch(f"baseline_wave_{wave + 1}", plans, tolerate_failure=True)
            if not done:
                raise RuntimeError("both discovery workers failed; see per-GPU logs")
            for file, sha in done:
                loops, clean_rows = discover_cases(file, sha)
                for target, rows in ((loop, loops), (clean, clean_rows)):
                    for row in rows:
                        key = row["row"]["id"]
                        if key in loop or key in clean:
                            raise ValueError("overlapping discovery shards")
                        target[key] = row
                shards.append([str(file), sha])
        if len(loop) < 64 or len(clean) < 24:
            raise RuntimeError(f"insufficient frozen cohort ({len(loop)} loop, {len(clean)} clean)")
        key = lambda item: hashlib.sha256(item["row"]["id"].encode()).digest()
        rows = sorted(loop.values(), key=key)[:TARGET_LOOP] + sorted(clean.values(), key=key)[:TARGET_CLEAN]
        save(self.root / "cohort.json", {"cases": rows, "pool": {"loop": len(loop), "clean": len(clean)},
                                        "shards": shards})
        self.status("cohort_frozen", loops=min(TARGET_LOOP, len(loop)), clean=min(TARGET_CLEAN, len(clean)))

    def phase(self, name, partitions, shift):
        manifest = self.wait_for(self.src / "exit-ready.json", "waiting_for_code")
        if digest(self.worker) != manifest["sha256"]:
            raise ValueError("worker SHA-256 differs from admission")
        plans = []
        for gpu in range(2):
            part = shift + gpu
            output = self.lane / f"{name}-{part:02d}.jsonl"
            cmd = [self.worker, "run", "--phase", name, "--part", part, "--partitions", partitions,
                   "--output", output, "--root", self.root, "--model", self.args.model, "--old", self.args.old,
                   "--data", self.args.data, "--deadline", self.args.deadline]
            plans.append((gpu, output, cmd))
        outputs = self.run_batch(name, plans)
        save(self.lane / f"{name}-done.json", {"outputs": [[str(f), sha] for f, sha in outputs]})

    def summary(self, phase):
        env = {**self.env, "CUDA_VISIBLE_DEVICES": ""}
        subprocess.run([sys.executable, "-B", str(self.worker), "summarize", "--phase", phase,
                        "--root", str(self.root)], env=env, check=True, timeout=140)

    def run(self):
        try:
            collector = self.args.role == "collection"
            if collector:
                self.collect()
            else:
                self.phase("dev", 2, 0)
                self.summary("dev")
                self.phase("localize", 2, 0)
                self.summary("localize")
            self.wait_for(self.root / "cohort.json", "waiting_for_cohort")
            self.wait_for(self.root / "localize-summary.json", "waiting_for_localization")
            self.wait_for(self.root / "dev-summary.json", "waiting_for_calibration")
            for name, reserve in (("heldout", 110 * 60), ("long", 75 * 60), ("patch", 50 * 60)):
                if not self.bounded(reserve):
                    self.status("partial_deadline", next_phase=name)
                    return
                self.phase(name, 4, 0 if collector else 2)
                if collector:
                    self.wait_for(self.root / "mechanism" / f"{name}-done.json", f"waiting_for_{name}_peer")
                    self.summary(name)
                else:
                    self.wait_for(self.root / f"{name}-summary.json", f"waiting_for_{name}_summary")
            self.status("completed")
        except Exception as exc:
            self.status("failed", error=str(exc))
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("launch", "supervise"))
    p.add_argument("--role", choices=tuple(HOSTS), required=True)
    for key in ("output", "stable", "overlay", "runtime", "model", "data", "mediation", "old"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--deadline", type=int, required=True)
    p.add_argument("--devices", required=True)
    args = p.parse_args()
    worker = ExitSupervisor(args)
    import time
    if (socket.gethostname() != HOSTS[args.role] or len(set(worker.devices)) != 2
            or not 120 < args.deadline - time.time() <= 43200):
        raise ValueError("invalid allocation")
    if args.action == "launch":
        check_devices(worker.devices)
        worker.lane.mkdir()
        with (worker.lane / "supervisor.log").open("x") as log:
            proc = subprocess.Popen([sys.executable, "-B", __file__, "supervise", *sys.argv[2:]], env=worker.env,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"supervisor": proc.pid, "role": args.role, "deadline": args.deadline,
                          "devices": worker.devices, "output": str(worker.root)}))
    else:
        worker.run()


if __name__ == "__main__":
    main()
