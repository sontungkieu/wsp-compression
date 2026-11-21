"""Run the paired Astra-prompt audit and canonical replay on two assigned GPUs."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import socket
import statistics
import subprocess
import sys
import time

from repetition_baseline import POLICIES
from repetition_mediation_queue import Supervisor, check_devices, digest, load, save
from repetition_replay import ASTRA_REVISION, terminal_cycle

HOST = "model-training-1-0-1"
THINK_END = 248069
CAP = 8192
SHARDS = ((34, 35), (36, 37), (38, 39), (40,))
BASE_METRICS = ("terminal_cycle", "answer_found", "numeric_correct", "exact_format", "eos", "cap")


def paired_interval(differences, seed=87131):
    means = [statistics.fmean(differences[k]) for k in sorted(differences) if differences[k]]
    if not means:
        return None
    rng = random.Random(seed)
    boot = sorted(statistics.fmean(rng.choices(means, k=len(means))) for _ in range(3000))
    return {"delta_pp": round(100 * statistics.fmean(means), 2),
            "ci95_pp": [round(100 * boot[i], 2) for i in (75, 2924)], "prompts": len(means)}


def audit_pair(path, count, cap, source_sha):
    with path.open() as stream:
        manifest = json.loads(next(stream))
        rows = {r["id"]: r for r in manifest["panel"]}
        expected = {(k, arm) for k in rows for arm in ("canonical", "legacy")}
        if (len(rows) != count or manifest["cap"] != cap or manifest["source_sha"] != source_sha
                or manifest["model_revision"] != ASTRA_REVISION or manifest["sample_seeds"] != 1
                or manifest["modes"] != ["think"] or manifest["prompt_arms"] != ["canonical", "legacy"]
                or tuple(manifest["decoder_settings"]["astra"].values()) != POLICIES["astra"]):
            raise ValueError(f"unexpected paired manifest {path.name}")
        seen, completed = {}, False
        for line in stream:
            r = json.loads(line)
            if r["record"] == "completed":
                completed = True
                continue
            key = r["id"], r["prompt_arm"]
            if (key not in expected or key in seen or r["decoder"] != "astra" or r["draw"] != 0
                    or len(r["render_sha256"]) != 64):
                raise ValueError(f"unexpected rollout in {path.name}")
            seen[key] = r | {"eos": r["outcomes"][str(cap)]["eos"], "cap": r["outcomes"][str(cap)]["cap"]}
        if seen.keys() != expected or not completed:
            raise ValueError(f"incomplete paired shard {path.name}")
        if any(seen[k, "canonical"]["render_sha256"] == seen[k, "legacy"]["render_sha256"] for k in rows):
            raise ValueError("canonical and legacy render hashes must differ")
    return manifest, seen


def rank_id(item):
    return hashlib.sha256(item["row"]["id"].encode()).digest()


def prefix_pool(paths, baseline_sha, excluded):
    loops, clean, stats = [], [], {}
    checked = []
    for f, checksum in paths:
        if digest(f) != checksum:
            raise ValueError("baseline shard hash mismatch")
        head, rows = audit_pair(f, 32, CAP, baseline_sha)
        if head["shard"] not in {k for wave in SHARDS for k in wave}:
            raise ValueError("unexpected paired shard number")
        checked.append([str(f), checksum])
        for row in head["panel"]:
            key = row["id"]
            if key in stats or key in excluded:
                raise ValueError(f"non-independent prompt ID {key}")
            if row.get("astra_prompt_arm") != "canonical" or chr(8) in row["prompt"]:
                raise ValueError("cohort source is not canonical")
            stats[key] = {a: {m: bool(rows[key, a][m]) for m in BASE_METRICS} for a in ("canonical", "legacy")}
            r = rows[key, "canonical"]
            tokens = r["token_ids"][:-1] if r["eos"] else r["token_ids"]
            cycle = terminal_cycle(tokens)
            common = {"row": row, "origin": f.name, "render_sha256": r["render_sha256"], "budget": 41}
            if cycle:
                period, cut = cycle
                prior = tokens[:cut]
                if THINK_END not in prior:
                    loops.append({**common, "completion_prefix": prior, "stratum": "loop",
                                  "period": period, "cut": cut})
            elif r["eos"] and THINK_END in tokens:
                clean.append({**common, "trace_before_close": tokens[:tokens.index(THINK_END)],
                              "stratum": "clean", "period": 0})
    return sorted(loops, key=rank_id), sorted(clean, key=rank_id), stats, checked


def freeze_cohort(loops, clean):
    retained = loops[:48]
    lengths = sorted(x["cut"] for x in retained)
    matched = []
    for i, case in enumerate(clean):
        if len(matched) == 24:
            break
        goal = lengths[i % len(lengths)] if lengths else 128
        trace = case.pop("trace_before_close")
        cut = min(goal, max(0, len(trace) - 128))
        prior = trace[:cut]
        if terminal_cycle(prior):
            continue
        matched.append({**case, "completion_prefix": prior, "cut": cut})
    return retained + matched


def baseline_summary(per_prompt):
    total = len(per_prompt)
    arms = {a: {m: sum(s[a][m] for s in per_prompt.values()) for m in BASE_METRICS}
            for a in ("canonical", "legacy")}
    paired = {m: paired_interval({k: [int(s["canonical"][m]) - int(s["legacy"][m])]
                                  for k, s in per_prompt.items()}) for m in BASE_METRICS}
    return {"prompts": total, "arms": arms, "canonical_minus_legacy": paired}


class CanonicalQueue(Supervisor):
    def __init__(self, args):
        args.role = "queue"
        super().__init__(args)
        self.src = Path(__file__).resolve().parent
        self.env["PYTHONPATH"] = f"{self.src}:{args.runtime}:{args.stable}:{args.mediation}:{args.overlay}"
        self.base = self.src / "repetition_baseline.py"
        self.worker = self.src / "repetition_canonical.py"

    def base_cmd(self, output, shard):
        return [self.base, "--model", self.args.model, "--data", self.args.data, "--output", output,
                "--deadline", self.args.deadline, "--mode", "think", "--decoder", "astra",
                "--astra-math", "--astra-legacy-audit", "--shard", shard, "--count", 32, "--batch", 8, "--cap", CAP]

    def discovery(self):
        old_ids = {x["row"]["id"] for x in load(self.args.old / "cohort.json")["cases"]}
        old_ids |= {x["row"]["id"] for x in load(self.args.prior / "holdout.json")["cases"]}
        paths = []
        for wave, shards in enumerate(SHARDS, 1):
            if wave > 1 and not self.bounded(43 * 60):
                break
            plans = []
            for gpu, shard in enumerate(shards):
                f = self.lane / f"paired-{shard:02d}.jsonl"
                plans.append((gpu, f, self.base_cmd(f, shard)))
            paths += self.run_batch(f"baseline_wave_{wave}", plans)
            loops, clean, stats, checked = prefix_pool(paths, self.args.baseline_sha, old_ids)
            save(self.root / "baseline-summary.json", baseline_summary(stats) | {"shards": checked})
            self.status("baseline_verified", wave=wave, baseline_prompts=len(stats), loop_pool=len(loops),
                        clean_pool=len(clean))
            if len(loops) >= 32 and len(clean) >= 24:
                break
        frozen = freeze_cohort(loops, clean)
        pool = dict(Counter(x["stratum"] for x in frozen))
        if not frozen:
            raise RuntimeError("no canonical cases qualified")
        save(self.root / "cohort.json", {"cases": frozen, "counts": pool, "loop_pool": len(loops),
                                         "clean_pool": len(clean), "sources": checked})
        self.status("cohort_frozen", cohort=pool)

    def stage(self, phase, total, reserve):
        if not self.bounded(reserve):
            self.status(skipped=self.state.get("skipped", []) + [phase])
            return None
        marker = self.wait_for(self.src / "canonical-ready.json", "waiting_for_replay_worker")
        for name, sha in marker["sources"].items():
            if digest(self.src / name) != sha:
                raise ValueError(f"source release digest mismatch: {name}")
        partitions = min(2, total)
        plans = []
        for part in range(partitions):
            output = self.lane / f"{phase}-{part:02d}.jsonl"
            params = [self.worker, "run", "--phase", phase, "--partitions", partitions, "--part", part,
                      "--root", self.root, "--model", self.args.model, "--data", self.args.data,
                      "--output", output, "--deadline", self.args.deadline]
            plans.append((part, output, params))
        shards = self.run_batch(phase, plans)
        save(self.lane / f"{phase}-done.json", {"outputs": [[str(path), sha] for path, sha in shards]})
        env = {**self.env, "CUDA_VISIBLE_DEVICES": ""}
        subprocess.run([sys.executable, "-B", str(self.worker), "summarize", "--phase", phase, "--root",
                        str(self.root)], env=env, timeout=120, check=True)
        return load(self.root / f"{phase}-summary.json")

    def run(self):
        try:
            self.discovery()
            cohort = load(self.root / "cohort.json")
            count = len(cohort["cases"])
            copy = self.stage("copy-dev", 18, 25 * 60)
            replay = self.stage("main", count, 18 * 60)
            if replay:
                loop = replay["strata"].get("loop", {})
                by = loop.get("arms", {})
                gate = (loop.get("prompts", 0) >= 8 and by["left"]["success"] > by["native"]["success"]
                        and by["think_transfer"]["success"] > by["blank"]["success"])
                self.status(mechanism_gate=gate)
                if gate:
                    self.stage("mechanism", count, 22 * 60)
                if by and any(v["cap"] * 2 >= loop["rollouts_per_arm"] for v in by.values()):
                    self.stage("long", count, 22 * 60)
            if copy and copy.get("selected_repeats") is not None:
                self.stage("copy-test", 16, 5 * 60)
            self.status("completed" if replay else "partial_deadline")
        except Exception as e:
            self.status("failed", error=str(e))
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("launch", "supervise"))
    for k in ("output", "model", "data", "stable", "overlay", "runtime", "mediation", "old", "prior"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--baseline-sha", required=True)
    p.add_argument("--deadline", type=int, required=True)
    p.add_argument("--devices", required=True)
    args = p.parse_args()
    queue = CanonicalQueue(args)
    if (socket.gethostname() != HOST or len(set(queue.devices)) != 2
            or not 120 < args.deadline - time.time() <= 10800):
        raise ValueError("invalid node, allocation or deadline")
    if digest(queue.base) != args.baseline_sha:
        raise ValueError("baseline source changed")
    if args.action == "launch":
        check_devices(queue.devices)
        audit_pair(queue.root / "canary.jsonl", 2, 32, args.baseline_sha)
        queue.lane.mkdir()
        with (queue.lane / "supervisor.log").open("x") as log:
            proc = subprocess.Popen([sys.executable, "-B", __file__, "supervise", *sys.argv[2:]], env=queue.env,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"supervisor": proc.pid, "root": str(queue.root), "deadline": args.deadline,
                          "devices": queue.devices, "baseline_sha256": args.baseline_sha}))
    else:
        queue.run()


if __name__ == "__main__":
    main()
