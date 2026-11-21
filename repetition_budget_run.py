"""Pair bounded decoder trajectories and keep raw IDs for nested budget scoring."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

from repetition_baseline import sha
from repetition_canonical import seed_for
from repetition_mediation_queue import Supervisor, check_devices, digest, load, save
from repetition_state_queue import HOST
from repetition_state_value import EXIT_STEPS, Runtime, THINK_END, inputs_for, post_seed

ARMS = ("native", "e0", "e32", "e128")
QUESTION = "question_e0"
CAP = 1536
BATCH_SIZE = 15
TARGET_DRAWS = 4
BASE_COMMIT = "41a7b909b724d0b6eafa2e34eb853f6255429cd8"


def tokens(row):
    return row["tokens"] + ([row["eos_token"]] if row["eos_token"] is not None else [])


def common_trunk(arms, total_native_cap):
    checks = {}
    for a, b, fork in (("e32", "e128", 32), ("native", "e32", 32), ("native", "e128", 128)):
        x, y = arms[a], arms[b]
        n = min(fork, total_native_cap) if a == "native" else fork
        checks[a + ":" + b] = {"before_fork": tokens(x)[:n] == tokens(y)[:n], "matched_post": None}
        t = x["think_step"]
        if t is not None and t < n and y["think_step"] == t:
            # Native can hit its smaller total cap after a natural close.
            left, right = tokens(x), tokens(y)
            m = min(len(left), len(right)) if a == "native" else max(len(left), len(right))
            checks[a + ":" + b]["matched_post"] = left[t+1:m] == right[t+1:m]
    return checks


def packed_hash(ids):
    return sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def run(args):
    cohort_path = args.root / "cohort.json"
    plan = load(cohort_path)
    rows = plan["cases"][args.part::args.partitions]
    question = set(plan["question_only"])
    cap = args.cap
    if args.action == "smoke":
        # Use short, medium and long prefixes and run question-only on each.
        ordered = sorted(plan["cases"], key=lambda x: len(x["completion_prefix"]))
        rows = [ordered[0], ordered[len(ordered)//2], ordered[-1]]
        question = {x["row"]["id"] for x in rows}
        cap = 12
    r = Runtime(args)
    items = inputs_for(r.tokenizer, rows)
    thin = inputs_for(r.tokenizer, [x | {"completion_prefix": []} for x in rows if x["row"]["id"] in question])
    header = {"kind": "manifest", "wave": args.wave, "part": args.part, "partitions": args.partitions,
              "native_total_cap": cap, "forced_post_cap": cap, "draw": args.wave,
              "thresholds": EXIT_STEPS, "base_commit": BASE_COMMIT, "batch_size": BATCH_SIZE,
              "cohort_sha256": sha(cohort_path), "budget_worker_sha256": sha(Path(__file__)),
              "cases": [{"id": x["row"]["id"], "stratum": x["stratum"], "eval_origin": x["eval_origin"],
                         "length": len(x["completion_prefix"]), "render_sha256": x["render_sha256"],
                         "native_sha256": packed_hash(x["native_ids"])} for x in items],
              "question_only": [{"id": x["row"]["id"], "prefix_sha256": packed_hash(x["native_ids"])}
                                for x in thin], **r.manifest}
    start = time.monotonic()
    with args.output.open("x") as f, r.torch.inference_mode():
        def emit(row):
            print(json.dumps(row, allow_nan=False), file=f)
        emit(header)
        for begin in range(0, len(items), BATCH_SIZE):
            batch = items[begin:begin+BATCH_SIZE]
            packed = r.pack([x["native_ids"] for x in batch])
            per_arm = {}
            for arm in ARMS:
                if time.time() >= args.deadline - 45:
                    raise TimeoutError("timing-budget worker deadline")
                per_arm[arm] = r.generate(batch, arm, args.wave, cap, packed=packed)
                for row in per_arm[arm]:
                    emit(row)
                f.flush()
                print(f"draw={args.wave} prompts={begin+len(batch)}/{len(items)} arm={arm}", flush=True)
            for i, item in enumerate(batch):
                emit({"kind": "trunk", "id": item["row"]["id"], "draw": args.wave,
                      "checks": common_trunk({k: v[i] for k, v in per_arm.items()}, cap)})
        if thin:
            if time.time() >= args.deadline - 45:
                raise TimeoutError("timing-budget question-only deadline")
            for row in r.generate(thin, "e0", args.wave, cap):
                emit(row | {"arm": QUESTION})
        emit({"kind": "complete", "seconds": time.monotonic()-start})


def verify_smoke(root, cohort_sha):
    with (root/"smoke.jsonl").open() as f:
        h, *rows, end = map(json.loads, f)
    if (h["budget_worker_sha256"] != sha(Path(__file__)) or h["cohort_sha256"] != cohort_sha
            or h["forced_post_cap"] != 12 or end["kind"] != "complete"):
        raise ValueError("incorrect smoke protocol")
    generated = [x for x in rows if x["kind"]=="rollout"]
    if len(generated)!=15:
        raise ValueError("canary did not complete the five-arm matrix")
    for r in generated:
        arm, t = r["arm"], r["think_step"]
        if r["seed"] != seed_for(r["id"],0) or r["post_seed"] != post_seed(r["id"],0):
            raise ValueError("invalid paired sample seed")
        if arm in ("e0", QUESTION) and (t!=0 or r["tokens"][0]!=THINK_END):
            raise ValueError("failed to emit the forced token at step 0")
        if arm in EXIT_STEPS and (t is not None and (t > EXIT_STEPS[arm] or r["after_think_tokens"] > 12)):
            raise ValueError("forced continuation exceeded its cap")
        if r["eos"] and (r["eos_token"] not in h["eos_ids"] or r["after_think_tokens"]==12):
            raise ValueError("pad-generated EOS in smoke")
    checks = [x for x in rows if x["kind"]=="trunk"]
    if len(checks)!=3 or any(not k["before_fork"] or k["matched_post"] is False
                             for x in checks for k in x["checks"].values()):
        raise ValueError("canary failed common-trunk parity")


class BudgetQueue(Supervisor):
    def __init__(self, args):
        args.role = "queue"
        super().__init__(args)
        self.entry = Path(__file__).resolve()
        self.env["PYTHONPATH"] = args.runtime_path

    def supervise(self):
        try:
            self.status("admitted", cohort_sha256=digest(self.root/"cohort.json"), completed_draws=[])
            for draw in range(TARGET_DRAWS):
                remaining = self.args.deadline-time.time()
                elapsed = self.state.get("last_draw_seconds")
                if remaining < (300 if elapsed is None else 1.12*elapsed+75):
                    self.status("budget_complete", cutoff_reason="insufficient time for a full paired draw")
                    return
                phase = f"draw{draw}"
                plans = []
                for lane in range(2):
                    output = self.lane/f"{phase}-{lane:02d}.jsonl"
                    command = [self.entry, "run", "--root", self.root, "--model", self.args.model,
                               "--data", self.args.data,"--deadline",self.args.deadline,"--output",output,
                               "--part",lane,"--partitions",2,"--wave",draw]
                    plans.append((lane, output, command))
                started = time.monotonic()
                receipts = self.run_batch(phase, plans)
                save(self.lane/f"{phase}-done.json", {"outputs": [[str(p), h] for p,h in receipts]})
                self.status(phase, completed_draws=[*range(draw+1)], last_draw_seconds=time.monotonic()-started)
            self.status("completed")
        except Exception as exc:
            self.status("failed", error=str(exc))
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("run","smoke","launch","supervise"))
    for name in ("root","model","data","output","source","overlay","stable"):
        p.add_argument("--"+name, type=Path)
    for name in ("part","partitions","deadline","wave","cap"):
        p.add_argument("--"+name, type=int, default={"wave":0,"cap":CAP}.get(name))
    for name in ("runtime-path","devices","cohort-sha256"):
        p.add_argument("--"+name)
    args = p.parse_args()
    if args.action in ("run","smoke"):
        run(args)
        return
    args.output = args.root
    q = BudgetQueue(args)
    if socket.gethostname() != HOST or len(set(q.devices)) != 2 or not 180 < args.deadline-time.time() <= 3600:
        raise ValueError("wrong node, GPUs or budget deadline")
    if digest(args.root/"cohort.json") != args.cohort_sha256:
        raise ValueError("frozen cohort digest mismatch")
    sources = load(args.source)
    if any(digest(q.entry.parent/name)!=value for name,value in sources.items()):
        raise ValueError("staged program SHA mismatch")
    if args.action == "supervise":
        q.supervise()
        return
    check_devices(q.devices)
    verify_smoke(args.root,args.cohort_sha256)
    q.lane.mkdir()
    with (q.lane/"supervisor.log").open("x") as f:
        child = subprocess.Popen([sys.executable,"-B",__file__,"supervise",*sys.argv[2:]],
                                 start_new_session=True, env=q.env, stdout=f, stderr=subprocess.STDOUT)
    print(json.dumps({"pid":child.pid,"deadline":args.deadline,"devices":q.devices,"root":str(q.root),
                      "cohort_sha256":args.cohort_sha256,"source_sha256":digest(args.source)}))


if __name__ == "__main__":
    main()
