"""Paired native-prefix experiments that intervene on the reasoning exit time."""
import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import time

from repetition_baseline import render, sha
from repetition_canonical import score as canonical_score, seed_for
from repetition_canonical_queue import paired_interval
from repetition_exit import padded
from repetition_mediation import SAMPLING, identity, processor_factory, token_groups
from repetition_mediation_queue import load, save
from repetition_replay import terminal_cycle
from repetition_state_data import THINK_END, copies

E_ARMS = ("native", "e0", "e32")
METRICS = ("success", "numeric_correct", "exact_format", "answer_found", "eos", "terminal_cycle", "cap",
           "wrong_parsed_eos", "missing_answer_eos", "think_at_zero", "copy_exact", "eos_before_think")
CAP = 512


def post_seed(prompt, draw):
    return seed_for(prompt, draw) ^ 0x19595DF4C2A31F17


def generated_span(ids, stop_ids, post_cap=None):
    """Remove stopped-row padding before interpreting EOS at a forced arm's cap."""
    if post_cap is not None and THINK_END in ids:
        ids = ids[:ids.index(THINK_END) + 1 + post_cap]
    stop = next((i for i, tok in enumerate(ids) if tok in stop_ids), None)
    return (ids, None) if stop is None else (ids[:stop], ids[stop])


def periodic_here(prior):
    for period in range(1, min(64, len(prior) // 4) + 1):
        width = max(4 * period, 24)
        if len(prior) >= width and prior[-width:-period] == prior[-width + period:]:
            return True
    return False


def inputs_for(tokenizer, cases):
    output = []
    for x in cases:
        row = x["row"]
        thinking = render(tokenizer, row, "think")
        rendered_sha = hashlib.sha256(thinking.encode()).hexdigest()
        if x.get("render_sha256", rendered_sha) != rendered_sha:
            raise ValueError(f"canonical rendered prompt changed for {row['id']}")
        prior = x["completion_prefix"]
        if THINK_END in prior:
            raise ValueError("frozen state has already left reasoning")
        ids = tokenizer.encode(thinking, add_special_tokens=False) + prior
        plain = (tokenizer.encode(render(tokenizer, row, "no-think"), add_special_tokens=False)
                 if row["task"] == "copy" else None)
        output.append({**x, "pending": True, "native_ids": ids, "no_think_ids": plain,
                       "detector": periodic_here(prior), "render_sha256": rendered_sha})
    return output


def make_timing_policy(torch, transformers, groups):
    base = processor_factory(torch, transformers, groups)

    class TimingPolicy(base):
        def __init__(self, group, width, seeds, post, exit_at, bias):
            super().__init__(group, width, (), seeds[0], row_seeds=seeds)
            self.post = post
            self.reset = [False] * self.n
            self.exit_at = exit_at
            self.bias = bias

        def __call__(self, ids, scores):
            history = ids[:, self.width:]
            step = history.shape[1]
            if step and not all(self.reset):
                for i, mark in enumerate((history[:, -1] == THINK_END).tolist()):
                    if mark and not self.reset[i]:
                        self.row_generators[i].manual_seed(self.post[i])
                        self.reset[i] = True
            if not step and self.bias:
                scores = scores.float().clone()
                scores[:, THINK_END] += self.bias
            sampled = super().__call__(ids, scores)
            if step == self.exit_at:
                force = ~(history == THINK_END).any(dim=-1)
                forced = torch.full_like(sampled, -torch.inf)
                forced[:, THINK_END] = 0.
                sampled = torch.where(force[:, None], forced, sampled)
            return sampled

    class PostBudget(transformers.StoppingCriteria):
        def __init__(self, width, cap):
            self.width, self.cap = width, cap

        def __call__(self, ids, scores, **kwargs):
            part = ids[:, self.width:]
            marker = part == THINK_END
            first = marker.to(torch.int32).argmax(-1)
            return marker.any(-1) & ((part.shape[1] - first - 1) >= self.cap)

    return TimingPolicy, PostBudget


class Runtime:
    def __init__(self, args):
        import torch
        import transformers
        import tokenizers

        if torch.cuda.device_count() != 1:
            raise ValueError("one assigned CUDA device required")
        self.torch, self.transformers = torch, transformers
        identity(args)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        self.groups = token_groups(self.tokenizer)
        if self.groups["think"] != [THINK_END]:
            raise ValueError("unexpected thinking-end token")
        self.model = transformers.AutoModelForImageTextToText.from_pretrained(
            args.model, local_files_only=True, dtype=torch.bfloat16).to("cuda:0").eval()
        eos = self.model.generation_config.eos_token_id
        self.eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
        if not self.eos or None in self.eos:
            raise ValueError("native EOS missing")
        self.pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else min(self.eos)
        chat_end = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        if len(chat_end) != 1 or self.tokenizer.decode(chat_end) != "<|im_end|>":
            raise ValueError("expected one Qwen chat turn terminator")
        self.copy_eos = self.eos | set(chat_end)
        self.policy, self.stop_after = make_timing_policy(torch, transformers, self.groups)
        self.defaults = ({"use_model_defaults": False}
                         if "use_model_defaults" in inspect.signature(self.model.generate).parameters else {})
        files = load(args.data.parent / "manifest.json")["artifacts"]
        self.manifest = {"runtime": {"torch": str(torch.__version__), "transformers": transformers.__version__,
                                     "tokenizers": tokenizers.__version__},
                         "worker_sha256": sha(Path(__file__)), "eos_ids": sorted(self.eos),
                         "copy_eos_ids": sorted(self.copy_eos), "pad_id": self.pad, "sampling": SAMPLING,
                         "tokenizer_sha256": sha(args.model / "tokenizer.json"),
                         "checkpoint": {k: v["sha256"] for k, v in files.items() if k.endswith(".safetensors")}}

    def pack(self, ids):
        return padded(ids, self.pad, self.torch)

    def generate(self, batch, arm, draw, cap, hook=nullcontext(), bias=0.):
        no_think = arm == "no_think"
        exit_at = 0 if arm == "e0" else 32 if arm == "e32" else None
        is_copy = batch[0]["row"]["task"] == "copy"
        if any((x["row"]["task"] == "copy") != is_copy for x in batch):
            raise ValueError("copy and math require separate generation batches")
        stop_ids = self.copy_eos if is_copy else self.eos
        packed = self.pack([x["no_think_ids"] if no_think else x["native_ids"] for x in batch])
        width = packed["input_ids"].shape[1]
        seeds = [seed_for(x["row"]["id"], draw) for x in batch]
        post = [post_seed(x["row"]["id"], draw) for x in batch]
        sampler = self.policy(batch, width, seeds, post, exit_at, bias)
        total = cap if exit_at is None else cap + exit_at + 1
        cfg = self.transformers.GenerationConfig(do_sample=False, use_cache=True, max_new_tokens=total,
                                                  eos_token_id=sorted(stop_ids), pad_token_id=self.pad)
        criteria = [self.stop_after(width, cap)] if exit_at is not None else None
        with hook:
            out = self.model.generate(**packed, generation_config=cfg, logits_processor=[sampler],
                                      stopping_criteria=criteria, **self.defaults)
        results = []
        for i, (x, ids) in enumerate(zip(batch, out[:, width:].tolist(), strict=True)):
            ids_no_eos, stop = generated_span(ids, stop_ids, cap if exit_at is not None else None)
            at = ids_no_eos.index(THINK_END) if THINK_END in ids_no_eos else None
            if exit_at is not None and (at is None and stop is None or at is not None and at > exit_at):
                raise ValueError("timing arm violated the requested closure")
            if exit_at == 0 and at != 0:
                raise ValueError("E0 did not pass the close token as the first decode token")
            post_len = len(ids_no_eos) - at - 1 if at is not None else None
            if exit_at is not None and post_len is not None and post_len > cap:
                raise ValueError("post-close cap exceeded")
            content = self.tokenizer.decode(ids_no_eos, skip_special_tokens=False)
            row = x["row"]
            assessed = canonical_score(row, content, ids_no_eos, stop is not None, total, at)
            copy_raw = {}
            if row["task"] == "copy":
                output = self.tokenizer.decode(ids_no_eos[at + 1:], skip_special_tokens=False) if at is not None else content
                eligible = no_think or at is not None
                exact = eligible and output.strip() == row["gold"]
                found = eligible and bool(output.strip())
                assessed |= {"copy_exact": exact, "success": exact and stop is not None,
                             "answer_found": found, "missing_answer_eos": stop is not None and not found}
                copy_raw = {"rendered_answer_sample": output.strip()[-120:], "gold": row["gold"],
                            "copy_error": copy_error(output, row["gold"], eligible, stop is not None, exact)}
            if exit_at is not None:
                assessed["cap"] = stop is None and post_len == cap
            results.append({"kind": "rollout", "id": row["id"], "arm": arm, "draw": draw, "tokens": ids_no_eos,
                            "seed": seeds[i], "post_seed": post[i], "eos_token": stop,
                            "first_think_mass": sampler.first_step["think"][i],
                            "forced_step": exit_at if at is not None and at == exit_at else None,
                            "eos_before_think": stop is not None and at is None and not no_think,
                            **assessed, **copy_raw})
        return results


def copy_error(output, gold, eligible, eos, exact):
    if not eligible:
        return "no_think_exit"
    if exact:
        return "exact_eos" if eos else "exact_without_eos"
    if not output.strip():
        return "empty"
    if output.split() == gold.split():
        return "whitespace"
    if all(w == gold.split()[0] for w in output.split()):
        return "wrong_repeat_count"
    return "extra_or_wrong_text"


def phase_input(root, phase):
    if phase == "copy-dev":
        return copies("dev"), ("native", "no_think"), 192, 2, None
    if phase == "copy-test":
        plan = root / "copy-dev-summary.json"
        choice = load(plan)["copy_selection"]
        return copies("test", choice["repetitions"]), tuple(choice["arms"]), 192, 2, sha(plan)
    cohort = root / "cohort.json"
    return load(cohort)[phase.removeprefix("labels-")], E_ARMS, CAP, 4, sha(cohort)


def delivery_header(runtime, args, rows, arms, cap, draws, total, source):
    return {"kind": "manifest", "phase": args.phase, "part": args.part, "partitions": args.partitions,
            "cases": [{"id": x["row"]["id"], "stratum": x["stratum"], "length": len(x["completion_prefix"]),
                       "detector": x["detector"], "repetitions": x.get("repetitions"),
                       "render_sha256": x["render_sha256"]} for x in rows],
            "arms": list(arms), "cap": cap, "draws": draws, "total": total, "cohort_sha256": source,
            **runtime.manifest}


def run(args):
    cases, arms, cap, draws, source = phase_input(args.root, args.phase)
    if args.smoke:
        cases, cap, draws = cases[:2], 8, 1
    r = Runtime(args)
    items = inputs_for(r.tokenizer, cases[args.part::args.partitions])
    header = delivery_header(r, args, items, arms, cap, draws, len(cases), source)
    start = time.monotonic()
    with args.output.open("x") as f, r.torch.inference_mode():
        print(json.dumps(header), file=f, flush=True)
        for draw in range(draws):
            for base in range(0, len(items), 6):
                batch = items[base:base + 6]
                for arm in arms:
                    if time.time() >= args.deadline - 60:
                        raise TimeoutError("worker hard deadline")
                    for row in r.generate(batch, arm, draw, cap):
                        print(json.dumps(row, allow_nan=False), file=f)
                f.flush()
                print(f"{args.phase} draw={draw} prompts={base + len(batch)}/{len(items)}", flush=True)
        print(json.dumps({"kind": "complete", "seconds": time.monotonic() - start}), file=f)


def read_matrix(root, phase):
    receipt = load(root / "queue" / f"{phase}-done.json")
    contract, parts, cases, rows = None, set(), {}, {}
    for filename, digest in receipt["outputs"]:
        path = Path(filename)
        if sha(path) != digest:
            raise ValueError("worker output SHA mismatch")
        with path.open() as s:
            h = json.loads(next(s))
            core = {k: v for k, v in h.items() if k not in ("part", "cases")}
            if h["phase"] != phase or h["part"] in parts or (contract and core != contract):
                raise ValueError("experiment contract mismatch")
            contract = core
            parts.add(h["part"])
            selected = {x["id"]: x for x in h["cases"]}
            if selected.keys() & cases.keys():
                raise ValueError("overlapping prompt partitions")
            cases.update(selected)
            done = False
            for line in s:
                row = json.loads(line)
                if row["kind"] == "complete":
                    done = True
                    continue
                key = row["id"], row["draw"], row["arm"]
                if (row["id"] not in selected or key in rows or row["seed"] != seed_for(*key[:2])
                        or row["post_seed"] != post_seed(*key[:2])):
                    raise ValueError("invalid or duplicate paired row")
                after = row["after_think_tokens"]
                if row["arm"] in ("e0", "e32") and after is not None and (
                        after > h["cap"] or (after == h["cap"] and row["eos"])):
                    raise ValueError("timing row includes impossible tokens after its matched budget")
                rows[key] = row
            if not done:
                raise ValueError("worker missed completion footer")
    expected = {(k, draw, arm) for k in cases for draw in range(contract["draws"]) for arm in contract["arms"]}
    if rows.keys() != expected or len(cases) != contract["total"] or parts != set(range(contract["partitions"])):
        raise ValueError("incomplete paired matrix")
    return contract, cases, rows


def median(values):
    return statistics.median(values) if values else None


def summarize(root, phase, pairs=()):
    contract, cases, rows = read_matrix(root, phase)
    draws = range(contract["draws"])
    arms = contract["arms"]
    strata = {"all": list(cases)}
    for key, item in cases.items():
        strata.setdefault(item["stratum"], []).append(key)
        if item["stratum"] != "copy":
            n = item["length"]
            length_bin = "lt_1024" if n < 1024 else "1024_3071" if n < 3072 else "gte_3072"
            strata.setdefault(length_bin, []).append(key)
    output = {"protocol": contract, "rows": len(rows), "strata": {}}
    if not pairs and "e0" in arms:
        pairs = (("e0", "e32"), ("e0", "native"), ("e32", "native"))
    for label, keys in strata.items():
        cells = {}
        for arm in arms:
            observations = [rows[p, draw, arm] for p in keys for draw in draws]
            cells[arm] = {metric: sum(bool(r[metric]) for r in observations) for metric in METRICS}
            for f in ("think_step", "after_think_tokens"):
                cells[arm]["median_" + f] = median([r[f] for r in observations if r[f] is not None])
            cells[arm]["forced_steps"] = sum(r["forced_step"] is not None for r in observations)
        contrasts = {f"{a}_minus_{b}": {metric: paired_interval({
            p: [int(rows[p, draw, a][metric]) - int(rows[p, draw, b][metric]) for draw in draws] for p in keys})
            for metric in ("success", "think_at_zero", "wrong_parsed_eos", "missing_answer_eos")}
            for a, b in pairs}
        output["strata"][label] = {"prompts": len(keys), "per_arm": len(keys) * len(draws),
                                   "arms": cells, "paired": contrasts}
    if "e0" in arms:
        endpoints = {}
        for p in cases:
            advantage = statistics.fmean(int(rows[p, d, "e0"]["success"]) - int(rows[p, d, "e32"]["success"])
                                         for d in draws)
            native = rows[p, 0, "native"]
            endpoints[p] = {**cases[p], "delta": advantage,
                            "p_first_native": native["first_think_mass"][0],
                            "p_first_sampled": native["first_think_mass"][2]}
        output["labels"] = endpoints
        for key, q in strata.items():
            output["strata"][key]["advantage_sign"] = {
                k: sum(test(endpoints[p]["delta"]) for p in q)
                for k, test in (("early", lambda v: v > 0), ("wait", lambda v: v < 0), ("tie", lambda v: not v))}
    if phase.startswith("copy-"):
        errors = {arm: dict(Counter(r["copy_error"] for r in rows.values() if r["arm"] == arm)) for arm in arms}
        output["errors"] = errors
        output["examples"] = {arm: [[r["copy_error"], r["gold"], r["rendered_answer_sample"]]
                                    for r in rows.values() if r["arm"] == arm and not r["success"]][:3] for arm in arms}
    if phase == "copy-dev":
        choices = {}
        for arm in arms:
            choices[arm] = {}
            for n in range(1, 5):
                matched = [p for p, x in cases.items() if x["repetitions"] == n]
                flags = [rows[p, d, arm]["success"] for p in matched for d in draws]
                good = sum(flags)
                choices[arm][n] = {"good": good, "total": len(flags)}
        # Freeze the hardest level that has a measurable native no-thinking baseline.
        qualified = [n for n, x in choices["no_think"].items() if x["good"] >= .25 * x["total"]]
        n = max(qualified, default=None)
        readable = n is not None and choices["native"][n]["good"] >= .25 * choices["native"][n]["total"]
        output["copy_dev"] = choices
        output["copy_selection"] = ({"repetitions": n, "arms": ["no_think", "native"] if readable else ["no_think"]}
                                    if n is not None else None)
    save(root / f"{phase}-summary.json", output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("run", "summarize"))
    p.add_argument("--phase", choices=("labels-dev", "labels-eval", "copy-dev", "copy-test"), required=True)
    p.add_argument("--part", type=int)
    p.add_argument("--partitions", type=int)
    p.add_argument("--deadline", type=int)
    p.add_argument("--smoke", action="store_true")
    for k in ("root", "model", "data", "output"):
        p.add_argument("--" + k, type=Path)
    args = p.parse_args()
    if args.action == "run":
        run(args)
    else:
        summarize(args.root, args.phase)


if __name__ == "__main__":
    main()
