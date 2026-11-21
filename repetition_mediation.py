"""Replay Qwen loops while exchanging selected next-token probability masses."""
import argparse
from collections import defaultdict
import hashlib
import inspect
import json
from pathlib import Path
import random
import statistics
import time

from repetition_baseline import assess, render, sha
from repetition_replay import (ASTRA_REVISION, REPLAY_SEED, cases, newline_boundaries,
                               prior_replay, terminal_cycle, with_newline_tags)

INSERTIONS = {name: newline_boundaries("", mark) for name, mark in (("blank", ""), ("left", "["), ("round", "()"))}
SAMPLING = {"temperature": .6, "top_p": .95, "top_k": 20, "presence_penalty": 1.5}
# Each arm specifies receiver, donor, and the groups whose masses follow the donor.
BASELINES = {name: (name, name, ()) for name in INSERTIONS}
JOINT = {"blank_from_left": ("blank", "left", ("wait", "think")),
         "left_from_blank": ("left", "blank", ("wait", "think"))}
SPLIT = {f"{name}_{group}": (receiver, donor, (group,))
         for name, (receiver, donor, _) in JOINT.items() for group in ("wait", "think")}


def load(path):
    return json.loads(path.read_text())


def identity(args):
    manifest = load(args.data.parent / "manifest.json")
    if sha(args.data) != manifest["artifacts"][args.data.name]["sha256"]:
        raise ValueError("data digest mismatch")
    if (manifest.get("model_revision") or manifest.get("revision")) != ASTRA_REVISION:
        raise ValueError("model revision mismatch")
    for name, expected in manifest["artifacts"].items():
        if Path(name).parts[0] == "model" and sha(args.model / Path(name).name) != expected["sha256"]:
            raise ValueError("model digest mismatch")


def select_cases(args):
    if args.phase == "dev":
        source, parent_sha, _ = prior_replay(args.parent)
        _, selected, verified = cases(args.baseline, args.data, astra=True)
        parent = {x["id"]: x for x in source["verified_cases"]}
        if verified != 34 or len(selected) != 33 or set(parent) != {x["row"]["id"] for x in selected}:
            raise ValueError("mismatched calibration cohort")
        rows = []
        for item in selected:
            previous = parent[item["row"]["id"]]
            prior = item["completion_prefix"]
            rows.append({**item, "budget": previous["prefix_len"]["placebo"] - previous["prefix_len"]["plain"],
                         "origin": item["phase"], "reference": previous, "cycle": prior[-item["period"]:]})
        return rows, range(args.part * 2, args.part * 2 + 2), parent_sha

    path = args.root / "holdout.json"
    all_cases = load(path)["cases"]
    if len({x["row"]["id"] for x in all_cases}) != 64 or len(all_cases) != 64:
        raise ValueError("expected 64 distinct held-out cases")
    if args.phase == "long":
        all_cases = all_cases[:24]
    rows = [{**x, "budget": 41, "reference": None} for x in all_cases[args.part::args.partitions]]
    return rows, range(8), sha(path)


def token_groups(tokenizer):
    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    forms = ("Wait", " Wait", "wait", " wait", "WAIT", " WAIT", "\nWait", "\nwait")
    wait = sorted({ids[0] for form in forms if len(ids := encode(form)) == 1})
    think = encode("</think>")
    if not wait or len(think) != 1 or think[0] in wait:
        raise ValueError("Wait or thinking marker tokenization mismatch")
    return {"wait": wait, "think": think}


def transfer_masses(receiver, donor, groups, torch):
    """Use donor masses and receiver conditionals for disjoint groups and their complement."""
    receiver = torch.log_softmax(receiver, dim=-1)
    donor = torch.log_softmax(donor, dim=-1)
    kept = torch.ones(receiver.shape[-1], dtype=torch.bool, device=receiver.device)
    target = receiver.clone()
    for ids in groups:
        kept[ids] = False
        real = receiver[:, ids]
        target[:, ids] = real - real.logsumexp(dim=-1, keepdim=True) + donor[:, ids].logsumexp(dim=-1, keepdim=True)
    donor_rest = donor.masked_fill(~kept, -torch.inf).logsumexp(dim=-1, keepdim=True)
    receiver_rest = receiver.masked_fill(~kept, -torch.inf).logsumexp(dim=-1, keepdim=True)
    return torch.where(kept, receiver - receiver_rest + donor_rest, target)


def presence_ids(item):
    ids = item["presence_seed_ids"] if "presence_seed_ids" in item else item["completion_prefix"]
    return sorted(set(ids))


def presence_scores(logits, seen, sampling=SAMPLING):
    return (logits.float() - sampling["presence_penalty"] * seen) / sampling["temperature"]


def processor_factory(torch, transformers, groups, sampling=SAMPLING):
    top_k = transformers.TopKLogitsWarper(top_k=sampling["top_k"])
    top_p = transformers.TopPLogitsWarper(top_p=sampling["top_p"])

    class CoupledPolicy(transformers.LogitsProcessor):
        def __init__(self, group, width, groups_to_swap, seed, *, row_seeds=None, force_first=None):
            self.width = width
            self.n = len(group)
            self.tensors = [torch.tensor(groups[name], device="cuda") for name in groups_to_swap]
            self.previous = [torch.tensor(presence_ids(x), dtype=torch.long, device="cuda") for x in group]
            self.pending = torch.tensor([x["pending"] for x in group], device="cuda")
            self.generator = torch.Generator(device="cuda").manual_seed(seed)
            self.row_generators = (None if row_seeds is None else
                                   [torch.Generator(device="cuda").manual_seed(s) for s in row_seeds])
            if self.row_generators is not None and len(self.row_generators) != self.n:
                raise ValueError("each receiver needs exactly one random stream")
            self.force_first = force_first
            self.first_step = None
            self.first_transform = None

        def seen_mask(self, history, scores):
            seen = torch.zeros_like(scores[:self.n], dtype=torch.bool)
            seen.scatter_(1, history, True)
            for index, prior in enumerate(self.previous):
                seen[index, prior] = True
            return seen

        def __call__(self, ids, scores):
            history = ids[:self.n, self.width:]
            seen = self.seen_mask(history, scores)
            if self.tensors:
                base, donor = scores.float().split(self.n, dim=0)
                base = presence_scores(base, seen, sampling)
                donor = presence_scores(donor, seen, sampling)
                active = self.pending & ~(history == groups["think"][0]).any(dim=-1)
                mixed = transfer_masses(base, donor, self.tensors, torch)
                result = torch.where(active[:, None], mixed, base)
            else:
                base = result = presence_scores(scores, seen, sampling)
            sampled = top_p(ids[:self.n], top_k(ids[:self.n], result))
            probabilities = sampled.softmax(dim=-1)
            if self.first_step is None:
                if self.first_transform is not None:
                    probabilities = self.first_transform(probabilities, base)
                raw = base.softmax(dim=-1)
                mixed = result.softmax(dim=-1)
                self.first_step = {
                    name: torch.stack([tensor[:, token_ids].sum(dim=-1) for tensor in (raw, mixed, probabilities)],
                                      dim=-1).cpu().tolist()
                    for name, token_ids in groups.items()
                }
            if self.force_first is not None and history.shape[1] == 0:
                token = ids.new_full((self.n, 1), self.force_first)
            elif self.row_generators is None:
                token = torch.multinomial(probabilities, 1, generator=self.generator)
            else:
                token = torch.stack([torch.multinomial(p, 1, generator=g)
                                     for p, g in zip(probabilities, self.row_generators, strict=True)])
            if self.tensors:
                token = token.repeat(2, 1)
            return torch.full_like(scores, -torch.inf).scatter_(1, token, 0.)

    return CoupledPolicy


def run(args):
    import torch
    import transformers

    if torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one assigned GPU is required")
    identity(args)
    selected, draws, cohort = select_cases(args)
    arms = SPLIT if args.phase == "split" else {**BASELINES, **JOINT}
    cap = 2048 if args.phase == "long" else 512
    if args.smoke:
        selected, draws, cap = selected[:2], range(1), 12

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    groups = token_groups(tokenizer)
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True).to("cuda:0").eval()
    defaults = ({"use_model_defaults": False}
                if "use_model_defaults" in inspect.signature(model.generate).parameters else {})
    decoder = processor_factory(torch, transformers, groups)
    end = model.generation_config.eos_token_id
    eos = set(end if isinstance(end, (list, tuple)) else [end])
    if None in eos:
        raise ValueError("EOS configuration is absent")
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else min(eos)
    config = transformers.GenerationConfig(do_sample=False, max_new_tokens=cap, pad_token_id=pad,
                                           eos_token_id=sorted(eos), use_cache=True, top_k=50)
    items = []
    for case in selected:
        row, prior = case["row"], case["completion_prefix"]
        prefix = tokenizer.encode(render(tokenizer, row, "think"), add_special_tokens=False) + prior
        original = tokenizer.decode(prior, skip_special_tokens=False)
        pending = "</think>" not in original
        before = assess(row, original, prior, False, 8192, True)["correct"]
        reference = case["reference"]
        if reference is not None:
            if (len(prefix) != reference["prefix_len"]["plain"] or before != reference["prior_final"]
                    or case["period"] != reference["period"]):
                raise ValueError("calibration prefix differs from the original replay")
        elif before or not pending:
            # Keep the frozen cohort and label ineligible states for stratified analysis.
            pass
        attached = with_newline_tags(tokenizer, prefix, case["budget"], INSERTIONS)
        if any(len(ids) != len(prefix) + case["budget"] for ids in attached.values()):
            raise ValueError("unequal insertion size")
        items.append({**case, "inputs": attached, "pending": pending, "prior_final": before,
                      "wait_loop": bool(set(case["cycle"]) & set(groups["wait"])),
                      "prefix_sha256": hashlib.sha256(json.dumps(prefix).encode()).hexdigest()})

    header = {"kind": "manifest", "phase": args.phase, "part": args.part, "partitions": args.partitions,
              "cohort_sha256": cohort, "source_sha256": sha(Path(__file__)), "model": str(args.model),
              "cap": cap, "sampling": SAMPLING, "groups": groups, "draws": list(draws), "arms": list(arms),
              "cases": [{key: item[key] for key in ("period", "pending", "prior_final", "wait_loop", "budget",
                                                   "prefix_sha256", "origin")} | {"id": item["row"]["id"]}
                        for item in items]}
    start = time.monotonic()
    with args.output.open("x") as stream, torch.inference_mode():
        print(json.dumps(header), file=stream, flush=True)
        for draw in draws:
            for begin in range(0, len(items), 8):
                batch = items[begin:begin + 8]
                seed = REPLAY_SEED + 10000 * draw + begin + (0 if args.phase == "dev" else 1000000 * (args.part + 1))
                for arm, (receiver, donor, swap) in arms.items():
                    if time.time() >= args.deadline - 30:
                        raise TimeoutError("replay deadline")
                    parts = [item["inputs"][receiver] for item in batch]
                    if swap:
                        parts += [item["inputs"][donor] for item in batch]
                    width = max(map(len, parts))
                    ids = torch.tensor([[pad] * (width - len(part)) + part for part in parts], device="cuda")
                    mask = torch.tensor([[0] * (width - len(part)) + [1] * len(part) for part in parts], device="cuda")
                    policy = decoder(batch, width, swap, seed)
                    generated = model.generate(input_ids=ids, attention_mask=mask, generation_config=config,
                                               logits_processor=[policy], **defaults)
                    output = generated[:, width:].tolist()
                    if swap and output[:len(batch)] != output[len(batch):]:
                        raise RuntimeError("donor and receiver continuation diverged")
                    for index, (case, tokens) in enumerate(zip(batch, output[:len(batch)], strict=True)):
                        stop = next((i for i, token in enumerate(tokens) if token in eos), None)
                        done = stop is not None
                        tokens = tokens[:stop] if done else tokens
                        text = tokenizer.decode(tokens, skip_special_tokens=False)
                        score = assess(case["row"], text, tokens, done, cap, case["pending"])
                        loop = bool(terminal_cycle(tokens))
                        # Record early exact recurrence independently of the terminal detector.
                        cycle = case["cycle"]
                        motif = tuple(cycle * 4)
                        tandem = cycle * 2
                        rotations = {tuple((tandem[j:j + len(cycle)]) * 4) for j in range(len(cycle))}
                        recurrence = any(tuple(tokens[i:i + len(motif)]) in rotations
                                         for i in range(min(128, len(tokens) - len(motif) + 1)))
                        result = {"kind": "rollout", "id": case["row"]["id"], "draw": draw, "arm": arm,
                                  "terminal_cycle": loop, "return_original_cycle": recurrence,
                                  "correct": bool(score["correct"]), "eos": done, "cap": len(tokens) == cap,
                                  "eos_without_correct": done and not score["correct"],
                                  "new_correct_eos": score["correct"] and done and not case["prior_final"] and not loop,
                                  "closed_think": groups["think"][0] in tokens, "tokens": tokens,
                                  "first_step_mass": {name: values[index] for name, values in policy.first_step.items()}}
                        print(json.dumps(result), file=stream)
                print(f"{args.phase} draw={draw} prompt_count={begin + len(batch)}/{len(items)} "
                      f"seconds={int(time.monotonic() - start)}", flush=True)
                stream.flush()
        print(json.dumps({"kind": "complete", "seconds": time.monotonic() - start}), file=stream)


def summarize(args):
    phase = args.phase
    lanes = ("calibration",) if phase == "dev" else ("discovery", "calibration")
    metrics = ("terminal_cycle", "return_original_cycle", "correct", "eos", "cap",
               "eos_without_correct", "new_correct_eos", "closed_think")
    rows = {}
    metadata = {}
    contract = None
    shards = set()
    for lane in lanes:
        for filename, expected in load(args.root / lane / f"{phase}-done.json")["outputs"]:
            path = Path(filename)
            if sha(path) != expected:
                raise ValueError("replay worker digest mismatch")
            with path.open() as stream:
                head = json.loads(next(stream))
                fingerprint = {key: head[key] for key in ("phase", "sampling", "model", "cap", "arms",
                                                           "cohort_sha256", "source_sha256", "partitions", "groups")}
                if (contract is not None and fingerprint != contract) or head["part"] in shards:
                    raise ValueError("replay shards disagree")
                contract = fingerprint
                shards.add(head["part"])
                checks = {(r["id"], draw, arm) for r in head["cases"] for draw in head["draws"] for arm in head["arms"]}
                for case in head["cases"]:
                    old = metadata.setdefault(case["id"], case)
                    if old != case:
                        raise ValueError("case metadata changed")
                complete = False
                for line in stream:
                    item = json.loads(line)
                    if item["kind"] == "complete":
                        complete = True
                        continue
                    key = item["id"], item["draw"], item["arm"]
                    if key in rows or key not in checks:
                        raise ValueError("duplicate or unexpected replay rollout")
                    checks.remove(key)
                    rows[key] = item
                if not complete or checks:
                    raise ValueError("missing replay rollout or completion footer")
    expected = 33 if phase == "dev" else (24 if phase == "long" else 64)
    if len(shards) != contract["partitions"] or len(metadata) != expected or len(rows) != expected * 8 * len(contract["arms"]):
        raise ValueError("incomplete replay matrix")

    partitions = {"all": list(metadata), "wait": [k for k, m in metadata.items() if m["wait_loop"]],
                  "other": [k for k, m in metadata.items() if not m["wait_loop"]],
                  "active_no_prior_answer": [k for k, m in metadata.items() if m["pending"] and not m["prior_final"]]}
    rng = random.Random(11738)
    outputs = {}
    for name, keys in partitions.items():
        if not keys:
            continue
        by_arm = {}
        counts = {}
        for arm in contract["arms"]:
            by_prompt = {k: [rows[k, draw, arm] for draw in range(8)] for k in keys}
            counts[arm] = {metric: sum(int(r[metric]) for runs in by_prompt.values() for r in runs) for metric in metrics}
            counts[arm]["prompt_successes"] = sum(any(r["new_correct_eos"] for r in runs) for runs in by_prompt.values())
            counts[arm]["first_step_mass"] = {
                group: [statistics.fmean(r["first_step_mass"][group][i] for runs in by_prompt.values() for r in runs)
                        for i in range(3)] for group in contract["groups"]}
            by_arm[arm] = {metric: {k: statistics.fmean(bool(x[metric]) for x in by_prompt[k]) for k in keys}
                           for metric in ("terminal_cycle", "new_correct_eos")}
        pairs = [("left", "blank"), ("left", "round"), ("blank_from_left", "blank"),
                 ("left", "left_from_blank"), ("blank_from_left", "left"), ("left_from_blank", "blank")]
        if phase == "split":
            pairs = [("blank_from_left_wait", "blank_from_left_think"),
                     ("left_from_blank_wait", "left_from_blank_think")]
        contrasts = {}
        for a, b in pairs:
            contrasts[f"{a}_minus_{b}"] = {}
            for metric in ("terminal_cycle", "new_correct_eos"):
                diff = [100 * (by_arm[a][metric][k] - by_arm[b][metric][k]) for k in sorted(keys)]
                boot = sorted(statistics.fmean(rng.choices(diff, k=len(diff))) for _ in range(3000))
                contrasts[f"{a}_minus_{b}"][metric] = [round(statistics.fmean(diff), 2),
                                                        round(boot[75], 2), round(boot[2924], 2)]
        outputs[name] = {"prompts": len(keys), "draws": 8, "arms": counts, "paired_interval_pp": contrasts}
    result = {"protocol": contract, "rollouts": len(rows), "strata": outputs}
    path = args.root / f"{phase}-summary.json"
    temp = path.with_suffix(".tmp")
    with temp.open("x") as stream:
        json.dump(result, stream, indent=2)
    temp.replace(path)


def self_test():
    import torch

    p = torch.tensor([[1., 2., 3., 4., 5.]])
    q = torch.tensor([[5., 4., 3., 2., 1.]])
    target = transfer_masses(p, q, [[0, 1], [4]], torch).exp()
    assert torch.allclose(target.sum(-1), torch.tensor([1.]))
    assert torch.allclose(target[:, [0, 1]].sum(-1), q.softmax(-1)[:, [0, 1]].sum(-1))
    assert torch.allclose(target[:, 4], q.softmax(-1)[:, 4])
    assert torch.allclose(target[:, 0] / target[:, 1], (p[:, 0] - p[:, 1]).exp())
    assert torch.allclose(transfer_masses(p, p, [[0, 1], [4]], torch).exp(), p.softmax(-1))
    print("probability mass and identity checks passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "summarize", "self-test"))
    parser.add_argument("--phase", choices=("dev", "main", "split", "long"))
    parser.add_argument("--part", type=int)
    parser.add_argument("--partitions", type=int)
    parser.add_argument("--deadline", type=int)
    parser.add_argument("--smoke", action="store_true")
    for name in ("root", "output", "model", "data", "baseline", "parent"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    {"run": run, "summarize": summarize, "self-test": lambda _: self_test()}[args.action](args)


if __name__ == "__main__":
    main()
