"""Replay canonical Astra prefixes and test the block-4 and forced-stop contrast."""
import argparse
from collections import Counter, defaultdict
import hashlib
import inspect
import json
from pathlib import Path
import random
import socket
import statistics
import time

from repetition_baseline import assess, math_answer, render, sha
from repetition_canonical_queue import THINK_END, paired_interval
from repetition_exit import padded, patch_context
from repetition_exit_localize import decoder_layers
from repetition_mediation import INSERTIONS, SAMPLING, identity, processor_factory, token_groups
from repetition_mediation_queue import load, save
from repetition_replay import terminal_cycle, with_newline_tags

CORE = ("native", "blank", "left", "think_transfer")
MECHANISM = ("blank", "patch_forward", "forced_blank", "forced_patch", "patch_random", "left", "patch_reverse")
PHASES = ("main", "long", "mechanism", "copy-dev", "copy-test")
METRICS = ("success", "numeric_correct", "exact_format", "answer_found", "eos", "terminal_cycle", "cap",
           "wrong_parsed_eos", "missing_answer_eos", "think_at_zero", "copy_exact")


def copy_panel(phase, repeats=None):
    dev = phase == "copy-dev"
    words = [a + b + c for a in "mnpbrst" for b in "aeiou" for c in "lnmrkst"]
    random.Random(92051).shuffle(words)
    words = words[:18] if dev else words[18:34]
    result = []
    for i, motif in enumerate(words):
        amount = i % 3 + 1 if dev else repeats
        row = {"id": f"{phase}:{i}", "task": "copy",
               "prompt": f"Write the word {motif} exactly {amount} times, separated by a space. "
                         "Your final answer should contain only the requested word or words.",
               "gold": " ".join([motif] * amount)}
        result.append({"row": row, "completion_prefix": [], "stratum": "copy", "budget": 41, "repeats": amount})
    return result


def phase_input(root, phase):
    if phase == "copy-dev":
        return copy_panel(phase), ("native",), 192, 2, None
    if phase == "copy-test":
        plan = root / "copy-dev-summary.json"
        n = load(plan)["selected_repeats"]
        if n is None:
            raise ValueError("native copy floor in development")
        return copy_panel(phase, n), CORE, 192, 4, sha(plan)
    manifest = root / "cohort.json"
    all_cases = load(manifest)["cases"]
    if phase in ("long", "mechanism"):
        max_loop, max_clean = (16, 8) if phase == "long" else (24, 12)
        all_cases = ([x for x in all_cases if x["stratum"] == "loop"][:max_loop] +
                     [x for x in all_cases if x["stratum"] == "clean"][:max_clean])
    return all_cases, MECHANISM if phase == "mechanism" else CORE, 2048 if phase == "long" else 512, 4, sha(manifest)


def seed_for(prompt_id, draw):
    # The stream is independent of the arm, partition, padding and other batch rows.
    raw = hashlib.sha256(f"canonical-v1:{prompt_id}:{draw}".encode()).digest()
    return int.from_bytes(raw[:8], "little") & ((1 << 63) - 1)


def receiver_for(arm):
    if arm in ("native", "blank", "left"):
        return arm
    return "left" if arm == "patch_reverse" else "blank"


def score(row, answer, tokens, eos, cap, first_think):
    if row["task"] == "math":
        outcome = math_answer(row, answer)
        found, numeric, exact = (outcome[key] for key in ("answer_found", "numeric_correct", "exact_format"))
        copy_exact = False
    else:
        outcome = assess(row, answer, [], eos, cap, True)
        found, numeric, exact = outcome["answer_found"], False, False
        copy_exact = outcome["correct"]
    terminal = terminal_cycle(tokens) is not None
    return {"success": bool(eos and not terminal and (numeric or copy_exact)), "numeric_correct": numeric,
            "exact_format": exact, "answer_found": found, "parsed_answer": outcome.get("parsed_answer"),
            "eos": eos, "terminal_cycle": terminal, "cap": not eos and len(tokens) == cap,
            "wrong_parsed_eos": row["task"] == "math" and eos and found and not numeric,
            "missing_answer_eos": eos and not found, "think_at_zero": first_think == 0,
            "think_step": first_think, "after_think_tokens": None if first_think is None else len(tokens) - first_think - 1,
            "copy_exact": copy_exact}


def run(args):
    import torch
    import transformers
    import tokenizers

    if torch.cuda.device_count() != 1:
        raise ValueError("exactly one assigned GPU required")
    identity(args)
    assets = load(args.data.parent / "manifest.json")
    checkpoint = {k: v["sha256"] for k, v in assets["artifacts"].items()
                  if k.startswith("model/") and k.endswith(".safetensors")}
    cases, arms, cap, draws, selection_sha = phase_input(args.root, args.phase)
    items = []
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    groups = token_groups(tokenizer)
    if groups["think"] != [THINK_END]:
        raise ValueError("unexpected reasoning terminator")
    model = transformers.AutoModelForImageTextToText.from_pretrained(args.model, local_files_only=True,
                                                                    dtype=torch.bfloat16).to("cuda:0").eval()
    eos_config = model.generation_config.eos_token_id
    eos_ids = set(eos_config if isinstance(eos_config, (list, tuple)) else [eos_config])
    if not eos_ids or None in eos_ids:
        raise ValueError("native EOS config missing")
    padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else min(eos_ids)
    defaults = {"use_model_defaults": False} if "use_model_defaults" in inspect.signature(model.generate).parameters else {}
    layer_map = {3: decoder_layers(model, torch)[3]} if args.phase == "mechanism" else {}
    policy_type = processor_factory(torch, transformers, groups)
    for case in cases[args.part::args.partitions]:
        row = case["row"]
        prompt = render(tokenizer, row, "think")
        prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
        if case.get("render_sha256", prompt_sha) != prompt_sha:
            raise ValueError(f"rendered prompt differs from baseline: {row['id']}")
        prior = case["completion_prefix"]
        if THINK_END in prior:
            raise ValueError(f"canonical prefix has already exited thinking: {row['id']}")
        prefix = tokenizer.encode(prompt, add_special_tokens=False) + prior
        arm_inputs = with_newline_tags(tokenizer, prefix, case["budget"], INSERTIONS)
        if any(len(x) != len(prefix) + case["budget"] for x in arm_inputs.values()):
            raise ValueError("token insertion budgets disagree")
        items.append({**case, "pending": True, "inputs": {"native": prefix, **arm_inputs},
                      "render_sha256": prompt_sha,
                      "prefix_sha256": hashlib.sha256(json.dumps(prefix).encode()).hexdigest()})

    header = {"kind": "manifest", "phase": args.phase, "cap": cap, "forced_post_cap": cap, "draws": draws,
              "arms": list(arms), "total": len(cases), "partitions": args.partitions, "part": args.part,
              "input_sha256": selection_sha, "worker_sha256": sha(Path(__file__)),
              "tokenizer_sha256": sha(args.model / "tokenizer.json"), "checkpoint": checkpoint,
              "checkpoint_revision": assets["model_revision"], "sampling": SAMPLING,
              "eos_ids": sorted(eos_ids),
              "helper_sha256": {m: sha(Path(sysmod.__file__)) for m, sysmod in modules().items()},
              "runtime": {"torch": str(torch.__version__), "transformers": transformers.__version__,
                          "tokenizers": tokenizers.__version__, "host": socket.gethostname()},
              "cases": [{"id": x["row"]["id"], "stratum": x["stratum"], "repeats": x.get("repeats"),
                         "render_sha256": x["render_sha256"], "prefix_sha256": x["prefix_sha256"]} for x in items]}
    start = time.monotonic()
    with args.output.open("x") as stream, torch.inference_mode():
        def emit(r):
            print(json.dumps(r, allow_nan=False), file=stream)
        emit(header)
        for draw in range(draws):
            for begin in range(0, len(items), 8):
                batch = items[begin:begin + 8]
                seeds = [seed_for(x["row"]["id"], draw) for x in batch]
                for arm in arms:
                    if time.time() >= args.deadline - 60:
                        raise TimeoutError("replay deadline")
                    receiver = receiver_for(arm)
                    parts = [x["inputs"][receiver] for x in batch]
                    coupled = arm == "think_transfer"
                    if coupled:
                        parts += [x["inputs"]["left"] for x in batch]
                    packed = padded(parts, padding, torch)
                    width = packed["input_ids"].shape[1]
                    forced = arm in ("forced_blank", "forced_patch")
                    cap_limit = cap + int(forced)
                    cfg = transformers.GenerationConfig(do_sample=False, use_cache=True, max_new_tokens=cap_limit,
                                                        eos_token_id=sorted(eos_ids), pad_token_id=padding)
                    method = policy_type(batch, width, ("think",) if coupled else (), seeds[0], row_seeds=seeds,
                                         force_first=THINK_END if forced else None)
                    patch_arm = "patch_forward" if arm == "forced_patch" else arm
                    with patch_context(patch_arm, model, layer_map, batch, padding, seeds[0], torch):
                        out = model.generate(**packed, generation_config=cfg, logits_processor=[method], **defaults)
                    token_rows = out[:, width:].tolist()
                    if coupled and token_rows[:len(batch)] != token_rows[len(batch):]:
                        raise ValueError("coupled contexts saw different sampled history")
                    for i, (case, ids) in enumerate(zip(batch, token_rows[:len(batch)], strict=True)):
                        stop = next((j for j, tok in enumerate(ids) if tok in eos_ids), None)
                        tokens = ids[:stop] if stop is not None else ids
                        think = tokens.index(THINK_END) if THINK_END in tokens else None
                        if forced and think != 0:
                            raise ValueError("forced closing tag was not the first generated token")
                        text = tokenizer.decode(tokens, skip_special_tokens=False)
                        emit({"kind": "rollout", "id": case["row"]["id"], "draw": draw, "arm": arm,
                              "seed": seeds[i], "forced_first": forced, "tokens": tokens,
                              "eos_token": ids[stop] if stop is not None else None,
                              "first_think_mass": method.first_step["think"][i],
                              **score(case["row"], text, tokens, stop is not None, cap_limit, think)})
                stream.flush()
                print(f"{args.phase} draw={draw} prompts={begin + len(batch)}/{len(items)}", flush=True)
        emit({"kind": "complete", "seconds": time.monotonic() - start})


def modules():
    import repetition_baseline
    import repetition_exit
    import repetition_exit_localize
    import repetition_mediation
    return {m.__name__: m for m in (repetition_baseline, repetition_exit, repetition_exit_localize, repetition_mediation)}


def matrix(root, phase):
    receipt = load(root / "queue" / f"{phase}-done.json")
    merged, cases, parts, contract, runtimes = {}, {}, set(), None, []
    for filename, checksum in receipt["outputs"]:
        path = Path(filename)
        if sha(path) != checksum:
            raise ValueError("worker checksum mismatch")
        with path.open() as stream:
            head = json.loads(next(stream))
            fingerprint = {k: v for k, v in head.items() if k not in ("runtime", "part", "cases")}
            if head["phase"] != phase or head["part"] in parts or (contract and fingerprint != contract):
                raise ValueError("worker contract or partition mismatch")
            contract = fingerprint
            parts.add(head["part"])
            runtimes.append(head["runtime"])
            local_ids = {x["id"]: x for x in head["cases"]}
            if cases.keys() & local_ids.keys():
                raise ValueError("overlapping prompt partitions")
            cases.update(local_ids)
            done = False
            for line in stream:
                r = json.loads(line)
                if r["kind"] == "complete":
                    done = True
                    continue
                key = r["id"], r["draw"], r["arm"]
                if r["id"] not in local_ids or key in merged or r["seed"] != seed_for(r["id"], r["draw"]):
                    raise ValueError("unexpected prompt, seed or duplicate rollout")
                merged[key] = r
            if not done:
                raise ValueError("worker has no completion footer")
    expected = {(p, draw, arm) for p in cases for draw in range(contract["draws"]) for arm in contract["arms"]}
    if merged.keys() != expected or len(cases) != contract["total"] or parts != set(range(contract["partitions"])):
        raise ValueError("incomplete rollout matrix")
    return contract, runtimes, cases, merged


def summarize(args):
    phase = args.phase
    protocol, runtime, cases, rows = matrix(args.root, phase)
    draws = protocol["draws"]
    strata = defaultdict(list)
    for p, record in cases.items():
        strata[record["stratum"]].append(p)
    contrasts = {
        "main": (("left", "native"), ("blank", "native"), ("think_transfer", "blank")),
        "long": (("left", "native"), ("blank", "native"), ("think_transfer", "blank")),
        "mechanism": (("patch_forward", "blank"), ("forced_patch", "forced_blank"),
                      ("patch_forward", "patch_random"), ("left", "patch_reverse")),
        "copy-dev": (),
        "copy-test": (("left", "native"), ("blank", "native"), ("think_transfer", "blank")),
    }[phase]
    summary = {}
    for label, ids in strata.items():
        totals = {}
        for arm in protocol["arms"]:
            observed = [rows[p, d, arm] for p in ids for d in range(draws)]
            totals[arm] = {m: sum(bool(r[m]) for r in observed) for m in METRICS}
            closed = [r for r in observed if r["think_step"] is not None]
            totals[arm]["closed"] = len(closed)
            totals[arm]["median_think_step"] = median([r["think_step"] for r in closed])
            totals[arm]["median_after_think_tokens"] = median([r["after_think_tokens"] for r in closed])
        intervals = {}
        for a, b in contrasts:
            intervals[f"{a}_minus_{b}"] = {metric: paired_interval(
                {p: [int(rows[p, d, a][metric]) - int(rows[p, d, b][metric]) for d in range(draws)] for p in ids})
                for metric in ("success", "copy_exact", "terminal_cycle", "wrong_parsed_eos", "missing_answer_eos")}
        if phase == "mechanism":
            intervals["natural_minus_forced_patch_effect"] = {"success": paired_interval({
                p: [int(rows[p, d, "patch_forward"]["success"]) - int(rows[p, d, "blank"]["success"])
                    - int(rows[p, d, "forced_patch"]["success"]) + int(rows[p, d, "forced_blank"]["success"])
                    for d in range(draws)] for p in ids})}
        summary[label] = {"prompts": len(ids), "rollouts_per_arm": len(ids) * draws,
                          "arms": totals, "paired": intervals}
    result = {"rows": len(rows), "protocol": protocol, "runtime": runtime, "strata": summary}
    if phase == "copy-dev":
        qualified = []
        by_repeats = {}
        for amount in (1, 2, 3):
            ids = [p for p, row in cases.items() if row["repeats"] == amount]
            flags = {p: [rows[p, d, "native"]["copy_exact"] and rows[p, d, "native"]["eos"]
                         for d in range(draws)] for p in ids}
            good = sum(sum(v) for v in flags.values())
            by_repeats[amount] = {"native_exact_eos": good, "rollouts": len(ids) * draws}
            if good >= len(ids) * draws * .25 and sum(any(v) for v in flags.values()) >= 2:
                qualified.append(amount)
        result.update(copy_dev=by_repeats, selected_repeats=max(qualified, default=None))
    save(args.root / f"{phase}-summary.json", result)


def median(values):
    return statistics.median(values) if values else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("run", "summarize"))
    p.add_argument("--phase", choices=PHASES, required=True)
    p.add_argument("--part", type=int)
    p.add_argument("--partitions", type=int)
    p.add_argument("--deadline", type=int)
    for k in ("root", "model", "data", "output"):
        p.add_argument("--" + k, type=Path)
    args = p.parse_args()
    (run if args.action == "run" else summarize)(args)


if __name__ == "__main__":
    main()
