"""Replay a finite block on the original motif while preserving stop-token mass."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

from repetition_baseline import sha
from repetition_budget_run import packed_hash, tokens
from repetition_canonical import seed_for
from repetition_followup_aggregate import contrast, tally
from repetition_followup_baseline import runtime_for
from repetition_mediation_queue import save
from repetition_motif_intervention import BREAK, CLOSE, NATIVE, break_transform, close_transform, outcome, source_cases
from repetition_motif_path_audit import first_return
from repetition_sampler_reach import pack_at
from repetition_state_value import inputs_for, post_seed


GUARD = "motif_guard"
ARMS = (BREAK, GUARD, CLOSE)
PERIODS = 2


def completions(motif):
    """Map each rotation's length-(m-1) prefix to tokens completing the motif."""
    result = defaultdict(set)
    for shift in range(len(motif)):
        rotation = tuple(motif[shift:] + motif[:shift])
        result[rotation[:-1]].add(rotation[-1])
    return {key: sorted(values) for key, values in result.items()}


def blocked_next(history, motif_size, moves, stops):
    step = len(history)
    if step == 0 or step >= PERIODS * motif_size or any(token in stops for token in history):
        return ()
    tail = tuple(history[-(motif_size - 1):]) if motif_size > 1 else ()
    return moves.get(tail, ())


def reallocate(torch, probabilities, adjusted, blocked, protected):
    """Redistribute blocked mass only over allowed ordinary tokens."""
    available = torch.ones_like(probabilities, dtype=torch.bool)
    available[blocked] = False
    available[protected] = False
    moved = probabilities[blocked].sum()
    if not bool(moved > 0):
        return probabilities, 0., False
    ordinary = probabilities.masked_fill(~available, 0.)
    support = ordinary.sum()
    fallback = not bool(support > 0)
    if fallback:
        weights = adjusted.masked_fill(~available, -torch.inf).softmax(dim=-1)
    else:
        weights = ordinary / support
    result = probabilities.clone()
    result[blocked] = 0.
    result += moved * weights
    if (not bool(result.isfinite().all()) or not torch.equal(result[protected], probabilities[protected])
            or not bool(torch.isclose(result.sum(), probabilities.sum(), atol=2e-6, rtol=0.))):
        raise ValueError("guard changed stop probability or failed to normalize")
    return result, float(moved.item()), fallback


def policy_for(runtime, items, protected, stops, audits):
    base = runtime.policy
    moves = [completions(item["motif"]) for item in items]
    sizes = [len(item["motif"]) for item in items]

    class GuardPolicy(base):
        def __init__(self, *args):
            super().__init__(*args)
            self.step_transform = self.guard

        def guard(self, history, filtered, adjusted):
            step = history.shape[1]
            if step == 0 or step >= PERIODS * max(sizes):
                return filtered
            protected_ids = sorted(token for token in protected if 0 <= token < filtered.shape[1])
            for i, prefix in enumerate(history.tolist()):
                blocked = blocked_next(prefix, sizes[i], moves[i], stops)
                if not blocked:
                    continue
                result, mass, fallback = reallocate(runtime.torch, filtered[i], adjusted[i], blocked, protected_ids)
                if mass:
                    filtered[i] = result
                    audits[i].append({"step": step + 1, "blocked": blocked, "moved_mass": mass,
                                      "opened_prefilter_support": fallback})
            return filtered

    return GuardPolicy


def cases_and_tokens(tokenizer, header, cases):
    [turn] = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    if turn != header["turn"]:
        raise ValueError("chat end differs from the archived baseline")
    stops = {*header["eos_ids"], header["marker"], turn, header["pad_id"]}
    protected = stops | set(tokenizer.all_special_ids)
    eligible = [case for case in cases if protected.isdisjoint(case["motif"])]
    return eligible, protected, stops


def run(args):
    header, cases = source_cases(args.raw, args.fingerprint, args.panel, args.cap, args.limit,
                                 args.source_draw, args.canary)
    runtime = runtime_for(args)
    if (runtime.groups["think"] != [header["marker"]] or any(runtime.manifest[key] != header[key]
            for key in ("tokenizer_sha256", "checkpoint", "runtime", "sampling", "eos_ids", "pad_id"))):
        raise ValueError("guard runtime differs from the archived baseline")
    eligible, protected, stops = cases_and_tokens(runtime.tokenizer, header, cases)
    if args.canary and not eligible:
        raise ValueError("canary requires a confirmed motif made of ordinary tokens")
    items = inputs_for(runtime.tokenizer, eligible)
    partial = args.output.with_suffix(".partial.jsonl")
    if args.output.exists():
        raise FileExistsError(args.output)
    with partial.open("x") as stream, runtime.torch.inference_mode():
        def emit(row):
            print(json.dumps(row, allow_nan=False), file=stream, flush=True)

        emit({"kind": "manifest", "worker_sha256_guard": sha(Path(__file__)), "source": str(args.raw),
              "source_sha256": args.fingerprint, "panel_sha256": sha(args.panel),
              "source_draw": args.source_draw, "draws": args.draws, "draw_start": args.draw_start,
              "cap": args.cap, "limit": args.limit, "canary": args.canary, "periods": PERIODS,
              "dataset": header["dataset"], "split": header["split"], "marker": header["marker"],
              "turn": header["turn"], "eligible": len(items), "excluded_special_motif": len(cases)-len(items),
              "arms": ARMS, **runtime.manifest})
        started = time.monotonic()
        for start in range(0, len(items), args.batch):
            group = items[start:start + args.batch]
            ids = [item["native_ids"] for item in group]
            packed = pack_at(runtime, ids, max(map(len, ids)))
            for draw in range(args.draw_start, args.draw_start + args.draws):
                audits = [[] for _ in group]
                guard_policy = policy_for(runtime, group, protected, stops, audits)
                initial_events = {BREAK: [], GUARD: []}
                for arm in ARMS:
                    if time.time() >= args.deadline - 120:
                        raise TimeoutError("motif guard nearing the allocation deadline")
                    if arm == CLOSE:
                        transform = close_transform(runtime, header["marker"])
                    else:
                        transform, _ = break_transform(runtime, group, header["marker"], header["turn"],
                                                       initial_events[arm])
                    rows = runtime.generate(group, NATIVE, draw, args.cap, packed=packed,
                                            first_transform=transform, policy=guard_policy if arm == GUARD else None)
                    for i, (item, row) in enumerate(zip(group, rows, strict=True)):
                        emit(row | {"arm": arm, "prefix_sha256": packed_hash(item["completion_prefix"]),
                                    "interventions": audits[i] if arm == GUARD else []})
                if initial_events[BREAK] != initial_events[GUARD] or len(initial_events[BREAK]) != len(group):
                    raise ValueError("first-step branch probabilities differ")
        emit({"kind": "complete", "seconds": time.monotonic() - started})
    partial.replace(args.output)


def assess(case, record, tokenizer, header, budget):
    y = outcome(case, record, tokenizer, header, budget)
    before = record["tokens"][:budget]
    natural_close = None
    if header["marker"] in before:
        at = before.index(header["marker"])
        natural_close = at + 1 if record["arm"] != CLOSE else None
        before = before[:at]
    _, returned = first_return(before, case["motif"])
    return y | {"first_return_confirmed": returned, "natural_close_token": natural_close}


def score(args):
    from transformers import AutoTokenizer

    if sha(args.raw) != args.fingerprint:
        raise ValueError("guard raw checksum changed")
    with args.raw.open() as stream:
        header, *body, end = map(json.loads, stream)
    if (header["worker_sha256_guard"] != sha(Path(__file__)) or header["panel_sha256"] != sha(args.panel)
            or tuple(header["arms"]) != ARMS or header["periods"] != PERIODS or end["kind"] != "complete"
            or header["tokenizer_sha256"] != sha(args.model / "tokenizer.json")):
        raise ValueError("guard provenance changed")
    source, cases = source_cases(Path(header["source"]), header["source_sha256"], args.panel, header["cap"],
                                 header["limit"], header["source_draw"], header["canary"])
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    eligible, protected, stops = cases_and_tokens(tok, source, cases)
    by_id = {case["row"]["id"]: case for case in eligible}
    raw = {}
    for row in body:
        p, d, a = row["id"], row["draw"], row["arm"]
        case = by_id[p]
        steps = tokens(row)
        key = (p, d, a)
        if (key in raw or row["kind"] != "rollout" or row["seed"] != seed_for(p, d)
                or row["post_seed"] != post_seed(p, d) or not steps or len(steps) > header["cap"]
                or (row["eos_token"] is None and len(steps) != header["cap"])
                or (row["eos_token"] is not None and row["eos_token"] not in header["eos_ids"])
                or row["prefix_sha256"] != packed_hash(case["completion_prefix"])
                or (a in (BREAK, GUARD) and (steps[0] in protected or steps[0] == case["motif"][0]))
                or (a == CLOSE and steps[0] != header["marker"]) or (a != GUARD and row["interventions"])):
            raise ValueError("invalid paired guard trajectory")
        if a == GUARD:
            motif = case["motif"]
            transitions = completions(motif)
            before_stop = range(1, min(len(steps), PERIODS * len(motif)))
            for i in before_stop:
                if steps[i] in blocked_next(steps[:i], len(motif), transitions, stops):
                    raise ValueError("guard trajectory emitted a prohibited recurrence")
            for event in row["interventions"]:
                i = event["step"] - 1
                if (i not in before_stop or event["blocked"] != blocked_next(steps[:i], len(motif), transitions, stops)
                        or not 0 < event["moved_mass"] <= 1.000002):
                    raise ValueError("guard audit disagrees with the sampled trajectory")
        raw[key] = row
    draws = range(header["draw_start"], header["draw_start"] + header["draws"])
    if len(by_id) != header["eligible"] or raw.keys() != {(p, d, a) for p in by_id for d in draws for a in ARMS}:
        raise ValueError("incomplete guard replay matrix")
    if any(raw[p, d, BREAK]["tokens"][0] != raw[p, d, GUARD]["tokens"][0] for p in by_id for d in draws):
        raise ValueError("guard and one-step arms must begin with the same token")
    windows = {}
    for budget in sorted({header["cap"], *(x for x in (512, 1024, 2048) if x <= header["cap"])}):
        cells = {a: {p: [assess(case, raw[p, d, a], tok, header, budget) for d in draws]
                     for p, case in by_id.items()} for a in ARMS}
        windows[str(budget)] = {"arms": {a: tally(y for ys in cells[a].values() for y in ys) for a in ARMS},
                                "guard_minus_break": contrast(cells[GUARD], cells[BREAK]),
                                "guard_minus_close": contrast(cells[GUARD], cells[CLOSE]), "cells": cells}
    return {"header": header, "raw_sha256": args.fingerprint, "seconds": end["seconds"], "windows": windows,
            "interventions": [{"id": p, "draw": d, "steps": raw[p, d, GUARD]["interventions"]}
                              for p in by_id for d in draws]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("run", "score"))
    for name in ("panel", "raw", "model", "data", "checkpoint", "output"):
        p.add_argument("--" + name, type=Path, required=name not in ("data", "checkpoint"))
    p.add_argument("--fingerprint", required=True)
    for name, default in (("draws", 2), ("draw-start", 0), ("source-draw", 0), ("batch", 4),
                          ("cap", 2048), ("limit", None), ("deadline", None)):
        p.add_argument("--" + name, type=int, default=default)
    p.add_argument("--canary", action="store_true")
    args = p.parse_args()
    if args.action == "run":
        if (args.data is None or args.deadline is None or not 128 <= args.cap <= 4096
                or (not args.canary and args.cap < 1024) or min(args.draws, args.batch) < 1
                or min(args.draw_start, args.source_draw) < 0 or (args.limit is not None and args.limit < 1)):
            p.error("run requires data, deadline, valid draw and batch counts, and a registered cap")
        run(args)
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        save(args.output, score(args))


if __name__ == "__main__":
    main()
