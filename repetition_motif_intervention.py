"""Separate breaking an exact token cycle from closing the thinking phase."""

import argparse
import json
from pathlib import Path
import time

from repetition_baseline import render, sha
from repetition_budget_run import packed_hash, tokens
from repetition_budget_score import cut
from repetition_canonical import seed_for
from repetition_followup_aggregate import contrast, tally
from repetition_followup_baseline import runtime_for, verified
from repetition_late_replay import cycle_landmarks, ends_in_motif
from repetition_mediation_queue import save
from repetition_phase_trace import first_long_cycle
from repetition_replay import terminal_cycle
from repetition_restart import score_policy
from repetition_sampler_reach import pack_at
from repetition_state_value import inputs_for, post_seed

NATIVE = "native"
BREAK = "motif_break"
CLOSE = "close"
RESTART = "no_think"
ARMS = (NATIVE, BREAK, CLOSE, RESTART)
EXTENDED_BASELINE_SHA256 = "5757db8ab6eb1fa44962b29297315469455b91f13b6bb289378bd2d181b456f3"


def baseline_reader(source_worker):
    if source_worker == "baseline":
        return verified
    if source_worker != "baseline_extended":
        raise ValueError("unregistered baseline source worker")
    import repetition_followup_baseline_extended as extended
    if sha(Path(extended.__file__)) != EXTENDED_BASELINE_SHA256:
        raise ValueError("archived extended-baseline worker checksum differs")
    return extended.verified


def source_cases(raw_path, fingerprint, panel, cap, limit, source_draw=0, canary=False, source_worker="baseline"):
    header, rows, raw, _ = baseline_reader(source_worker)(raw_path, fingerprint, panel)
    native_cap = next((n for arm, d, n in header["scheme"] if arm == NATIVE and d == source_draw), None)
    if native_cap is None or (not canary and native_cap != 16384):
        raise ValueError("use a verified 16K native baseline draw for evaluation")
    marker = header["marker"]
    cases = []
    for pid, row in sorted(rows.items()):
        trace = raw[pid, NATIVE, source_draw]
        prefix = trace["tokens"]
        before_close = prefix[:prefix.index(marker)] if marker in prefix else prefix
        index = first_long_cycle(before_close[:max(0, native_cap-cap)])
        if index is None:
            continue
        state = cycle_landmarks(before_close, index)
        cases.append({"row": row, "completion_prefix": before_close[:index],
                      "render_sha256": trace["render_sha256"], "prefix_tokens": index,
                      "motif": state["motif"]})
    return header, cases if limit is None else cases[:limit]


def break_transform(runtime, group, marker, turn, events):
    """Condition pre-truncation mass on an ordinary non-motif token for one step."""
    torch = runtime.torch
    excluded = set(runtime.eos) | set(runtime.tokenizer.all_special_ids) | {marker, turn, runtime.pad}

    def transform(filtered, adjusted):
        motif_next = torch.tensor([x["motif"][0] for x in group], device=adjusted.device)
        blocked = torch.zeros_like(adjusted, dtype=torch.bool)
        shared_blocked = sorted(token for token in excluded if 0 <= token < adjusted.shape[1])
        blocked[:, shared_blocked] = True
        blocked.scatter_(1, motif_next[:, None], True)
        modified = adjusted.masked_fill(blocked, -torch.inf)
        log_mass = modified.logsumexp(dim=-1) - adjusted.logsumexp(dim=-1)
        if not bool(log_mass.isfinite().all()):
            raise ValueError("non-motif thinking alternatives have zero reference mass")
        motif_mass = filtered.gather(1, motif_next[:, None]).squeeze(1).tolist()
        for item, p_motif, log_event in zip(group, motif_mass, log_mass.tolist(), strict=True):
            events.append({"id": item["row"]["id"], "original_motif_probability": p_motif,
                           "allowed_log_mass_before_truncation": log_event})
        return modified.softmax(dim=-1)

    return transform, excluded


def close_transform(runtime, marker):
    def transform(filtered, _adjusted):
        forced = runtime.torch.zeros_like(filtered)
        forced[:, marker] = 1.
        return forced

    return transform


def run(args):
    header, cases = source_cases(args.raw, args.fingerprint, args.panel, args.cap, args.limit,
                                 args.source_draw, args.canary, args.source_worker)
    if not cases and args.canary:
        raise ValueError("canary needs at least one confirmed motif")
    runtime = runtime_for(args)
    [marker] = runtime.groups["think"]
    [turn] = runtime.tokenizer.encode("<|im_end|>", add_special_tokens=False)
    if (marker != header["marker"] or turn != header["turn"]
            or any(runtime.manifest[key] != header[key]
                   for key in ("tokenizer_sha256", "checkpoint", "runtime", "sampling"))):
        raise ValueError("replay checkpoint, tokenizer, runtime or decoding protocol differs")
    groups = inputs_for(runtime.tokenizer, cases)
    partial = args.output.with_suffix(".partial.jsonl")
    if args.output.exists():
        raise FileExistsError(args.output)
    with partial.open("x") as stream, runtime.torch.inference_mode():
        def emit(row):
            print(json.dumps(row, allow_nan=False), file=stream, flush=True)

        emit({"kind": "manifest", "worker_sha256_motif": sha(Path(__file__)), "source": str(args.raw),
              "source_sha256": args.fingerprint, "panel_sha256": sha(args.panel),
              "source_draw": args.source_draw, "source_worker": args.source_worker,
              "draws": args.draws, "draw_start": args.draw_start,
              "cap": args.cap, "limit": args.limit, "canary": args.canary,
              "dataset": header["dataset"], "split": header["split"],
              "eligible": len(groups), "panel_size": header["count"], "arms": ARMS,
              "marker": marker, "turn": turn, **runtime.manifest})
        started = time.monotonic()
        for start in range(0, len(groups), args.batch):
            group = groups[start:start+args.batch]
            native_ids = [x["native_ids"] for x in group]
            plain_ids = [runtime.tokenizer.encode(render(runtime.tokenizer, x["row"], "no-think"),
                                                   add_special_tokens=False) for x in group]
            width = max(map(len, native_ids + plain_ids))
            native = pack_at(runtime, native_ids, width)
            plain = pack_at(runtime, plain_ids, width)
            for draw in range(args.draw_start, args.draw_start + args.draws):
                events = []
                break_first, excluded = break_transform(runtime, group, marker, turn, events)
                treatments = ((NATIVE, None), (BREAK, break_first),
                              (CLOSE, close_transform(runtime, marker)), (RESTART, None))
                for arm, treatment in treatments:
                    if time.time() >= args.deadline - 120:
                        raise TimeoutError("motif replay nearing the allocation deadline")
                    restart = arm == RESTART
                    rows = runtime.generate(group, RESTART if restart else NATIVE, draw, args.cap,
                                            packed=plain if restart else native,
                                            first_transform=treatment, start_after_marker=restart)
                    for item, row in zip(group, rows, strict=True):
                        first = next(iter(row["tokens"]), row["eos_token"])
                        if arm == BREAK and (first is None or first in excluded or first == item["motif"][0]):
                            raise ValueError("motif intervention did not produce an ordinary alternative")
                        if arm == CLOSE and first != marker:
                            raise ValueError("phase intervention failed to close")
                        emit(row | {"arm": arm, "prefix_sha256": packed_hash(item["completion_prefix"])})
                if len(events) != len(group):
                    raise ValueError("first-step audit is incomplete")
                for event in events:
                    emit(event | {"kind": "event", "draw": draw})
        emit({"kind": "complete", "seconds": time.monotonic() - started})
    partial.replace(args.output)


def outcome(case, record, tokenizer, header, cap):
    inside, eos = cut(record, "total", cap)
    row = score_policy(case["row"], record | {"tokens": inside,
                       "eos_token": record["eos_token"] if eos else None}, tokenizer, header["turn"])
    close = inside.index(header["marker"]) if header["marker"] in inside else None
    before_close = inside if close is None else inside[:close]
    return row | {"generated_close": close is not None, "terminal_exact": terminal_cycle(inside) is not None,
                  "original_motif_tail": record["arm"] != RESTART and ends_in_motif(before_close, case["motif"]),
                  "correct_without_eos": row["numeric_correct"] and not eos,
                  "cost_from_start": case["prefix_tokens"] + len(inside) + int(eos)}


def report(path, fingerprint, panel, model):
    from transformers import AutoTokenizer

    if sha(path) != fingerprint:
        raise ValueError("motif replay raw checksum changed")
    with path.open() as stream:
        h, *body, end = map(json.loads, stream)
    if (h["worker_sha256_motif"] != sha(Path(__file__)) or h["panel_sha256"] != sha(panel)
            or tuple(h["arms"]) != ARMS or end["kind"] != "complete"):
        raise ValueError("motif replay provenance changed")
    _, selected = source_cases(Path(h["source"]), h["source_sha256"], panel, h["cap"], h["limit"],
                               h["source_draw"], h["canary"], h.get("source_worker", "baseline"))
    cases = {x["row"]["id"]: x for x in selected}
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    excluded = set(tok.all_special_ids) | set(h["eos_ids"]) | {h["marker"], h["turn"], h["pad_id"]}
    raw, events = {}, {}
    for row in body:
        pid, draw = row["id"], row["draw"]
        if row["kind"] == "event":
            if (pid, draw) in events:
                raise ValueError("duplicate first-step event")
            events[pid, draw] = row
            continue
        arm = row["arm"]
        key = (pid, draw, arm)
        expected_seed = (post_seed if arm == RESTART else seed_for)(pid, draw)
        first = next(iter(row["tokens"]), row["eos_token"])
        if (row["kind"] != "rollout" or key in raw or row["seed"] != expected_seed
                or row["post_seed"] != post_seed(pid, draw) or len(tokens(row)) > h["cap"]
                or (row["eos_token"] is None and len(tokens(row)) != h["cap"])
                or row["prefix_sha256"] != packed_hash(cases[pid]["completion_prefix"])
                or (arm == BREAK and (first is None or first in excluded or first == cases[pid]["motif"][0]))
                or (arm == CLOSE and first != h["marker"])
                or (row["eos_token"] is not None and row["eos_token"] not in h["eos_ids"])):
            raise ValueError("motif replay evidence disagrees with the intervention")
        raw[key] = row
    draws = range(h["draw_start"], h["draw_start"] + h["draws"])
    if (len(cases) != h["eligible"] or raw.keys() != {(p, d, a) for p in cases for d in draws for a in ARMS}
            or events.keys() != {(p, d) for p in cases for d in draws}):
        raise ValueError("motif replay score requires every paired branch")
    cuts = sorted({h["cap"], *(n for n in (512, 1024, 2048, 4096) if n <= h["cap"])})
    windows = {}
    for cap in cuts:
        arms = {a: {p: [outcome(case, raw[p, d, a], tok, h, cap) for d in draws]
                    for p, case in cases.items()} for a in ARMS}
        tables = {}
        for a, by_case in arms.items():
            flat = [row for group in by_case.values() for row in group]
            table = tally(flat)
            table.update({k: sum(r[k] for r in flat) for k in ("original_motif_tail", "correct_without_eos")})
            tables[a] = table
        windows[str(cap)] = {"arms": tables, "break_minus_close": contrast(arms[BREAK], arms[CLOSE]),
                             "break_minus_native": contrast(arms[BREAK], arms[NATIVE]),
                             "break_minus_restart": contrast(arms[BREAK], arms[RESTART]), "cells": arms}
    return {"header": h, "raw_sha256": fingerprint, "seconds": end["seconds"],
            "windows": windows, "events": list(events.values())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("run", "score"))
    for name in ("panel", "raw", "model", "data", "checkpoint", "output"):
        p.add_argument("--" + name, type=Path, required=name not in ("data", "checkpoint"))
    p.add_argument("--fingerprint", required=True)
    p.add_argument("--source-worker", choices=("baseline", "baseline_extended"), default="baseline")
    for name, default in (("draws", 2), ("draw-start", 0), ("source-draw", 0), ("batch", 4),
                          ("cap", 2048), ("limit", None), ("deadline", None)):
        p.add_argument("--" + name, type=int, default=default)
    p.add_argument("--canary", action="store_true")
    a = p.parse_args()
    if a.action == "run":
        if (a.data is None or a.deadline is None or not 128 <= a.cap <= 8192
                or (not a.canary and a.cap < 1024) or min(a.draws, a.batch) < 1
                or min(a.draw_start, a.source_draw) < 0 or (a.limit is not None and a.limit < 1)):
            p.error("run requires data, deadline, valid draw and batch counts, and a registered continuation cap")
        run(a)
    else:
        if a.output.exists():
            raise FileExistsError(a.output)
        save(a.output, report(a.raw, a.fingerprint, a.panel, a.model))


if __name__ == "__main__":
    main()
