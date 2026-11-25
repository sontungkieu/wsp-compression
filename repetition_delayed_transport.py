"""Locate where an early convolution perturbation propagates in a fixed READ query."""

import argparse
from collections import defaultdict
import math
from pathlib import Path
from statistics import mean
import time
from types import SimpleNamespace

from repetition_baseline import sha
from repetition_cache_swap import replace_states
from repetition_canonical import paired_interval
from repetition_delayed_query import FIELDS, KEEP, KV, NONE, THINK_END, prefixes
from repetition_delayed_timing import at_offset, donor_states, frozen_panel
from repetition_followup_baseline import runtime_for
from repetition_mediation_queue import load, save
from repetition_sampler_reach import pack_at


ANCHORS = (7, 17)
MATRIX = ("recurrent_states",)
ALL = FIELDS + MATRIX + KV
RESTORE = {"conv": FIELDS, "conv_matrix": FIELDS + MATRIX, "conv_kv": FIELDS + KV, "all": ALL}
PULSE = "pulse_0"
PROTOCOL = "fixed-read-cache-restoration-v1"


def snapshot(cache):
    return SimpleNamespace(**{field: [None if x is None else x.detach().clone() for x in getattr(cache, field)]
                              for field in ALL})


def marker_forward(runtime, packed, query, captures=(), edits=None):
    """Return raw first-answer logits and snapshots before the specified forced tokens."""
    edits = {} if edits is None else edits
    torch = runtime.torch
    width = packed["input_ids"].shape[-1]
    forced = query + [THINK_END, min(runtime.eos)]
    targets = {offset: forced[offset] for offset in (*captures, *edits.keys(), len(query))}
    states, changes, logits = {}, {}, []
    at_marker = False

    class Forced(runtime.transformers.LogitsProcessor):
        def __call__(self, ids, scores):
            token = forced[ids.shape[1] - width]
            return torch.full_like(scores, -torch.inf).scatter_(1, ids.new_full((2, 1), token), 0.)

    def pre(_module, _args, kwargs):
        nonlocal at_marker
        hit = at_offset(kwargs, width, targets)
        if hit is None:
            return
        offset, cache = hit
        at_marker = offset == len(query)
        if offset in captures:
            if offset in states:
                raise ValueError("duplicate KEEP cache snapshot")
            states[offset] = snapshot(cache)
        if offset in edits:
            source, fields = edits[offset]
            if offset in changes:
                raise ValueError("duplicate cache restoration")
            untouched = {f: tuple(map(id, getattr(cache, f))) for f in ALL if f not in fields}
            changes[offset] = replace_states(cache, source, torch, fields)
            if any(ids != tuple(map(id, getattr(cache, f))) for f, ids in untouched.items()):
                raise ValueError("restoration mutated an unselected cache component")

    def post(_module, _args, _kwargs, output):
        if at_marker:
            logits.append(output.logits[:, -1].float().detach().clone())

    h_pre = runtime.model.register_forward_pre_hook(pre, with_kwargs=True)
    h_post = runtime.model.register_forward_hook(post, with_kwargs=True)
    try:
        cfg = runtime.transformers.GenerationConfig(
            do_sample=False, use_cache=True, max_new_tokens=len(forced),
            eos_token_id=sorted(runtime.eos), pad_token_id=runtime.pad,
        )
        out = runtime.model.generate(**packed, generation_config=cfg, logits_processor=[Forced()], **runtime.defaults)
    finally:
        h_pre.remove()
        h_post.remove()
    if (len(logits) != 1 or set(states) != set(captures) or set(changes) != set(edits)
            or out[:, width:].tolist() != [forced, forced]):
        raise ValueError("incomplete or incorrectly timed fixed-query forward")
    return logits[0], states, changes


def probe(runtime, case):
    torch = runtime.torch
    query = case["query"]["read"]
    width = max(len(ids) for arm in (KEEP, NONE, "fresh_record") for ids in prefixes(case, arm))
    packed = pack_at(runtime, prefixes(case, KEEP), width)
    keep, states, _ = marker_forward(runtime, packed, query, captures=ANCHORS)
    donor = donor_states(runtime, case, width, query)[0]
    pulse = {0: (donor, FIELDS)}
    scores = {}

    def add(arm, logits, changes):
        tv = (keep.softmax(-1) - logits.softmax(-1)).abs().sum(-1).mul(.5).tolist()
        linf = (keep - logits).abs().amax(-1).tolist()
        if any(not math.isfinite(x) for x in tv + linf):
            raise ValueError("non-finite first-answer distribution")
        scores[arm] = {"tv": tv, "max_logit_delta": linf, "changed_layers": changes}

    logits, _, changes = marker_forward(runtime, packed, query, edits=pulse)
    add(PULSE, logits, changes)
    for anchor in ANCHORS:
        for name, fields in RESTORE.items():
            logits, _, changes = marker_forward(
                runtime, packed, query, edits=pulse | {anchor: (states[anchor], fields)},
            )
            if name == "all" and not torch.equal(logits, keep):
                raise ValueError(f"full restore at offset {anchor} did not reconstruct the KEEP logits")
            add(f"{anchor}_{name}", logits, changes)
    return {"id": case["id"], "family": case["family"], "slot": case["slot"], "arms": scores}


def run(args):
    runtime = runtime_for(args)
    panel = frozen_panel(args.panel, args.panel_sha256, runtime.tokenizer, runtime.manifest["tokenizer_sha256"])
    cohort = panel["panels"][args.split][args.start:args.start + args.count]
    if not cohort or len(cohort) != args.count:
        raise ValueError("incomplete fixed READ shard")
    rows = []
    start = time.monotonic()
    with runtime.torch.inference_mode():
        for item in cohort:
            if time.time() >= args.deadline - 120:
                raise TimeoutError("READ restoration probe nearing allocation deadline")
            rows.append(probe(runtime, item))
    return {"protocol": PROTOCOL, "worker_sha": sha(Path(__file__)), "panel_sha256": args.panel_sha256,
            "split": args.split, "start": args.start, "count": args.count, "anchors": ANCHORS,
            "seconds": time.monotonic() - start, "rows": rows, **runtime.manifest}


def summarize(args):
    rows = defaultdict(dict)
    first = None
    cases = set()
    expected = {PULSE, *(f"{offset}_{name}" for offset in ANCHORS for name in RESTORE)}
    for path, pin in zip(args.reports, args.fingerprints, strict=True):
        if sha(path) != pin:
            raise ValueError("READ restoration shard checksum changed")
        report = load(path)
        identity = {k: report[k] for k in ("protocol", "worker_sha", "panel_sha256", "split",
                                          "checkpoint", "tokenizer_sha256", "runtime")}
        if report["protocol"] != PROTOCOL or (first is not None and identity != first):
            raise ValueError("mixed READ restoration shards")
        first = identity
        if len(report["rows"]) != report["count"]:
            raise ValueError("incomplete READ restoration shard")
        for row in report["rows"]:
            if row["id"] in cases or set(row["arms"]) != expected:
                raise ValueError("duplicate or incomplete READ restoration row")
            cases.add(row["id"])
            for arm, values in row["arms"].items():
                if len(values["tv"]) != 2 or any(not 0 <= x <= 1.000001 for x in values["tv"]):
                    raise ValueError("invalid paired first-answer total variation")
                rows[arm][row["id"]] = values["tv"]

    def diff(a, b):
        return paired_interval({pid: [x - y for x, y in zip(rows[a][pid], rows[b][pid], strict=True)]
                                for pid in sorted(cases)})

    arm_tv = {arm: mean(mean(pair) for pair in cases_by_id.values()) for arm, cases_by_id in rows.items()}
    contrasts = {}
    for anchor in ANCHORS:
        conv = f"{anchor}_conv"
        contrasts[str(anchor)] = {
            "pulse_minus_conv": diff(PULSE, conv),
            "conv_minus_conv_matrix": diff(conv, f"{anchor}_conv_matrix"),
            "conv_minus_conv_kv": diff(conv, f"{anchor}_conv_kv"),
        }
    return first | {"prompts": len(cases), "shard_shas": args.fingerprints, "mean_tv": arm_tv,
                    "paired_tv_differences_pp": contrasts}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("run", "summarize"))
    for key in ("panel", "model", "data", "checkpoint", "output"):
        p.add_argument("--" + key, type=Path, required=key == "output")
    p.add_argument("--panel-sha256")
    p.add_argument("--split", choices=("canary", "eval"))
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=1)
    p.add_argument("--deadline", type=int)
    p.add_argument("--reports", type=Path, nargs="+")
    p.add_argument("--fingerprints", nargs="+")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.action == "run" and (args.deadline is None or args.split is None or args.start < 0 or args.count < 1):
        p.error("run requires a frozen split, valid range and deadline")
    save(args.output, run(args) if args.action == "run" else summarize(args))


if __name__ == "__main__":
    main()
