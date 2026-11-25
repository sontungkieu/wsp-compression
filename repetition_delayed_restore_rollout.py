"""Test whether native cache restoration rescues correct completion after a conv pulse."""

import argparse
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
import time

from repetition_baseline import sha
from repetition_cache_swap import replace_states
from repetition_delayed_query import FIELDS, KEEP, NONE, THINK_END, assess, interval_by_id, prefixes, query_policy, tally
from repetition_delayed_timing import at_offset, donor_states, frozen_panel
from repetition_delayed_transport import ALL, ANCHORS, PULSE, RESTORE, marker_forward
from repetition_followup_baseline import runtime_for
from repetition_mediation_queue import load, save
from repetition_sampler_reach import pack_at


ARMS = {"canary": (KEEP, PULSE, *RESTORE), "eval": (KEEP, PULSE, "conv", "conv_matrix", "conv_kv")}
PROTOCOL = "fixed-read-cache-restore-rollout-v1"


@contextmanager
def swap_cache(runtime, width, query, edits, audit):
    def pre(_module, _args, kwargs):
        hit = at_offset(kwargs, width, {offset: query[offset] for offset in edits})
        if hit is None:
            return
        offset, cache = hit
        source, fields = edits[offset]
        if offset in audit:
            raise ValueError("duplicate completion cache intervention")
        unchanged = {f: tuple(map(id, getattr(cache, f))) for f in ALL if f not in fields}
        audit[offset] = replace_states(cache, source, runtime.torch, fields)
        if any(ids != tuple(map(id, getattr(cache, f))) for f, ids in unchanged.items()):
            raise ValueError("completion intervention changed an unselected cache")

    hook = runtime.model.register_forward_pre_hook(pre, with_kwargs=True)
    try:
        yield
    finally:
        hook.remove()
    if set(audit) != set(edits):
        raise ValueError("completion omitted a requested cache intervention")


def run_case(runtime, case, anchor, cap, draws, arms, turn):
    query = case["query"]["read"]
    width = max(len(ids) for arm in (KEEP, NONE, "fresh_record") for ids in prefixes(case, arm))
    packed = pack_at(runtime, prefixes(case, KEEP), width)
    _, native, _ = marker_forward(runtime, packed, query, captures=(anchor,))
    donor = donor_states(runtime, case, width, query)[0]
    pulse = {0: (donor, FIELDS)}
    policy = query_policy(runtime, query)
    group = [{"row": {"id": case["id"], "task": "math", "gold": str(gold)}, "pending": True,
              "presence_seed_ids": []} for gold in case["targets"]["read"]]
    out = []
    for draw in range(draws):
        traces = {}
        for arm in arms:
            edits = {} if arm == KEEP else pulse if arm == PULSE else pulse | {anchor: (native[anchor], RESTORE[arm])}
            audit = {}
            traces[arm] = runtime.generate(
                group, "native", draw, len(query) + 1 + cap, packed=packed, policy=policy,
                hook=swap_cache(runtime, width, query, edits, audit),
            )
            for variant, row in enumerate(traces[arm]):
                if row["tokens"][:len(query) + 1] != query + [THINK_END]:
                    raise ValueError("completion did not use the fixed query and marker")
                control = traces[KEEP][variant]
                if row["seed"] != control["seed"] or row["post_seed"] != control["post_seed"]:
                    raise ValueError("completion changed a matched draw")
                if arm == "all" and (row["tokens"] != control["tokens"] or row["eos_token"] != control["eos_token"]):
                    raise ValueError("full restored cache did not reconstruct the seeded KEEP completion")
                scored = {str(window): assess(runtime.tokenizer, case, row | {"mode": "read", "variant": variant},
                                              window, turn) for window in sorted({min(256, cap), cap})}
                out.append(row | {"arm": arm, "variant": variant, "edit_audit": audit, "windows": scored})
    return out


def run(args):
    runtime = runtime_for(args)
    panel = frozen_panel(args.panel, args.panel_sha256, runtime.tokenizer, runtime.manifest["tokenizer_sha256"])
    cases = panel["panels"][args.split][args.start:args.start + args.count]
    if len(cases) != args.count or not cases:
        raise ValueError("incomplete restoration completion shard")
    [turn] = runtime.tokenizer.encode("<|im_end|>", add_special_tokens=False)
    rows = []
    start = time.monotonic()
    with runtime.torch.inference_mode():
        for case in cases:
            if time.time() >= args.deadline - 120:
                raise TimeoutError("restoration completion shard nearing the deadline")
            rows.extend(run_case(runtime, case, args.anchor, args.cap, args.draws, ARMS[args.split], turn))
    return {"protocol": PROTOCOL, "worker_sha": sha(Path(__file__)), "panel_sha256": args.panel_sha256,
            "start": args.start, "count": args.count, "cap": args.cap, "draws": args.draws, "anchor": args.anchor,
            "split": args.split, "seconds": time.monotonic() - start, "rows": rows, **runtime.manifest}


def summarize(args):
    cells = defaultdict(dict)
    traces = defaultdict(dict)
    first = None
    for path, pin in zip(args.reports, args.fingerprints, strict=True):
        if sha(path) != pin:
            raise ValueError("restoration completion checksum changed")
        report = load(path)
        identity = {k: report[k] for k in ("protocol", "split", "anchor", "cap", "draws", "panel_sha256",
                                          "worker_sha", "tokenizer_sha256", "checkpoint", "runtime", "sampling")}
        if report["protocol"] != PROTOCOL or (first is not None and identity != first):
            raise ValueError("mixed restoration completion shards")
        first = identity
        if len(report["rows"]) != report["count"] * report["draws"] * 2 * len(ARMS[report["split"]]):
            raise ValueError("incomplete restoration completion shard")
        for row in report["rows"]:
            key = row["id"], row["draw"], row["variant"]
            if key in cells[row["arm"]]:
                raise ValueError("duplicate restoration completion cell")
            cells[row["arm"]][key] = row["windows"]
            traces[row["arm"]][key] = (row["tokens"], row["eos_token"], row["seed"], row["post_seed"])
    keys = cells[KEEP].keys()
    if set(cells) != set(ARMS[first["split"]]) or any(arm.keys() != keys for arm in cells.values()):
        raise ValueError("unpaired restoration completion cells")
    parity = {}
    for arm, group in traces.items():
        if any(group[k][2:] != traces[KEEP][k][2:] for k in keys):
            raise ValueError("restoration completion changed the paired random stream")
        parity[arm] = sum(group[k][:2] == traces[KEEP][k][:2] for k in keys)
    result = {}
    for cap in sorted({min(256, first["cap"]), first["cap"]}):
        cap = str(cap)
        scores = {arm: {key: row[cap] for key, row in group.items()} for arm, group in cells.items()}

        def diff(a, b):
            return interval_by_id({k: int(scores[a][k]["correct_eos"]) - int(scores[b][k]["correct_eos"])
                                   for k in keys})

        result[cap] = {
            "counts": {arm: tally(group.values()) for arm, group in scores.items()},
            "correct_eos_vs_keep": {arm: diff(arm, KEEP) for arm in cells if arm != KEEP},
            "restore_vs_pulse": {arm: diff(arm, PULSE) for arm in cells if arm in RESTORE},
            "matrix_vs_conv": diff("conv_matrix", "conv"),
            "kv_vs_conv": diff("conv_kv", "conv"),
        }
    return first | {"prompts": len({key[0] for key in keys}), "score_shas": args.fingerprints,
                    "exact_trace_plus_eos_matches_keep": parity, "windows": result}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("run", "summarize"))
    for key in ("panel", "model", "data", "checkpoint", "output"):
        p.add_argument("--" + key, type=Path, required=key == "output")
    p.add_argument("--panel-sha256")
    p.add_argument("--split", choices=tuple(ARMS))
    p.add_argument("--anchor", choices=ANCHORS, type=int, default=7)
    for key, default in (("start", 0), ("count", 1), ("draws", 1), ("cap", 512), ("deadline", None)):
        p.add_argument("--" + key, type=int, default=default)
    p.add_argument("--reports", type=Path, nargs="+")
    p.add_argument("--fingerprints", nargs="+")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.action == "run":
        if (args.split is None or args.deadline is None or args.start < 0 or args.count <= 0
                or (args.cap, args.draws) != ((128, 1) if args.split == "canary" else (512, 2))):
            p.error("expected canary 128/1 or eval 512/2 and a valid scope")
        save(args.output, run(args))
    else:
        save(args.output, summarize(args))


if __name__ == "__main__":
    main()
