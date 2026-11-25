"""Identify which cache component changes the queried numeral after a value-only swap."""

import argparse
from contextlib import contextmanager
import math
from pathlib import Path
from statistics import mean
import time
from types import SimpleNamespace

from repetition_baseline import sha
from repetition_cache_swap import replace_states, state_digest
from repetition_delayed_numeric import candidate_ids, conditional_scores, numerical_interval, stable_logsumexp
from repetition_delayed_query import FIELDS, KV, NONE, prefixes
from repetition_delayed_timing import at_offset, frozen_panel
from repetition_delayed_transport import MATRIX
from repetition_followup_baseline import runtime_for
from repetition_mediation_queue import load, save
from repetition_sampler_reach import pack_at


COMPONENTS = {"conv": FIELDS, "matrix": MATRIX, "kv": KV}
KEEP = "keep"
ARMS = {KEEP: ()}
for a, x in COMPONENTS.items():
    ARMS.update({f"{arm}+{a}" if arm != KEEP else a: fields + x for arm, fields in tuple(ARMS.items())})
FULL = "conv+matrix+kv"
PROTOCOL = "delayed-read-cross-value-v1"


@contextmanager
def cross_receiver(runtime, width, query, arm, _donors, audit, _marker):
    torch = runtime.torch
    fields = ARMS[arm]

    def pre(_module, _args, kwargs):
        hit = at_offset(kwargs, width, {0: query[0]})
        if hit is None:
            return
        if audit:
            raise ValueError("duplicate value-cross intervention")
        _, cache = hit
        all_fields = FIELDS + MATRIX + KV
        audit["pre_sha"] = state_digest(cache, all_fields, torch)
        if fields:
            untouched = {f: tuple(map(id, getattr(cache, f))) for f in all_fields if f not in fields}
            donor = SimpleNamespace(**{f: [None if x is None else x.flip(0) for x in getattr(cache, f)] for f in fields})
            audit["changed_layers"] = replace_states(cache, donor, torch, fields)
            if any(ids != tuple(map(id, getattr(cache, f))) for f, ids in untouched.items()):
                raise ValueError("value swap touched an unselected cache component")

    hook = runtime.model.register_forward_pre_hook(pre, with_kwargs=True)
    try:
        yield
    finally:
        hook.remove()
    if not audit:
        raise ValueError("missed value-cross intervention")


def cross_case(runtime, case):
    query = case["query"]["read"]
    width = max(len(ids) for arm in (KEEP, NONE, "fresh_record") for ids in prefixes(case, arm))
    receiver_ids = prefixes(case, KEEP)
    if len(receiver_ids[0]) != len(receiver_ids[1]) or len(case["record_changed_positions"]) == 0:
        raise ValueError("value-cross receivers are not token aligned")
    packed = pack_at(runtime, receiver_ids, width)
    scaffold, numerals = candidate_ids(runtime.tokenizer, case["targets"]["read"])
    reference, scores, outcomes = {}, {}, {}
    for arm in ARMS:
        matrix = []
        for candidate, numeral in enumerate(numerals):
            lp, audit = conditional_scores(runtime, packed, query, scaffold, numeral, arm, {},
                                           scoring_hook=cross_receiver)
            if arm == KEEP:
                reference[candidate] = audit["pre_sha"]
            elif audit["pre_sha"] != reference[candidate]:
                raise ValueError("receiver differed from KEEP before the value swap")
            matrix.append(lp)
        scores[arm] = matrix
        outcomes[arm] = {"correct_minus_opposite_nats": [matrix[v][v] - matrix[1-v][v] for v in (0, 1)],
                         "log_candidate_mass": [stable_logsumexp(matrix[0][v], matrix[1][v]) for v in (0, 1)]}
    error = max(abs(scores[FULL][c][v] - scores[KEEP][c][1-v]) for c in (0, 1) for v in (0, 1))
    if not math.isfinite(error) or error > 1e-4:
        raise ValueError(f"full crossed cache did not reconstruct the donor's conditional scores: {error:.6g}")
    return {"id": case["id"], "family": case["family"], "slot": case["slot"],
            "full_donor_recovery_error_nats": error, "arms": outcomes}


def run(args):
    runtime = runtime_for(args)
    panel = frozen_panel(args.panel, args.panel_sha256, runtime.tokenizer, runtime.manifest["tokenizer_sha256"])
    cases = panel["panels"][args.split][args.start:args.start + args.count]
    if len(cases) != args.count or not cases:
        raise ValueError("incomplete crossed-value shard")
    rows = []
    start = time.monotonic()
    with runtime.torch.inference_mode():
        for case in cases:
            if time.time() >= args.deadline - 120:
                raise TimeoutError("crossed-value shard nearing allocation deadline")
            rows.append(cross_case(runtime, case))
    return {"protocol": PROTOCOL, "worker_sha": sha(Path(__file__)), "panel_sha256": args.panel_sha256,
            "split": args.split, "start": args.start, "count": args.count, "seconds": time.monotonic() - start,
            "rows": rows, **runtime.manifest}


def summarize(args):
    cases, first = {}, None
    for path, pin in zip(args.reports, args.fingerprints, strict=True):
        if sha(path) != pin:
            raise ValueError("crossed-value shard checksum changed")
        report = load(path)
        identity = {k: report[k] for k in ("panel_sha256", "worker_sha", "split", "checkpoint", "tokenizer_sha256",
                                          "runtime", "protocol")}
        if report["protocol"] != PROTOCOL or (first is not None and identity != first):
            raise ValueError("mixed crossed-value shards")
        first = identity
        if len(report["rows"]) != report["count"]:
            raise ValueError("incomplete crossed-value rows")
        for row in report["rows"]:
            if row["id"] in cases or set(row["arms"]) != set(ARMS):
                raise ValueError("duplicate or incomplete crossed-value case")
            cases[row["id"]] = row

    def values(row, arm):
        return row["arms"][arm]["correct_minus_opposite_nats"]

    def effect(left, right):
        return numerical_interval({pid: [x - y for x, y in zip(values(row, left), values(row, right), strict=True)]
                                   for pid, row in cases.items()})

    field_set = {frozenset(fields): arm for arm, fields in ARMS.items()}
    marginal = {}
    for component, fields in COMPONENTS.items():
        treated = frozenset(fields)
        pairs = [(arm, field_set[base - treated]) for base, arm in field_set.items() if treated <= base]
        marginal[component] = numerical_interval({
            pid: [mean(values(row, a)[v] - values(row, b)[v] for a, b in pairs) for v in (0, 1)]
            for pid, row in cases.items()
        })
    means = {arm: mean(mean(values(row, arm)) for row in cases.values()) for arm in ARMS}
    return first | {
        "prompts": len(cases), "score_shas": args.fingerprints, "mean_correct_minus_opposite_nats": means,
        "mean_log_candidate_mass": {arm: mean(mean(row["arms"][arm]["log_candidate_mass"]) for row in cases.values())
                                    for arm in ARMS},
        "value_flip_rate": {arm: mean(x < 0 for row in cases.values() for x in values(row, arm)) for arm in ARMS},
        "single_component_minus_keep": {arm: effect(arm, KEEP) for arm in COMPONENTS},
        "factorial_mean_component_effect": marginal,
        "max_full_donor_recovery_error_nats": max(r["full_donor_recovery_error_nats"] for r in cases.values()),
    }


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
