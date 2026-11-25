"""Measure record-conditioned numeric odds under a fixed answer scaffold."""

import argparse
import math
from pathlib import Path
from statistics import mean
import time

from repetition_baseline import sha
from repetition_canonical import paired_interval
from repetition_delayed_query import KEEP, NONE, PAD_AFTER, PAD_SPLIT, PADDED, THINK_END, prefixes
from repetition_delayed_timing import RESETS, donor_states, receiver
from repetition_followup_baseline import runtime_for
from repetition_mediation_queue import load, save
from repetition_sampler_reach import pack_at


PLACEMENTS = {"12+0": PADDED, "6+6": PAD_SPLIT, "0+12": PAD_AFTER}
PROTOCOL = "record-conditioned-exact-numeral-v1"


def candidate_ids(tokenizer, values):
    scaffold = tokenizer.encode("\n", add_special_tokens=False)
    digits = [tokenizer.encode(str(value), add_special_tokens=False) for value in values]
    if (not scaffold or not digits[0] or len(digits[0]) != len(digits[1]) or digits[0] == digits[1]
            or any(tokenizer.decode(scaffold + ids) != "\n" + str(value)
                   for ids, value in zip(digits, values, strict=True))):
        raise ValueError("paired values do not have distinct, equal-length numeral continuations")
    return scaffold, digits


def panel_cases(args, runtime):
    panels = {}
    for placement, path, pin in zip(PLACEMENTS, args.panels, args.panel_sha256, strict=True):
        if sha(path) != pin:
            raise ValueError("delayed READ panel pin changed")
        plan = load(path)
        if (plan["protocol"] != PLACEMENTS[placement]
                or plan["tokenizer_sha256"] != runtime.manifest["tokenizer_sha256"]):
            raise ValueError("wrong padding placement or tokenizer")
        panels[placement] = plan["panels"][args.split][args.start:args.start + args.count]
    if any(len(cases) != args.count for cases in panels.values()):
        raise ValueError("incomplete numeral cohort")
    for aligned in zip(*panels.values(), strict=True):
        original = aligned[0]
        for case in aligned:
            if ({k: v for k, v in case.items() if k != "query"} !=
                    {k: v for k, v in original.items() if k != "query"}
                    or len(case["query"]["read"]) != 23
                    or case["query"]["apply"] != original["query"]["apply"]):
                raise ValueError("numeric comparison changed case identity, content or query length")
    return panels


def conditional_scores(runtime, packed, query, scaffold, numeral, arm, donors, scoring_hook=receiver):
    torch = runtime.torch
    width = packed["input_ids"].shape[-1]
    stop = min(runtime.eos)
    forced = query + [THINK_END] + scaffold + numeral + [stop]
    first = len(query) + 1 + len(scaffold)
    logps, audit, marker = [], {}, []

    class ForcedNumeral(runtime.transformers.LogitsProcessor):
        def __call__(self, ids, scores):
            index = ids.shape[-1] - width
            token = forced[index]
            if first <= index < first + len(numeral):
                logps.append(scores.float().log_softmax(-1)[:, token].detach())
            return torch.full_like(scores, -torch.inf).scatter_(1, ids.new_full((2, 1), token), 0.)

    cfg = runtime.transformers.GenerationConfig(
        do_sample=False, use_cache=True, max_new_tokens=len(forced), eos_token_id=sorted(runtime.eos),
        pad_token_id=runtime.pad,
    )
    with scoring_hook(runtime, width, query, arm, donors, audit, marker):
        out = runtime.model.generate(**packed, generation_config=cfg, logits_processor=[ForcedNumeral()],
                                     **runtime.defaults)
    if out[:, width:].tolist() != [forced, forced] or len(logps) != len(numeral):
        raise ValueError("incomplete teacher-forced numeral")
    return torch.stack(logps).sum(0).tolist(), audit


def stable_logsumexp(a, b):
    return max(a, b) + math.log1p(math.exp(-abs(a - b)))


def numerical_interval(groups):
    paired = paired_interval(groups)
    return {"delta_nats": paired["delta_pp"] / 100,
            "ci95_nats": [x / 100 for x in paired["ci95_pp"]], "prompts": paired["prompts"]}


def evidence(runtime, case, placement):
    query = case["query"]["read"]
    width = max(len(ids) for arm in (KEEP, NONE, "fresh_record") for ids in prefixes(case, arm))
    packed = pack_at(runtime, prefixes(case, KEEP), width)
    donors = donor_states(runtime, case, width, query)
    scaffold, numerals = candidate_ids(runtime.tokenizer, case["targets"]["read"])
    arms = (KEEP, *RESETS) if placement == "6+6" else (KEEP, RESETS[0])
    scores, reference = {}, []
    for arm in arms:
        matrix = []
        for index, numeral in enumerate(numerals):
            lp, audit = conditional_scores(runtime, packed, query, scaffold, numeral, arm, donors)
            matrix.append(lp)
            if arm == KEEP:
                reference.append(audit)
            else:
                offset = arm.rsplit("_", 1)[1]
                if any(audit[offset][k] != reference[index][offset][k]
                       for k in ("kv", "mask", "token", "cache_position")):
                    raise ValueError("receiver changed before the fixed numeral intervention")
        if not all(math.isfinite(x) for pair in matrix for x in pair):
            raise ValueError("non-finite conditional numeral score")
        scores[arm] = {"correct_minus_opposite_nats": [matrix[v][v] - matrix[1-v][v] for v in (0, 1)],
                       "log_total_candidate_mass": [stable_logsumexp(matrix[0][v], matrix[1][v]) for v in (0, 1)]}
    return scores


def run(args):
    runtime = runtime_for(args)
    panels = panel_cases(args, runtime)
    rows = []
    start = time.monotonic()
    with runtime.torch.inference_mode():
        for index, case in enumerate(panels["6+6"]):
            if time.time() >= args.deadline - 120:
                raise TimeoutError("READ numeral shard nearing allocation deadline")
            scores = {p: evidence(runtime, panel[index], p) for p, panel in panels.items()}
            rows.append({"id": case["id"], "family": case["family"], "slot": case["slot"], "placements": scores})
    return {"protocol": PROTOCOL, "worker_sha": sha(Path(__file__)), "panels": args.panel_sha256,
            "split": args.split, "start": args.start, "count": args.count, "rows": rows,
            "seconds": time.monotonic() - start, **runtime.manifest}


def summarize(args):
    merged = {}
    first = None
    for path, pin in zip(args.reports, args.fingerprints, strict=True):
        if sha(path) != pin:
            raise ValueError("READ numeral shard checksum changed")
        report = load(path)
        identity = {k: report[k] for k in ("protocol", "worker_sha", "panels", "split", "checkpoint",
                                          "runtime", "tokenizer_sha256")}
        if report["protocol"] != PROTOCOL or (first is not None and identity != first):
            raise ValueError("mixed READ numeral shards")
        first = identity
        if len(report["rows"]) != report["count"]:
            raise ValueError("incomplete numeral shard")
        for row in report["rows"]:
            if row["id"] in merged or set(row["placements"]) != set(PLACEMENTS):
                raise ValueError("duplicate or incomplete READ numeral row")
            merged[row["id"]] = row["placements"]

    def metric(spec, field):
        placement, arm = spec
        return {pid: row[placement][arm][field] for pid, row in sorted(merged.items())}

    output = {}
    for name in ("correct_minus_opposite_nats", "log_total_candidate_mass"):
        effects = {}
        summaries, deltas = {}, {}
        for placement in PLACEMENTS:
            keep = metric((placement, KEEP), name)
            arms = RESETS if placement == "6+6" else RESETS[:1]
            summaries[placement] = {KEEP: mean(mean(row) for row in keep.values())}
            effects[placement] = {}
            for arm in arms:
                test = metric((placement, arm), name)
                summaries[placement][arm] = mean(mean(row) for row in test.values())
                groups = {pid: [a - b for a, b in zip(test[pid], keep[pid], strict=True)] for pid in keep}
                if arm == RESETS[0]:
                    deltas[placement] = groups
                effects[placement][arm] = numerical_interval(groups)
        contrasts = {f"6+6_minus_{p}": numerical_interval({
            pid: [a - b for a, b in zip(deltas["6+6"][pid], deltas[p][pid], strict=True)] for pid in merged
        }) for p in ("12+0", "0+12")}
        output[name] = {"means": summaries, "reset_minus_keep": effects, "reset0_placement_contrasts": contrasts}
    return first | {"prompts": len(merged), "shard_shas": args.fingerprints, "metrics": output}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("run", "summarize"))
    for key in ("model", "data", "checkpoint", "output"):
        p.add_argument("--" + key, type=Path, required=key == "output")
    p.add_argument("--panels", type=Path, nargs=3)
    p.add_argument("--panel-sha256", nargs=3)
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
