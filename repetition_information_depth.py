"""Measure held-out hidden-checkpoint retention at nested continuation budgets."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

from repetition_baseline import math_answer, sha
from repetition_budget_run import packed_hash, tokens
from repetition_budget_score import cut
from repetition_cache_factorial import BOTH, CONV, DROP, MATRIX, NATIVE, ZERO_CONV, sample
from repetition_canonical import paired_interval, seed_for
from repetition_followup_baseline import runtime_for
from repetition_followup_evidence import new_row
from repetition_followup_information import CELLS, EVIDENCE
from repetition_mediation_queue import load, save
from repetition_sampler_reach import pack_at
from repetition_state_value import inputs_for, post_seed

VERIFIED = "verified_record"
ARMS = (NATIVE, CONV, MATRIX, BOTH, ZERO_CONV, DROP, EVIDENCE, VERIFIED)
METRICS = ("numeric_correct", "correct_eos", "correct_without_eos", "wrong_eos", "missing_eos", "cap", "tokens")


def cases(panel, split, start, count):
    result = load(panel)["panels"][split][start:start + count]
    if len(result) != count or not result:
        raise ValueError("information shard lies outside the frozen panel")
    return result


def windows(cap):
    return [n for n in (128, 256, 512, 1024, 2048) if n <= cap]


def check_question_only(traces):
    for i in range(0, len(traces), len(CELLS)):
        block = traces[i:i + len(CELLS)]
        if len(block) != len(CELLS) or any(tokens(block[j]) != tokens(block[j + 1]) for j in (0, 2)):
            raise ValueError("question-only arm leaked the hidden value")


def run(args):
    plan = cases(args.panel, args.split, args.start, args.count)
    r = runtime_for(args)
    marker, = r.groups["think"]
    arms = ARMS + (("self_copy",) if args.canary else ())
    temp = args.output.with_suffix(".partial.jsonl")
    if args.output.exists():
        raise FileExistsError(args.output)
    with temp.open("x") as out, r.torch.inference_mode():
        def emit(row):
            print(json.dumps(row, allow_nan=False), file=out, flush=True)

        emit({"kind": "manifest", "worker_sha256_information": sha(Path(__file__)),
              "panel_sha256": sha(args.panel), "split": args.split, "start": args.start,
              "count": args.count, "arms": arms, "draws": args.draws, "total_cap": args.cap,
              "marker": marker, **r.manifest})
        tick = time.monotonic()
        for draw in range(args.draws):
            for at in range(0, len(plan), 2):
                if time.time() >= args.deadline - 150:
                    raise TimeoutError("information depth shard reached reserved end window")
                source = [x["cells"][f"{v}_{m}"] for x in plan[at:at + 2] for v, m in CELLS]
                items = inputs_for(r.tokenizer, source)
                original = inputs_for(r.tokenizer, [
                    {"row": x["evidence_row"], "completion_prefix": []} for x in items])
                revised = inputs_for(r.tokenizer, list(map(new_row, items)))
                retained_width = max(len(x["native_ids"]) for x in items)
                evidence_width = max(len(x["native_ids"]) for x in (*original, *revised))
                width = max(retained_width, evidence_width)
                traces, audit = sample(r, items, draw, args.cap - 1, args.canary, zero_conv=True)
                for arm, group in ((EVIDENCE, original), (VERIFIED, revised)):
                    packed = pack_at(r, [x["native_ids"] for x in group], width)
                    traces[arm] = r.generate(group, "e0", draw, args.cap - 1, packed=packed)
                check_question_only(traces[DROP])
                emit({"kind": "audit", "draw": draw, "cache": audit, "retained_width": retained_width,
                      "evidence_width": width, "ids": list(dict.fromkeys(x["row"]["id"] for x in items))})
                input_by_arm = {EVIDENCE: original, VERIFIED: revised}
                for arm, results in traces.items():
                    for raw, item, input_row in zip(results, items, input_by_arm.get(arm, items), strict=True):
                        emit(raw | {"arm": arm, "variant": item["variant"],
                                    "source_prefix_sha256": packed_hash(item["completion_prefix"]),
                                    "input_sha256": packed_hash(input_row["native_ids"])})
        emit({"kind": "complete", "seconds": time.monotonic() - tick})
    temp.replace(args.output)


def score_window(row, trace, budget, tok):
    ids, eos = cut(trace, "total", budget)
    result = math_answer(row, tok.decode(ids, skip_special_tokens=False))
    good, found = result["numeric_correct"], result["answer_found"]
    return {"numeric_correct": good, "correct_eos": good and eos, "correct_without_eos": good and not eos,
            "wrong_eos": found and not good and eos, "missing_eos": not found and eos,
            "cap": not eos, "tokens": len(ids) + int(eos)}


def aggregate(cells, ids, arms, draws):
    totals, pairs, numeric_pairs = {}, {}, {}
    for arm in arms:
        rows = [r for (p, d, a, v), r in cells.items() if a == arm]
        def paired(metric):
            return {p: [all(cells[p, d, arm, v + "_" + m][metric] for v in ("v0", "v1"))
                        for d in draws for m in ("m0", "m1")] for p in ids}

        pairs[arm] = paired("correct_eos")
        numeric_pairs[arm] = paired("numeric_correct")
        totals[arm] = {k: sum(x[k] for x in rows) for k in METRICS}
        totals[arm].update(n=len(rows), both_values_correct=sum(sum(x) for x in pairs[arm].values()),
            both_values_numeric=sum(sum(x) for x in numeric_pairs[arm].values()),
            all_four_correct=sum(all(cells[p, d, arm, v + "_" + m]["correct_eos"] for v, m in CELLS)
                                 for p in ids for d in draws))
    targets = ((CONV, NATIVE), (ZERO_CONV, NATIVE), (ZERO_CONV, CONV), (VERIFIED, EVIDENCE), (BOTH, NATIVE))
    contrasts = {kind: {f"{a}-{b}": paired_interval({
        p: [int(u) - int(v) for u, v in zip(table[a][p], table[b][p], strict=True)] for p in ids})
                     for a, b in targets} for kind, table in (("correct_eos", pairs), ("numeric", numeric_pairs))}
    return {"arms": totals, "pairs": pairs, "paired": contrasts}


def report(path, fingerprint, panel, model):
    from transformers import AutoTokenizer
    if sha(path) != fingerprint:
        raise ValueError("information depth raw checksum mismatch")
    with path.open() as stream:
        h, *body, done = map(json.loads, stream)
    if (h["panel_sha256"] != sha(panel) or h["worker_sha256_information"] != sha(Path(__file__))
            or done["kind"] != "complete"):
        raise ValueError("information depth source differs")
    plan = cases(panel, h["split"], h["start"], h["count"])
    gold = {(r["id"], v): x for r in plan for v, x in r["cells"].items()}
    ids = {r["id"] for r in plan}
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    raw = {}
    for row in body:
        if row["kind"] == "audit":
            continue
        p, d, a, v = (row[k] for k in ("id", "draw", "arm", "variant"))
        key = p, d, a, v
        if (key in raw or not row["tokens"] or row["tokens"][0] != h["marker"]
                or len(tokens(row)) > h["total_cap"] or row["seed"] != seed_for(p, d)
                or row["post_seed"] != post_seed(p, d)
                or row["source_prefix_sha256"] != packed_hash(gold[p, v]["completion_prefix"])
                or (row["eos_token"] is not None and row["eos_token"] not in h["eos_ids"])):
            raise ValueError("invalid information depth trajectory")
        raw[key] = row
    draws = range(h["draws"])
    if raw.keys() != {(p, d, a, v + "_" + m) for p in ids for d in draws for a in h["arms"] for v, m in CELLS}:
        raise ValueError("incomplete information depth matrix")
    budgets = {}
    for budget in windows(h["total_cap"]):
        cells = {k: score_window(gold[k[0], k[3]]["row"], row, budget, tok) for k, row in raw.items()}
        budgets[str(budget)] = aggregate(cells, ids, h["arms"], draws)
    longest = budgets[str(h["total_cap"])]
    return {"header": h, "template_counts": dict(Counter(x["family"] for x in plan)),
            "arms": longest["arms"], "pairs": longest["pairs"], "budgets": budgets,
            "raw_sha256": fingerprint, "seconds": done["seconds"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("run", "score"))
    for name in ("panel", "model", "data", "checkpoint", "raw", "output"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--split")
    p.add_argument("--fingerprint")
    p.add_argument("--canary", action="store_true")
    for name, default in (("start", 0), ("count", 8), ("cap", 2048), ("draws", 2), ("deadline", None)):
        p.add_argument("--" + name, type=int, default=default)
    a = p.parse_args()
    if a.action == "run":
        if a.cap not in windows(a.cap):
            raise ValueError("expected a registered nested token budget")
        run(a)
    else:
        if a.output.exists():
            raise FileExistsError(a.output)
        save(a.output, report(a.raw, a.fingerprint, a.panel, a.model))


if __name__ == "__main__":
    main()
