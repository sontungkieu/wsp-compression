"""Freeze the cap-sensitivity subset and audit nested trajectories."""
import argparse
from collections import Counter
from decimal import Decimal
import json
from pathlib import Path

from repetition_baseline import BOXED, DECIMAL, sha
from repetition_budget_score import assess_window, cut
from repetition_mediation_queue import load, save
from repetition_scratchpad_score import DR, KR, read_matrix
from repetition_state_data import STRATA, keyed, length_summary
from repetition_state_value import THINK_END


def freeze(source, output):
    old = load(source)
    selected = []
    for group in STRATA:
        cells = sorted((r for r in old["cases"] if r["stratum"]==group),
                       key=lambda r:keyed(r["row"]["id"],"scratchpad-cap:2048"))
        selected += cells[:12]
    if len(selected)!=36 or output.exists():
        raise ValueError("expected a fresh cap agenda with 12 IDs per stratum")
    order = {r["row"]["id"]:i for i,r in enumerate(old["cases"])}
    selected.sort(key=lambda r:order[r["row"]["id"]])
    result = {"version":"scratchpad-cap-v1","arms":[KR,DR],"cap":2048,"cases":selected,
              "counts":dict(Counter(r["stratum"] for r in selected)),"length_summary":length_summary(selected),
              "parent_cohort_sha256":sha(source),
              "selection":"hash-ranked 12 per stratum, drawn from the earlier factorial cohort"}
    save(output,result)
    print(json.dumps({"n":len(selected),"cohort_sha256":sha(output),"strata":result["counts"],
                      "origins":dict(Counter(r["origin"] for r in selected))}))


def compare(old_root, new_root, output):
    _, old, _, _ = read_matrix(old_root)
    _, new, _, _ = read_matrix(new_root)
    arms = {}
    for a in (KR,DR):
        selected = [k for k in new if k[2]==a]
        equal = 0
        for k in selected:
            old_ids, old_eos = cut(old[k],"post",512)
            new_ids, new_eos = cut(new[k],"post",512)
            equal += (old_ids==new_ids and old_eos==new_eos and
                      (not old_eos or old[k]["eos_token"]==new[k]["eos_token"]))
        arms[a] = {"exact_512_token_trace":equal,"pairs":len(selected)}
    out = {"arms":arms,"interpretation":"cross-run structure check; cap effects use nested cuts in the new run"}
    if output.exists():
        raise FileExistsError(output)
    save(output,out)
    print(json.dumps(out))


def boxed_diagnostic(ids, eos, turn, tokenizer, gold):
    if THINK_END not in ids:
        return False, False, False
    after_close = len(ids)-ids[::-1].index(THINK_END)
    answer = ids[after_close:]
    if turn in answer:
        answer = answer[:answer.index(turn)]
    visible = tokenizer.decode(answer, skip_special_tokens=False)
    values = [m.group(1).strip() for m in BOXED.finditer(visible)]
    numeric = [Decimal(v.replace(",", "")) if DECIMAL.fullmatch(v) else None for v in values]
    correct = [v == gold for v in numeric]
    changed_to_wrong = False
    seen_correct = False
    for hit, value in zip(correct, numeric, strict=True):
        changed_to_wrong |= seen_correct and value is not None and not hit
        seen_correct |= hit
    return seen_correct, bool(correct and correct[-1] and not eos), changed_to_wrong


def audit_answers(root, output, tokenizer):
    cases, raw, receipts, draws = read_matrix(root)
    [turn] = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    windows = {}
    for budget in (256, 512, 1024, 1536, 2048):
        paired = Counter()
        paired_box = Counter()
        counts = {arm: Counter() for arm in (KR, DR)}
        for p, case in cases.items():
            row = case["row"]
            gold = Decimal(row["gold"])
            for draw in range(draws):
                states, boxes = [], []
                for arm in (KR, DR):
                    record = raw[p, draw, arm]
                    ids, eos = cut(record, "post", budget)
                    y = assess_window(row, record, "post", budget, tokenizer, turn)
                    ever, last_pending, changed = boxed_diagnostic(ids, eos, turn, tokenizer, gold)
                    totals = counts[arm]
                    totals["ever_correct_box"] += ever
                    totals["last_correct_box_without_eos"] += last_pending
                    totals["correct_box_later_wrong_box"] += changed
                    totals["correct_without_eos"] += y["numeric_correct"] and not eos
                    outcome = ("eos_correct" if y["eos_correct"] else "eos_wrong" if y["wrong_parsed_eos"]
                               else "eos_missing" if eos else "no_eos_correct" if y["numeric_correct"] else "no_eos")
                    states.append(outcome)
                    boxes.append("correct_box" if ever else "no_correct_box")
                paired[" -> ".join(states)] += 1
                paired_box[" -> ".join(boxes)] += 1
        windows[str(budget)] = {"arm_counts": counts, "keep_to_drop_outcome": paired,
                                "keep_to_drop_ever_boxed_correct": paired_box}
    if output.exists():
        raise FileExistsError(output)
    save(output, {"draws": draws, "prompts": len(cases), "verified_shards": receipts, "windows": windows})
    print(json.dumps({"report_sha256": sha(output), "prompts": len(cases), "draws": draws}))


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("freeze","compare","answer-audit"))
    for name in ("source","root","output","model"):
        p.add_argument("--"+name,type=Path)
    a = p.parse_args()
    if a.action=="freeze":
        freeze(a.source,a.output)
    elif a.action=="compare":
        compare(a.source,a.root,a.output)
    else:
        from transformers import AutoTokenizer
        audit_answers(a.root,a.output,AutoTokenizer.from_pretrained(a.model,local_files_only=True))
