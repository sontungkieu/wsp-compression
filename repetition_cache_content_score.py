"""Measure paired sensitivity to the value available only in the scratchpad."""
import argparse
from decimal import Decimal
from pathlib import Path

from repetition_baseline import sha
from repetition_cache_factorial import DROP, NATIVE
from repetition_cache_retention import CONTENTS, TASKS, read_receipts
from repetition_canonical import paired_interval
from repetition_mediation_queue import load, save


def compare_values(place, results):
    expected = [place["variants"][c]["row"]["gold"] for c in CONTENTS]
    numeric = place["task"]=="math"

    def matches(record,value):
        pred = record["value"]
        if not record["eos"] or pred is None:
            return False
        return Decimal(pred)==Decimal(value) if numeric else pred==value

    own = [matches(row,y) for row,y in zip(results,expected,strict=True)]
    opposite = [matches(row,y) for row,y in zip(results,reversed(expected),strict=True)]
    if any(a and b for a,b in zip(own,opposite,strict=True)):
        raise ValueError("the two intervention values must have different targets")
    return {"own_correct_eos":sum(own),"opposite_correct_eos":sum(opposite),
            "sensitivity":(sum(own)-sum(opposite))/len(CONTENTS)}


def summarize(root,tokenizer):
    manifests = sorted((root/"queue").glob("draw*-done.json"))
    paths = [pair for file in manifests for pair in load(file)["outputs"]]
    places,cells,arms,draws,proof,_ = read_receipts(root,paths,"eval",tokenizer)
    effects = {(p,d,a):compare_values(places[p],[cells[p,d,a,c] for c in CONTENTS])
               for p in places for d in range(draws) for a in arms}
    if any(r["sensitivity"]!=0 for k,r in effects.items() if k[2]==DROP):
        raise ValueError("identical question-only pairs must have zero content sensitivity")
    results = {}
    for task in TASKS:
        pids = [p for p,v in places.items() if v["task"]==task]
        arms_out = {}
        for arm in arms:
            relevant = [effects[p,d,arm] for p in pids for d in range(draws)]
            absolute = {p:[effects[p,d,arm]["sensitivity"] for d in range(draws)] for p in pids}
            difference = {p:[effects[p,d,arm]["sensitivity"]-effects[p,d,NATIVE]["sensitivity"]
                             for d in range(draws)] for p in pids}
            arms_out[arm] = {"pairs":len(relevant),"correct_eos_own_value":sum(r["own_correct_eos"] for r in relevant),
                             "correct_eos_other_value":sum(r["opposite_correct_eos"] for r in relevant),
                             "content_sensitivity":paired_interval(absolute),"sensitivity_minus_raw":paired_interval(difference)}
        results[task] = {"prompts":len(pids),"arms":arms_out}
    return {"tasks":results,"draws":draws,"verified_shards":proof}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("root","model","output"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    from transformers import AutoTokenizer
    result=summarize(a.root,AutoTokenizer.from_pretrained(a.model,local_files_only=True))
    if a.output.exists():
        raise FileExistsError(a.output)
    save(a.output,result)
    print(sha(a.output))
