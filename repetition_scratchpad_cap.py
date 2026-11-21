"""Freeze the cap-sensitivity subset and audit nested trajectories."""
import argparse
from collections import Counter
import json
from pathlib import Path

from repetition_baseline import sha
from repetition_budget_score import cut
from repetition_mediation_queue import load, save
from repetition_scratchpad_score import DR, KR, read_matrix
from repetition_state_data import STRATA, keyed, length_summary


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


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("freeze","compare"))
    for name in ("source","root","output"):
        p.add_argument("--"+name,type=Path)
    a = p.parse_args()
    if a.action=="freeze":
        freeze(a.source,a.output)
    else:
        compare(a.source,a.root,a.output)
