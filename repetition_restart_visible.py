"""Measure the assistant turn boundary separately from the configured EOS."""
import argparse
from pathlib import Path

from repetition_baseline import math_answer
from repetition_canonical import paired_interval
from repetition_mediation_queue import save
from repetition_restart import ARMS
from repetition_restart_score import CONTRASTS, read_run


def score_visible(case, row, tokenizer, turn):
    ids = row["tokens"]
    at = ids.index(turn) if turn in ids else None
    eos = row["eos_token"] is not None
    answer = tokenizer.decode(ids[:at], skip_special_tokens=False)
    parsed = math_answer(case, answer, pending_thought=row["arm"] != "no_think")
    return {"turn_marker":at is not None, "configured_eos":eos, "terminated":eos or at is not None,
            "turn_without_eos":at is not None and not eos,
            "correct_terminated":parsed["numeric_correct"] and (eos or at is not None),
            "correct_turn_without_eos":parsed["numeric_correct"] and at is not None and not eos,
            "visible_exact_boxed":parsed["exact_format"]}


def summarize(root, phase, tokenizer):
    plan,cases,raw,_,verified,draws = read_run(root,phase)
    [turn] = tokenizer.encode("<|im_end|>",add_special_tokens=False)
    assessed = {k:score_visible(cases[k[0]],row,tokenizer,turn) for k,row in raw.items()}
    table = {}
    for arm in ARMS:
        rows = [r for k,r in assessed.items() if k[2] == arm]
        table[arm] = {"n":len(rows),**{name:sum(r[name] for r in rows) for name in rows[0]}}
    effects = {f"{a}_minus_{b}":paired_interval({
        p:[int(assessed[p,d,a]["correct_terminated"])-int(assessed[p,d,b]["correct_terminated"])
           for d in range(draws)] for p in cases
    }) for a,b in CONTRASTS}
    return {"phase":phase, "budget":plan["budget"], "turn_marker_id":turn, "draws":draws,
            "verified_shards":verified, "arms":table, "paired_correct_terminated":effects}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("pilot","eval"), required=True)
    for name in ("root","model","output"):
        p.add_argument("--"+name, type=Path, required=True)
    a = p.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    if a.output.exists():
        raise FileExistsError(a.output)
    save(a.output,summarize(a.root,a.phase,tokenizer))
