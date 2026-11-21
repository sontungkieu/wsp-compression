"""Measure exact-token repetition before phase changes and at the online switch landmark."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from repetition_baseline import first_period, first_recurrent, sha
from repetition_budget_score import cut
from repetition_canonical import paired_interval
from repetition_mediation_queue import load, save
from repetition_replay import terminal_cycle
from repetition_restart import reaches_switch, score_policy
from repetition_restart_score import read_run


def first_long_cycle(ids):
    """Return the first token count that satisfies the terminal detector's run length."""
    earliest = None
    for period in range(1,min(128,len(ids)//5)+1):
        needed = max(128,5*period)-period
        consecutive = 0
        for j in range(period,len(ids) if earliest is None else earliest-1):
            consecutive = consecutive+1 if ids[j]==ids[j-period] else 0
            if consecutive>=needed:
                earliest = j+1
                break
    return earliest


def repeated_ngram_fraction(ids, n):
    grams = [tuple(ids[i:i+n]) for i in range(len(ids)-n+1)]
    return (len(grams)-len(set(grams)))/len(grams)


def before_marker(raw, marker, budget):
    ids,eos = cut(raw,"total",budget)
    if marker not in ids:
        return ids,False,eos
    return ids[:ids.index(marker)],True,eos


def measurements(thought):
    exact_terminal = terminal_cycle(thought) is not None
    exact_anywhere = exact_terminal or first_long_cycle(thought) is not None
    features = {"thought_length":len(thought),"exact_cycle_anywhere":exact_anywhere,"exact_cycle_terminal":exact_terminal,
                "v3_periodic":first_period(thought) is not None,"v3_recurrent16":first_recurrent(thought) is not None}
    if len(thought)>=256:
        for n in (8,16):
            features[f"early_{n}"] = repeated_ngram_fraction(thought[:128],n)
            features[f"late_{n}"] = repeated_ngram_fraction(thought[-128:],n)
    return features


def aggregate(items):
    if not items:
        return {"n":0}
    cols = {key:sum(r[key] for _,r in items) for key in
            ("exact_cycle_anywhere","exact_cycle_terminal","v3_periodic","v3_recurrent16")}
    result = {"n":len(items),"thought_token_median":statistics.median(r["thought_length"] for _,r in items),**cols}
    comparable = [r for _,r in items if "early_8" in r]
    result["nonoverlapping_windows"]=len(comparable)
    for n in (8,16):
        a,b = f"early_{n}",f"late_{n}"
        result[f"repeated_{n}gram_fraction"] = {
            "early_mean":statistics.fmean(r[a] for r in comparable) if comparable else None,
            "late_mean":statistics.fmean(r[b] for r in comparable) if comparable else None,
            "paired_mean_change":statistics.fmean(r[b]-r[a] for r in comparable) if comparable else None,
        }
    result["example_prompt_draw_ids"] = {
        "terminal":[k for k,r in items if r["exact_cycle_terminal"]][:4],
        "nonterminal_cycle":[k for k,r in items if r["exact_cycle_anywhere"] and not r["exact_cycle_terminal"]][:4],
        "recurrent16_only":[k for k,r in items if r["v3_recurrent16"] and not r["exact_cycle_anywhere"]][:4],
        "neither":[k for k,r in items if not r["v3_recurrent16"] and not r["exact_cycle_anywhere"]][:4],
    }
    return result


def phase_windows(native, marker, budgets):
    common_open = {key for key,row in native.items() if not before_marker(row,marker,budgets[-1])[1]}
    result = {}
    for budget in budgets:
        strata = defaultdict(list)
        for key,row in sorted(native.items()):
            thought,closed,eos = before_marker(row,marker,budget)
            data = measurements(thought)
            strata["closed" if closed else "open"].append((key,data))
            if not closed:
                strata["open_eos" if eos else "open_unfinished"].append((key,data))
            if key in common_open and len(budgets)>1:
                strata["open_at_both_budgets"].append((key,data))
        result[str(budget)] = {name:aggregate(strata[name]) for name in
                              ("closed","open","open_eos","open_unfinished","open_at_both_budgets")
                              if name!="open_at_both_budgets" or len(budgets)>1}
    return result


def switch_groups(raw,cases,plan,tokenizer,marker,budgets):
    switch = plan["switch"]
    [turn] = tokenizer.encode("<|im_end|>",add_special_tokens=False)
    eligible = [(p,d) for (p,d,a),r in raw.items() if a=="native" and reaches_switch(r,switch,marker)]
    outcomes = {}
    for budget in budgets:
        groups = defaultdict(list)
        for p,d in sorted(eligible):
            trunk = raw[p,d,"native"]["tokens"][:switch]
            tags = {
                "v3_recurrent16_by_switch":first_recurrent(trunk) is not None,
                "v3_periodic_by_switch":first_period(trunk) is not None,
            }
            scores = []
            for arm in ("switch_keep","switch_restart"):
                row = raw[p,d,arm]
                ids,eos = cut(row,"total",budget)
                result = row | {"tokens":ids,"eos_token":row["eos_token"] if eos else None}
                scores.append(score_policy(cases[p],result,tokenizer,turn)["correct_eos"])
            groups["all"].append((p,d,*scores))
            for name in ("v3_recurrent16_by_switch","v3_periodic_by_switch"):
                suffix = "detected" if tags[name] else "undetected"
                groups[name+":"+suffix].append((p,d,*scores))
        outcomes[str(budget)] = {}
        for name,rows in groups.items():
            by_prompt = defaultdict(list)
            for p,_,keep,drop in rows:
                by_prompt[p].append(int(drop)-int(keep))
            outcomes[str(budget)][name] = {
                "pairs":len(rows),"prompts":len(by_prompt),"keep_correct_eos":sum(a for _,_,a,_ in rows),
                "restart_correct_eos":sum(b for _,_,_,b in rows),
                "restart_minus_keep":paired_interval(by_prompt),
            }
    return outcomes


def analyze(root, kind, tokenizer):
    [marker] = tokenizer.encode("</think>",add_special_tokens=False)
    if kind=="restart":
        plan,cases,raw,_,receipts,_ = read_run(root,"eval")
        budgets = sorted({plan["budget"],*plan.get("nested_total_budgets",[])})
        native = {(p,d):r for (p,d,a),r in raw.items() if a=="native"}
        switch = switch_groups(raw,cases,plan,tokenizer,marker,budgets)
    else:
        summary = load(root/"summary.json")
        budgets = sorted(map(int,summary["windows"]))
        receipts = summary["verified_shards"]
        native = {}
        for name,digest in receipts:
            path = Path(name)
            if sha(path)!=digest:
                raise ValueError("depth trace checksum mismatch")
            with path.open() as f:
                for line in f:
                    row = json.loads(line)
                    if row["kind"]=="rollout" and row["arm"]=="native":
                        key = row["id"],row["draw"]
                        if key in native:
                            raise ValueError("duplicate depth trace")
                        native[key] = row
        if len(native)!=summary["n_prompts"]*summary["draws"]:
            raise ValueError("missing depth traces")
        switch = {}
    return {"thinking_marker_id":marker,"windows":phase_windows(native,marker,budgets),
            "switch_group_outcomes":switch,"source_shards":receipts,
            "ngram_rate":"within-window duplicate fraction in non-overlapping first and last 128-token windows"}


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind",choices=("restart","depth"),required=True)
    for name in ("root","model","output"):
        p.add_argument("--"+name,type=Path,required=True)
    a = p.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    if a.output.exists():
        raise FileExistsError(a.output)
    save(a.output,analyze(a.root,a.kind,tokenizer))
    print(sha(a.output))
