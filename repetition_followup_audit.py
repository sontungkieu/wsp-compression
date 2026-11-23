"""Audit factorial retention and replay timing on a shared prompt cohort."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from repetition_baseline import math_answer, sha
from repetition_canonical import paired_interval
from repetition_followup_aggregate import contrast, submitted_scores, tally
from repetition_mediation_queue import load, save

FACTOR_ARMS=("native","conv_only","matrix_only","both")


def information_factorial(items):
    groups=defaultdict(dict)
    for blob in items:
        keys=blob["pairs"]["native"].keys()
        if not all(blob["pairs"][arm].keys()==keys for arm in FACTOR_ARMS):
            raise ValueError("factorial information arms have different prompts")
        for p in keys:
            _,family,_=p.split(":")
            values={arm:blob["pairs"][arm][p] for arm in FACTOR_ARMS}
            if p in groups["pooled"]:
                raise ValueError("factorial audit encountered a duplicate prompt")
            groups["pooled"][p]=values
            groups[family][p]=values
    result={}
    for family, data in groups.items():
        estimates={}
        for p, y in data.items():
            estimates[p]=[int(both)-int(conv)-int(matrix)+int(base)
                          for base,conv,matrix,both in zip(*(y[a] for a in FACTOR_ARMS),strict=True)]
        result[family]={"paired_id_interaction":paired_interval(estimates),
                        "trials":sum(map(len,estimates.values())),
                        "both_values_eos":{a:sum(sum(y[a]) for y in data.values()) for a in FACTOR_ARMS}}
    return result


def information_rows(root,status,scores,panel,model):
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
    gold={(item["id"],variant):cell["row"] for item in load(panel)["panels"]["eval"]
          for variant,cell in item["cells"].items()}
    rows=defaultdict(dict)
    raw_shas={}
    for scorer, result in scores.items():
        parent=scorer.removesuffix("-score")
        raw=root/"jobs"/(parent+".jsonl")
        digest=status["jobs"][parent]["sha256"]
        if sha(raw)!=digest or result["raw_sha256"]!=digest:
            raise ValueError("C2 raw file differs from its scored receipt")
        raw_shas[parent]=digest
        with raw.open() as stream:
            head,*body,end=map(json.loads,stream)
        if head["panel_sha256"]!=sha(panel) or end["kind"]!="complete" or head["split"]!="eval":
            raise ValueError("C2 row audit does not match the frozen evaluation panel")
        for trace in body:
            if trace["kind"]!="rollout":
                continue
            p,d,v,a=(trace[k] for k in ("id","draw","variant","arm"))
            key=p,d,v
            if a in rows[key]:
                raise ValueError("duplicate C2 rollout in row audit")
            text=tok.decode(trace["tokens"],skip_special_tokens=False)
            score=math_answer(gold[p,v],text)
            correct,found=score["numeric_correct"],score["answer_found"]
            eos=trace["eos_token"] is not None
            rows[key][a]={"correct_eos":correct and eos,"wrong_eos":found and not correct and eos,
                          "missing_eos":not found and eos,"correct_without_eos":correct and not eos,"cap":not eos}
    totals=defaultdict(lambda:defaultdict(Counter))
    pairs=defaultdict(lambda:defaultdict(list))
    for (prompt,_,_),arms in rows.items():
        _,family,_=prompt.split(":")
        if not set(FACTOR_ARMS)<=arms.keys():
            raise ValueError("C2 row is missing a factorial arm")
        base,conv,matrix,both=(int(arms[arm]["correct_eos"]) for arm in FACTOR_ARMS)
        for group in ("pooled",family):
            pairs[group][prompt].append(both-conv-matrix+base)
            for arm,outcome in arms.items():
                totals[group][arm].update(outcome | {"n":1})
    return {"raw_sha256":raw_shas,"groups":{
        g:{"arms":totals[g],"correct_eos_interaction":paired_interval(pair)} for g,pair in pairs.items()}}


def replay_intersection(items):
    groups=defaultdict(dict)
    for blob in items:
        h=blob["header"]
        if h["dataset"]!="amc_val" or h["split"]!="eval":
            continue
        anchor=h["anchor"]
        data=blob["items"]
        if groups[anchor].keys()&data.keys():
            raise ValueError("replay intersection audit encountered duplicate prompts")
        groups[anchor].update(data)
    early,late=groups["512"],groups["exact"]
    keys=early.keys()&late.keys()
    outcomes={"eligible_512":len(early),"eligible_exact":len(late),"intersection":len(keys),"cuts":{}}
    if not keys:
        return outcomes
    for window in ("1024","2048"):
        e={arm:{p:early[p]["arms"][window][arm] for p in keys} for arm in early[min(keys)]["arms"][window]}
        l={arm:{p:late[p]["arms"][window][arm] for p in keys} for arm in late[min(keys)]["arms"][window]}
        if e.keys()!=l.keys():
            raise ValueError("anchor comparison is missing a replay arm")
        side=lambda data:{a:tally(ob for runs in v.values() for ob in runs) for a,v in data.items()}
        outcomes["cuts"][window]={"early":side(e),"late":side(l),
            "late_minus_early":{a:contrast(l[a],e[a]) for a in e},
            "zero_minus_conv_early":contrast(e["zero_conv"],e["conv_only"]),
            "zero_minus_conv_late":contrast(l["zero_conv"],l["conv_only"])}
    return outcomes


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mode",choices=("information","amc"))
    cli.add_argument("--root",type=Path,required=True)
    cli.add_argument("--output",type=Path,required=True)
    cli.add_argument("--panel",type=Path)
    cli.add_argument("--model",type=Path)
    args=cli.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    expected, compute=("C2-eval-",information_factorial) if args.mode=="information" else ("NTR-",replay_intersection)
    ref=load(args.root/"status.json")
    scores={k:d for k,d in submitted_scores(args.root,ref).items()
            if k.startswith(expected) and "canary" not in k}
    n_required=4 if args.mode=="information" else 22
    if len(scores)!=n_required:
        raise ValueError("audit requires every predeclared shard")
    output={"mode":args.mode,"worker_sha256":sha(Path(__file__)),
            "score_receipts":{k:ref["jobs"][k]["sha256"] for k in scores},
            "results":compute(scores.values())}
    if args.mode=="information" and args.panel and args.model:
        output["row_audit"]=information_rows(args.root,ref,scores,args.panel,args.model)
    save(args.output,output)


if __name__=="__main__":
    main()
