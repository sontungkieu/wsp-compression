"""Audit factorial retention and replay timing on a shared prompt cohort."""
import argparse
from collections import defaultdict
from pathlib import Path

from repetition_baseline import sha
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
    save(args.output,output)


if __name__=="__main__":
    main()
