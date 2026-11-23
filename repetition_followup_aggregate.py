"""Aggregate receipt-verified WSP results and keep a durable campaign report."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
import time

from repetition_baseline import sha
from repetition_canonical import paired_interval
from repetition_mediation_queue import load, save

OUTCOMES = ("correct_eos", "wrong_eos", "missing_eos", "cap", "eos",
            "generated_close", "closed", "terminal_exact", "first_exact_before_close", "recurrent16")
REPLAY = ("native", "conv_only", "no_think", "native_suffix")


def tally(rows):
    rows=list(rows)
    total={key:sum(bool(x.get(key)) for x in rows) for key in OUTCOMES if any(key in x for x in rows)}
    total["n"]=len(rows)
    total["answers"]=total.get("correct_eos",0)+total.get("wrong_eos",0)
    total["unanswered"]=len(rows)-total["answers"]
    total["risk_given_answer"]=total.get("wrong_eos",0)/total["answers"] if total["answers"] else None
    costs=[x.get("cost_from_start",x.get("cost",x.get("tokens"))) for x in rows]
    total["tokens_mean"]=sum(costs)/len(rows) if rows and all(x is not None for x in costs) else None
    return total


def contrast(x,y,key="correct_eos"):
    ids=x.keys()&y.keys()
    if not ids:
        return None
    cluster={p:[int(a[key])-int(b[key]) for a,b in zip(x[p],y[p],strict=True)] for p in sorted(ids)}
    discordant=Counter(difference for values in cluster.values() for difference in values)
    return paired_interval(cluster) | {"wins":discordant[1],"losses":discordant[-1],"ties":discordant[0]}


def reject_rate(calibration,acceptance,label):
    results={}
    for p,outcome in calibration.items():
        u=int.from_bytes(sha256(f"wsp-risk-v1:{label}:{p}".encode()).digest()[:8])
        accepted=outcome["answered"] and u/2**64 < acceptance
        results[p]=outcome | {"answered":accepted,"correct_eos":accepted and outcome["correct_eos"],
                              "wrong_eos":accepted and outcome["wrong_eos"]}
    return results


def risk_match(dev_policy,dev_direct):
    """Choose hash-thinned direct or agreement using development outcomes only."""
    goal=tally(dev_policy.values())
    if not goal["n"]:
        return None
    target_coverage=goal["answers"]/goal["n"]
    target_wrong=goal.get("wrong_eos",0)/goal["n"]
    grid=[]
    for scheme, data in dev_direct.items():
        if data.keys()!=dev_policy.keys():
            continue
        for step in range(21):
            keep=step/20
            score=tally(reject_rate(data,keep,scheme).values())
            error=abs(score["answers"]/score["n"]-target_coverage)+abs(score.get("wrong_eos",0)/score["n"]-target_wrong)
            grid.append((error,-score.get("correct_eos",0),scheme,keep,score))
    if not grid:
        return None
    mismatch,_,scheme,keep,score=min(grid,key=lambda entry:entry[:4])
    return dict(scheme=scheme,keep=keep,development=score,target=goal,development_mismatch=mismatch)


def submitted_scores(root,status):
    scores={}
    for name, mark in status["jobs"].items():
        if not name.endswith("-score") or mark["stage"]!="completed":
            continue
        p=root/"jobs"/(name+".json")
        if str(p)!=mark["output"] or sha(p)!=mark["sha256"]:
            raise ValueError(f"scorer receipt does not match bytes for {name}")
        score=load(p)
        parent="F-canary-math500" if name=="F2-canary-math500-score" else name.removesuffix("-score")
        fingerprint=status["jobs"][parent]["sha256"]
        if score["raw_sha256"]!=fingerprint:
            raise ValueError(f"raw receipt mismatch for {name}")
        scores[name]=score
    return scores


def joint_policy(baseline,replay,anchor,window=2048):
    result={}
    for split in ("dev","eval"):
        base=baseline.get(split,{})
        blocks=replay.get((split,anchor),[])
        if not base or not blocks:
            continue
        covered=set()
        active={}
        for score in blocks:
            h=score["header"]
            source=Path(h["source"]).name.removesuffix(".jsonl")
            if source not in base:
                continue
            ids=base[source]["native"].keys()
            if len(ids)!=score["eligible"]+score["excluded"] or not score["items"].keys()<=ids:
                raise ValueError("timing and baseline shard prompt counts disagree")
            if covered&ids:
                raise ValueError("duplicate policy evaluation prompt")
            covered.update(ids)
            active.update(score["items"])
        if not covered:
            continue
        refs={arm:{p:x for shard in base.values() for p,x in shard.get(arm,{}).items() if p in covered}
              for arm in ("native","no_think","direct_agreement")}
        if not refs["direct_agreement"]:
            del refs["direct_agreement"]
        coarse={arm:{p:[outcome] for p,outcome in rows.items()} for arm,rows in refs.items()}
        interventions=(*REPLAY[:3],"zero_conv") if active and all(
            "zero_conv" in y["arms"][str(window)] for y in active.values()) else REPLAY[:3]
        for arm in interventions:
            coarse["trigger_"+arm]={p:[active[p]["arms"][str(window)][arm][0]] if p in active else [n]
                                    for p,n in refs["native"].items()}
        oracle={}
        for p,native in refs["native"].items():
            options=[native] if p not in active else [
                active[p]["arms"][str(window)][arm][0] for arm in (*interventions,"native_suffix")]
            oracle[p]=max(options,key=lambda x:(x["correct_eos"],-int(x["wrong_eos"]),-x.get("cost_from_start",0)))
        result[split]={"covered":len(covered),"triggers":len(active),"cells":coarse,"oracle":tally(oracle.values())}
    if "eval" not in result:
        return {}
    evaluation=result["eval"]
    cells=evaluation["cells"]
    means={k:tally(x[0] for x in v.values()) for k,v in cells.items()}
    out={"n":evaluation["covered"],"coverage_count":evaluation["triggers"],"arms":means,"oracle":evaluation["oracle"],
         "conv_minus_keep":contrast(cells["trigger_conv_only"],cells["trigger_native"]),
         "conv_minus_direct_from_start":contrast(cells["trigger_conv_only"],cells["no_think"]),
         "estimand":"replay-simulated draw0 policy, counted from the original question"}
    if "trigger_zero_conv" in cells:
        out["zero_minus_keep"]=contrast(cells["trigger_zero_conv"],cells["trigger_native"])
    if "dev" in result:
        dev=result["dev"]["cells"]
        control={k:{p:v[0] for p,v in dev[k].items()} for k in ("no_think","direct_agreement") if k in dev}
        selection=risk_match({p:y[0] for p,y in dev["trigger_conv_only"].items()},control)
        if selection:
            branch=selection["scheme"]
            evaluated=reject_rate({p:v[0] for p,v in cells[branch].items()},selection["keep"],branch)
            out["dev_locked_thinned_direct"]=selection | {"eval":tally(evaluated.values()),
                "conv_minus_thinned_direct":contrast(cells["trigger_conv_only"],{p:[y] for p,y in evaluated.items()})}
    return out


def replay_results(scores):
    by_split=defaultdict(list)
    for data in scores:
        h=data["header"]
        by_split[(h["dataset"],h["split"],h["anchor"],h["cap"],tuple(h["arms"]))].append(data)
    out={}
    for (dataset,split,anchor,cap,observed_arms),records in sorted(by_split.items()):
        cells={}
        for record in records:
            if cells.keys()&record["items"].keys():
                raise ValueError("timing cohort overlaps its own shards")
            cells.update(record["items"])
        stats={}
        for budget in records[0]["counts"]:
            arms={arm:{p:v["arms"][budget][arm] for p,v in cells.items()} for arm in (*observed_arms,"native_suffix")}
            stats[budget]={"arms":{a:tally(y for v in group.values() for y in v) for a,group in arms.items()},
                           "conv_minus_keep":contrast(arms["conv_only"],arms["native"]),
                           "conv_minus_fresh":contrast(arms["conv_only"],arms["no_think"])}
            if "zero_conv" in arms:
                stats[budget]["zero_minus_keep"]=contrast(arms["zero_conv"],arms["native"])
        out[f"{dataset}/{split}/{anchor}/{cap}"]={"eligible":len(cells),
                "excluded":sum(d["excluded"] for d in records),"budgets":stats}
    return out


def information_results(scores):
    results={}
    models=defaultdict(list)
    for data in scores:
        h=data["header"]
        label=h.get("model_type","qwen3_5") + ":" + h.get("checkpoint_revision","base")
        models[label,h["panel_sha256"],h["split"]].append(data)
    for key,items in sorted(models.items()):
        pair=defaultdict(dict)
        totals=defaultdict(Counter)
        families=Counter()
        for item in items:
            families.update(item["template_counts"])
            for arm, stats in item["arms"].items():
                totals[arm].update(stats)
                current=item["pairs"][arm]
                if pair[arm].keys()&current.keys():
                    raise ValueError("copy/information cohort overlaps its own shards")
                pair[arm].update(current)
        reference={p:[{"ok":value} for value in v] for p,v in pair["native"].items()}
        results["/".join(key)]={"arms":totals,"template_counts":families,"pairs":len(reference),
            "both_values_vs_keep":{arm:contrast({p:[{"ok":y} for y in v] for p,v in rows.items()},reference,"ok")
                                   for arm,rows in pair.items() if arm!="native"}}
    return results


def astra_results(scores):
    grouped=defaultdict(list)
    copies={}
    for data in scores:
        if copies.keys()&data["copy"].keys():
            raise ValueError("duplicate Astra copy probe")
        copies.update(data["copy"])
        for cap,modes in data["windows"].items():
            for mode, cells in modes.items():
                grouped[cap,mode].append(cells)
    windows={}
    for (cap,mode), pieces in sorted(grouped.items()):
        all_cells={}
        for piece in pieces:
            if all_cells.keys()&piece.keys():
                raise ValueError("duplicate Astra math prompt")
            all_cells.update(piece)
        arms={v:{p:y[v] for p,y in all_cells.items()} for v in ("base","astra","direct_baseline")}
        windows[f"{cap}/{mode}"]={"arms":{a:tally(x for v in y.values() for x in v) for a,y in arms.items()},
                                  "astra_minus_base":contrast(arms["astra"],arms["base"])}
    return {"windows":windows,"copy":{v:sum(x[v] for x in copies.values()) for v in ("base","astra")},
            "copy_n":len(copies),"copy_discordant":sum(x["astra"]!=x["base"] for x in copies.values())}


def summarize(root,status):
    sections={x:[] for x in ("baseline","replay","information","astra")}
    base_by_split=defaultdict(dict)
    replay_by_split=defaultdict(lambda:defaultdict(list))
    scores=submitted_scores(root,status)
    for name,item in scores.items():
        if "canary" in name:
            continue
        h=item["header"]
        if "worker_sha256_timing" in h:
            sections["replay"].append(item)
            if h["cap"]==2048:
                replay_by_split[h["dataset"]][h["split"],h["anchor"]].append(item)
        elif "worker_sha256_astra" in h:
            sections["astra"].append(item)
        elif "worker_sha256_information" in h:
            sections["information"].append(item)
        else:
            sections["baseline"].append(item)
            if (name.startswith(("A-math500-","N-qwen35-08b-amc_val-"))
                    and h["scheme"][0][1:]==[0,16384] and "16384" in item["windows"]):
                splits=base_by_split.setdefault(h["dataset"],defaultdict(dict))
                splits[h["split"]][name.removesuffix("-score")]=item["windows"]["16384"]["cells"]
    totals=defaultdict(Counter)
    for item in sections["baseline"]:
        h=item["header"]
        for cap,data in item["windows"].items():
            for arm, stats in data["arms"].items():
                key=f"{h['dataset']}/{h['model_label']}/{h['split']}/draw{h['scheme'][0][1]}/traincap{h['scheme'][0][2]}/{cap}/{arm}"
                totals[key].update(stats)
    for stats in totals.values():
        stats["unanswered"]=stats["n"]-stats["answered"]
        stats["tokens_mean"]=stats["cost"]/stats["n"]
    release=root/"release.json"
    released=load(release) if release.is_file() else None
    if released and released["status_sha256"]!=sha(root/"status.json"):
        raise ValueError("release receipt has a different status SHA")
    return {"state":status["state"],"host":status["host"],"deadline":status["deadline"],"updated":status["updated"],
            "release_sha256":sha(release) if released else None,"release":released,
            "job_counts":dict(Counter(x["stage"] for x in status["jobs"].values())),
            "score_receipts":{k:status["jobs"][k]["sha256"] for k in scores},
            "baseline":totals,"timing":replay_results(sections["replay"]),
            "information":information_results(sections["information"]),"astra":astra_results(sections["astra"]),
            "policies":{dataset:{anchor:joint_policy(splits,replay_by_split[dataset],anchor)
                                  for anchor in ("512","2048","exact")} for dataset,splits in base_by_split.items()}}


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root",type=Path,required=True)
    cli.add_argument("--output",type=Path,required=True)
    cli.add_argument("--watch",action="store_true")
    args=cli.parse_args()
    latest=None
    while True:
        status=load(args.root/"status.json")
        terminal=(args.root/"release.json").is_file()
        signature=tuple((k,v["stage"]) for k,v in status["jobs"].items()),terminal
        if signature!=latest:
            report=summarize(args.root,status)
            path=args.output if terminal or not args.watch else args.output.with_suffix(".partial.json")
            save(path,report)
            latest=signature
        if terminal or not args.watch or time.time()>status["deadline"]+240:
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
