"""Verify from-question policies and report full-panel paired comparisons."""
import argparse
import json
from pathlib import Path
import statistics

from repetition_baseline import sha
from repetition_budget_run import tokens
from repetition_budget_score import cut
from repetition_canonical import paired_interval, seed_for
from repetition_mediation_queue import load, save
from repetition_restart import ARMS, nested_restart, reaches_switch, score_policy
from repetition_state_value import post_seed

CONTRASTS = (("switch_restart","switch_keep"),("switch_restart","question_e0"),
             ("switch_restart","no_think"),("switch_keep","native"),("question_e0","no_think"),
             ("no_think","native"),("switch_restart","native"))


def restart_decomposition(outcomes, prompts, draws, switched):
    early, lost = {}, {}
    for p in prompts:
        early[p], lost[p] = [], []
        for d in range(draws):
            native, direct, restart = (int(outcomes[p,d,a]["correct_eos"])
                                       for a in ("native", "question_e0", "switch_restart"))
            if (p,d) in switched:
                early[p].append(0)
                lost[p].append(direct-restart)
                if restart > direct:
                    raise ValueError("a nested direct cut cannot gain correct EOS by shortening")
            else:
                if restart != native:
                    raise ValueError("restart failed to keep the early native outcome")
                early[p].append(native-direct)
                lost[p].append(0)
    return {"pairs":len(prompts)*draws, "switched_pairs":len(switched),
            "early_native_gain_pairs":sum(v > 0 for row in early.values() for v in row),
            "early_native_loss_pairs":sum(v < 0 for row in early.values() for v in row),
            "switched_budget_loss_pairs":sum(map(sum,lost.values())),
            "early_native_net":paired_interval(early),
            "switched_budget_cost":paired_interval(lost)}


def read_run(root, phase):
    cohort = root/"cohort.json"
    plan = load(cohort)
    cases = {r["row"]["id"]:r["row"] for r in plan[phase]}
    budget, switch = plan["budget"],plan["switch"]
    receipts = sorted((root/"queue").glob(phase+"*-done.json"))
    if not receipts:
        raise ValueError("no completed policy wave")
    raw, batches, verified = {},[],[]
    for draw,receipt in enumerate(receipts):
        if receipt.name != f"{phase}{draw}-done.json":
            raise ValueError("policy draw receipts are noncontiguous")
        parts,covered = set(),set()
        for name,digest in load(receipt)["outputs"]:
            path=Path(name)
            if sha(path)!=digest:
                raise ValueError("policy output checksum mismatch")
            verified.append([name,digest])
            with path.open() as inp:
                h=json.loads(next(inp))
                if (h["cohort_sha256"]!=sha(cohort) or h["draw"]!=draw or h["phase"]!=phase
                        or h["switch"]!=switch or h["budget"]!=budget or h["part"] in parts
                        or covered.intersection(h["ids"])):
                    raise ValueError("unexpected policy worker protocol")
                parts.add(h["part"])
                covered.update(h["ids"])
                done=False
                for line in inp:
                    r=json.loads(line)
                    if r["kind"]=="complete":
                        done=True
                    elif r["kind"]=="batch":
                        batches.append(r)
                    else:
                        p=r["id"]
                        k=p,draw,r["arm"]
                        if (r["kind"]!="rollout" or r["draw"]!=draw or p not in h["ids"] or k in raw
                                or r["seed"]!=seed_for(p,draw) or r["post_seed"]!=post_seed(p,draw)
                                or len(tokens(r))>budget):
                            raise ValueError("unexpected or over-budget policy rollout")
                        raw[k]=r
                if not done:
                    raise ValueError("incomplete policy output")
        if parts!=set(range(h["partitions"])) or covered!=set(cases):
            raise ValueError("missing policy prompt partition")
    expected={(p,d,a) for p in cases for d in range(len(receipts)) for a in ARMS}
    if raw.keys()!=expected:
        raise ValueError("incomplete policy matrix")
    return plan,cases,raw,batches,verified,len(receipts)


def summarize(root,phase,tokenizer):
    plan,cases,rows,batches,receipts,draws=read_run(root,phase)
    [turn]=tokenizer.encode("<|im_end|>",add_special_tokens=False)
    [marker]=tokenizer.encode("</think>",add_special_tokens=False)
    chosen=set()
    for p in cases:
        for d in range(draws):
            native, question, keep, reset=(rows[p,d,a] for a in
                                           ("native","question_e0","switch_keep","switch_restart"))
            s=reaches_switch(native,plan["switch"],marker)
            if keep["switched"]!=s or reset["switched"]!=s:
                raise ValueError("switch used a different early-stop decision")
            if s:
                t=plan["switch"]
                for row in (keep,reset):
                    if row["tokens"][:t]!=native["tokens"][:t] or row["tokens"][t]!=marker or row["spent_before_switch"]!=t:
                        raise ValueError("switch trunk or budget accounting diverged")
                if reset["source_arm"]=="question_e0":
                    expected=nested_restart(question,plan["budget"],t)
                    if reset["tokens"][t:]!=expected["tokens"] or reset["eos_token"]!=expected["eos_token"]:
                        raise ValueError("nested restart does not equal same-template direct cut")
                chosen.add((p,d))
            elif tokens(native)!=tokens(keep) or tokens(native)!=tokens(reset):
                raise ValueError("native early close must be carried to both switch policies")
    y={k:score_policy(cases[k[0]],r,tokenizer,turn) for k,r in rows.items()}
    nested = {}
    for budget in plan.get("nested_total_budgets", []):
        if not plan["switch"] < budget < plan["budget"]:
            raise ValueError("nested budgets must keep the switch and shorten the trajectory")
        reduced = {}
        for k,r in rows.items():
            ids,eos = cut(r,"total",budget)
            sample = r | {"tokens":ids,"eos_token":r["eos_token"] if eos else None}
            reduced[k] = score_policy(cases[k[0]],sample,tokenizer,turn)
        cells = {
            a:{metric:sum(v[metric] for k,v in reduced.items() if k[2]==a)
               for metric in next(iter(reduced.values()))} for a in ARMS
        }
        effects = {f"{a}_minus_{b}":paired_interval({
            p:[int(reduced[p,d,a]["correct_eos"])-int(reduced[p,d,b]["correct_eos"]) for d in range(draws)]
            for p in cases
        }) for a,b in CONTRASTS}
        nested[str(budget)] = {"arms":cells,"effects":effects}
    parts={"all":{(p,d) for p in cases for d in range(draws)},"actually_switched":chosen}
    result={}
    for name,ids in parts.items():
        if not ids:
            continue
        cells={}
        for arm in ARMS:
            obs=[y[p,d,arm] for p,d in ids]
            cells[arm]={"n":len(obs),**{k:sum(bool(r[k]) for r in obs) for k in obs[0]}}
        paired={}
        for a,b in CONTRASTS:
            diffs={}
            for p,d in sorted(ids):
                diffs.setdefault(p,[]).append(int(y[p,d,a]["correct_eos"])-int(y[p,d,b]["correct_eos"]))
            paired[f"{a}_minus_{b}"]=paired_interval(diffs)
        result[name]={"prompts":len({p for p,_ in ids}),"arms":cells,"paired":paired}
    timings={}
    for category in ("time_seconds","prefill_seconds"):
        keys={k for r in batches for k in r.get(category,{})}
        timings[category]={k:statistics.median(r[category][k] for r in batches if k in r.get(category,{}))
                           for k in sorted(keys)}
    parity=[r.get("nested_parity") for r in rows.values() if r.get("nested_parity") is not None]
    trunks=[ok for b in batches for ok in b.get("short_trunk_exact",[])]
    return {"phase":phase,"dataset":plan["dataset"],"scope":plan["scope"],"budget":plan["budget"],"switch":plan["switch"],
            "draws":draws,"verified_shards":receipts,"strata":result,"median_batch_seconds":timings,
            "nested_total":nested,
            "restart_decomposition":restart_decomposition(y,cases,draws,chosen),
            "pilot_explicit_restart_parity":{"same":sum(parity),"total":len(parity)},
            "pilot_short_trunk_parity":{"same":sum(trunks),"total":len(trunks)}}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase",choices=("pilot","eval"),required=True)
    for name in ("root","model","output"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    out=summarize(a.root,a.phase,tok)
    if a.output.exists():
        raise FileExistsError(a.output)
    save(a.output,out)
    print(json.dumps({"phase":a.phase,"sha256":sha(a.output),"prompts":out["strata"]["all"]["prompts"],
                      "draws":out["draws"],"switching_parity":out["pilot_explicit_restart_parity"],
                      "restart_decomposition":out["restart_decomposition"]}))
