"""Freeze new math panels and run matched thinking and direct baselines."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time

from repetition_baseline import DECIMAL, render, sha
from repetition_budget_run import packed_hash, tokens
from repetition_budget_score import cut
from repetition_canonical import paired_interval, seed_for
from repetition_mediation_queue import load, save
from repetition_phase_trace import first_long_cycle
from repetition_qwen35_horizon import measure
from repetition_restart import prepare_items, score_policy
from repetition_restart_cohort import canonical_problem, read_source
from repetition_restart_transfer import Qwen3Runtime
from repetition_sampler_reach import pack_at
from repetition_state_data import keyed
from repetition_state_value import Runtime, post_seed

TOTAL = 16384
WINDOWS = (4096,8192,TOTAL)
MODES = ("native","no_think")
EXPOSED = re.compile(rb"(?:math500:[0-9a-f]{64}(?![0-9a-f])|gsm8k:[0-9]+(?![0-9]))")


def exposure(archive,current):
    seen,receipts = set(),[]
    for path in sorted(archive.rglob("*")):
        if (path.suffix not in {".json",".jsonl"} or current in path.parents
                or path.is_symlink() or not path.is_file()):
            continue
        ids = set()
        with path.open("rb") as stream:
            for line in stream:
                ids.update(m.decode("ascii") for m in EXPOSED.findall(line))
        if ids:
            seen.update(ids)
            receipts.append([str(path.relative_to(archive)),len(ids)])
    return seen,receipts


def freeze(root,previous,archive,gsm):
    source = Path(load(previous)["source_path"])
    if sha(source)!=load(previous)["source_sha256"]:
        raise ValueError("MATH-500 mirror changed")
    gsm_sha = load(gsm.parent/"manifest.json")["artifacts"][gsm.name]["sha256"]
    if sha(gsm)!=gsm_sha:
        raise ValueError("GSM8K test artifact changed")
    seen,receipts = exposure(archive,root)
    panels,digests = {},{}
    for kind,file,counts in (("math500",source,(24,80)),("gsm8k",gsm,(32,128))):
        pool = {}
        for i,item in enumerate(read_source(file)):
            problem = item["problem"] if kind=="math500" else item["question"]
            gold = (str(item["answer"]) if kind=="math500" else item["answer"].rsplit("####",1)[-1]).strip()
            ident = ("math500:"+hashlib.sha256(problem.encode()).hexdigest()
                     if kind=="math500" else f"gsm8k:{i}")
            if ident in seen or not DECIMAL.fullmatch(gold):
                continue
            if ident in pool:
                raise ValueError("duplicate prompt ID")
            pool[ident] = {"row":{"id":ident,"task":"math","prompt":canonical_problem(problem),
                                  "gold":gold.replace(",",""),"boxed_answer_required":True,
                                  "astra_prompt_arm":"canonical"}}
        rank = sorted(pool,key=lambda q:keyed(q,"wsp-followup-20261124-v1"))
        dev,eval_count = counts
        if len(rank)<24:
            panels[kind] = {"dev":[],"eval":[],"clean":len(rank),"blocked":"fewer than 24 distinct archived-clean prompts"}
            digests[kind] = {"path":str(file),"sha256":sha(file),"previously_exposed":sum(p.startswith(kind+":") for p in seen)}
            continue
        dev=min(dev,len(rank)-16)
        panels[kind] = {"dev":[pool[p] for p in rank[:dev]],"eval":[pool[p] for p in rank[dev:dev+eval_count]],
                        "clean":len(rank)}
        digests[kind] = {"path":str(file),"sha256":sha(file),"previously_exposed":sum(p.startswith(kind+":") for p in seen)}
    result = {"datasets":digests,"panels":panels,"archive_files_with_ids":receipts,
              "selection":"hash rank within each dataset, all archived IDs excluded","total":TOTAL}
    save(root/"panel.json",result)
    return result


def scheme(cap,consensus,draw=0):
    arms=(("native",draw,cap),("no_think",draw,cap))
    return arms+(("no_think",draw+1,cap//2),) if consensus else arms


def runtime_for(a):
    return Qwen3Runtime(a) if a.checkpoint else Runtime(a)


def run(a):
    panel=load(a.panel)
    cases=panel["panels"][a.dataset][a.split][a.start:a.start+a.count]
    if len(cases)!=a.count or not cases:
        raise ValueError("baseline shard does not match locked prompt range")
    r=runtime_for(a)
    [marker]=r.groups["think"]
    [turn]=r.tokenizer.encode("<|im_end|>",add_special_tokens=False)
    items=prepare_items(r,cases)
    partial=a.output.with_suffix(".partial.jsonl")
    if a.output.exists():
        raise FileExistsError(a.output)
    with partial.open("x") as stream,r.torch.inference_mode():
        def emit(row):
            print(json.dumps(row,allow_nan=False),file=stream,flush=True)
        arms=scheme(a.cap,a.consensus,a.draw)
        emit({"kind":"manifest","dataset":a.dataset,"split":a.split,"model_label":a.label,
              "start":a.start,"count":a.count,"scheme":arms,"panel_sha256":sha(a.panel),
              "worker_sha256_baseline":sha(Path(__file__)),"marker":marker,"turn":turn,**r.manifest})
        started=time.monotonic()
        for i in range(0,len(items),a.batch):
            group=items[i:i+a.batch]
            width=max(max(len(x["native_ids"]),len(x["no_think_ids"])) for x in group)
            if any(marker in x["native_ids"] or marker not in x["no_think_ids"] for x in group):
                raise ValueError("thinking marker appears in the wrong template")
            for mode,draw,cap in arms:
                if time.time()>=a.deadline-150:
                    raise TimeoutError("baseline shard nearing allocation deadline")
                batch_ids=[x["native_ids" if mode=="native" else "no_think_ids"] for x in group]
                packed=pack_at(r,batch_ids,width)
                tick=time.monotonic()
                generated=r.generate(group,mode,draw,cap,packed=packed)
                for token_ids,item,trace in zip(batch_ids,group,generated,strict=True):
                    emit(trace|{"total_cap":cap,"render_sha256":item["render_sha256"],
                                "input_sha256":packed_hash(token_ids)})
                emit({"kind":"batch","mode":mode,"draw":draw,"elapsed":time.monotonic()-tick,
                      "steps":max(len(tokens(y)) for y in generated)})
        emit({"kind":"complete","seconds":time.monotonic()-started})
    partial.replace(a.output)


def verified(path,fingerprint,panel):
    if sha(path)!=fingerprint:
        raise ValueError(f"baseline SHA changed: {path}")
    with path.open() as stream:
        head,*body,end=map(json.loads,stream)
    if (head["panel_sha256"]!=sha(panel) or head["worker_sha256_baseline"]!=sha(Path(__file__))
            or end["kind"]!="complete"):
        raise ValueError("baseline receipts disagree with frozen source")
    selected=load(panel)["panels"][head["dataset"]][head["split"]][head["start"]:head["start"]+head["count"]]
    rows={x["row"]["id"]:x["row"] for x in selected}
    limits={(a,d):n for a,d,n in head["scheme"]}
    raw={}
    for r in body:
        if r["kind"]=="batch":
            continue
        p,a,d=r["id"],r["arm"],r["draw"]
        limit=limits[a,d]
        if ((p,a,d) in raw or p not in rows or r["total_cap"]!=limit or len(tokens(r))>limit or
                (r["eos_token"] is None and len(tokens(r))!=limit) or
                (r["eos_token"] is not None and r["eos_token"] not in head["eos_ids"]) or
                r["seed"]!=seed_for(p,d) or r["post_seed"]!=post_seed(p,d)):
            raise ValueError("invalid baseline raw trajectory")
        raw[p,a,d]=r
    if raw.keys()!={(p,a,d) for p in rows for a,d in limits}:
        raise ValueError("incomplete baseline shard")
    return head,rows,raw,end


def numeric_result(row,trace,budget,tok,turn):
    from repetition_baseline import math_answer
    ids,eos=cut(trace,"total",budget)
    if turn in ids:
        ids=ids[:ids.index(turn)]
    answer=math_answer(row,tok.decode(ids,skip_special_tokens=False),pending_thought=trace["arm"]!="no_think")
    return answer["parsed_answer"] if eos else None, min(len(tokens(trace)),budget)


def consensus(row,x,y,budget,tok,turn):
    from decimal import Decimal
    first,cost_first=numeric_result(row,x,budget//2,tok,turn)
    second,cost_second=numeric_result(row,y,budget//2,tok,turn)
    accepted=first is not None and second is not None and Decimal(first)==Decimal(second)
    good=accepted and Decimal(first)==Decimal(row["gold"])
    return {"answered":accepted,"correct_eos":good,"wrong_eos":accepted and not good,
            "cost":cost_first+cost_second}


def summary(path,fingerprint,panel,model):
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
    h,rows,raw,end=verified(path,fingerprint,panel)
    windows={}
    budget=h["scheme"][0][2]
    cuts=sorted({n for n in (*WINDOWS,budget) if n<=budget})
    draw=h["scheme"][0][1]
    for cap in cuts:
        reports={}
        outcomes={}
        for arm in MODES:
            cells={}
            for p,row in rows.items():
                r=raw[p,arm,draw]
                thought=r["tokens"]
                if arm=="native" and h["marker"] in thought:
                    thought=thought[:thought.index(h["marker"])]
                y=measure(row,r,h["marker"],h["turn"],cap,tok,first_long_cycle(thought))
                y["cost"]=min(len(tokens(r)),cap)
                y["answered"]=y["correct_eos"] or y["wrong_eos"]
                cells[p]=y
            outcomes[arm]=cells
        if ("no_think",draw+1) in {(a,d) for a,d,_ in h["scheme"]}:
            outcomes["direct_agreement"]={p:consensus(row,raw[p,"no_think",draw],raw[p,"no_think",draw+1],cap,tok,h["turn"])
                                           for p,row in rows.items()}
        for mode,cells in outcomes.items():
            tab=Counter(y for v in cells.values() for y in ("correct_eos","wrong_eos","cap","missing_eos",
                 "generated_close","terminal_exact","first_exact_before_close","recurrent16") if v.get(y))
            reports[mode]=dict(tab)|{"n":len(cells),"answered":sum(v["answered"] for v in cells.values()),
                                     "cost":sum(v["cost"] for v in cells.values())}
        windows[str(cap)]={"arms":reports,"cells":outcomes}
    return {"header":h,"windows":windows,"raw_sha256":fingerprint,"seconds":end["seconds"]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("freeze","run","score"))
    for name in ("root","previous","archive","data","panel","model","checkpoint","output","raw"):
        p.add_argument("--"+name,type=Path)
    for name in ("dataset","split","label","fingerprint"):
        p.add_argument("--"+name)
    for name,default in (("start",0),("count",16),("batch",8),("cap",TOTAL),("draw",0),("deadline",None)):
        p.add_argument("--"+name,type=int,default=default)
    p.add_argument("--consensus",action="store_true")
    a=p.parse_args()
    if a.action=="freeze":
        if (a.root/"panel.json").exists():
            raise FileExistsError(a.root/"panel.json")
        frozen=freeze(a.root,a.previous,a.archive,a.data)
        print(json.dumps({"panel_sha256":sha(a.root/"panel.json"),"counts":{k:{s:len(v[s]) for s in ("dev","eval")}
                                                                         for k,v in frozen["panels"].items()}}))
    elif a.action=="run":
        run(a)
    else:
        if a.output.exists():
            raise FileExistsError(a.output)
        save(a.output,summary(a.raw,a.fingerprint,a.panel,a.model))


if __name__=="__main__":
    main()
