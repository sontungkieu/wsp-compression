"""Replay fixed and exact-cycle decision points with prefix-only features."""
import argparse
import json
import math
from pathlib import Path
import time

from repetition_baseline import first_recurrent, render, sha
from repetition_budget_run import packed_hash, tokens
from repetition_budget_score import cut
from repetition_cache_factorial import CONV, NATIVE, ZERO_CONV, sample
from repetition_canonical import paired_interval, seed_for
from repetition_followup_baseline import runtime_for, verified
from repetition_late_confirmation import INCREMENT, LATEST, TOTAL
from repetition_mediation_queue import load, save
from repetition_phase_trace import first_long_cycle, repeated_ngram_fraction
from repetition_replay import terminal_cycle
from repetition_restart import score_policy
from repetition_sampler_reach import pack_at
from repetition_state_value import inputs_for, post_seed

ANCHORS=("512","2048","exact")
DIRECT="no_think"
ARMS=(NATIVE,CONV,DIRECT)
FEATURES=("end_survives","end_mass","end_gap","recurrence","repeat16")


def decisions(raw,marker,anchor,budget=INCREMENT):
    ids=raw["tokens"]
    end=ids.index(marker) if marker in ids else len(ids)
    thought=ids[:end]
    if anchor=="exact":
        return first_long_cycle(thought[:min(LATEST,TOTAL-budget)])
    fixed=int(anchor)
    # An emitted EOS at exactly the decision time is already finished.
    return fixed if fixed<=end and fixed<=len(ids) else None


def prefix_cases(rows,base,marker,anchor,budget=INCREMENT):
    cases=[]
    for p,row in rows.items():
        trace=base[p,"native",0]
        at=decisions(trace,marker,anchor,budget)
        if at is None:
            continue
        prefix=trace["tokens"][:at]
        if marker in prefix or at+budget>TOTAL:
            raise ValueError("late timing case is outside the verified reasoning budget")
        cases.append({"row":row,"completion_prefix":prefix,"render_sha256":trace["render_sha256"],
                      "anchor":anchor,"prefix_tokens":at,
                      "recurrence":first_recurrent(prefix) is not None,
                      "repeat16":repeated_ngram_fraction(prefix[-128:],16)})
    return cases


def preview(r,group):
    features=[]
    [marker]=r.groups["think"]
    def capture(probabilities,logits):
        top=logits.max(dim=-1).values
        values=logits[:,marker]
        gaps=(values-top).tolist()
        mass=probabilities[:,marker].tolist()
        for p,g in zip(mass,gaps,strict=True):
            features.append({"end_survives":p>0,"end_mass":float(p),
                             "end_gap":float(g) if math.isfinite(g) else None})
        return probabilities
    r.generate(group,"native",0,1,first_transform=capture)
    if len(features)!=len(group):
        raise ValueError("first-step feature callback did not capture every pre-decision prefix")
    return features


def run(a):
    head,rows,raw,_=verified(a.raw,a.fingerprint,a.panel)
    if head["model_label"]!="Qwen3.5-0.8B" or (not a.canary and head["scheme"][0][2]!=TOTAL):
        raise ValueError("timing stage needs the locked A native 16K baseline")
    r=runtime_for(a)
    [marker]=r.groups["think"]
    if marker!=head["marker"] or r.manifest["tokenizer_sha256"]!=head["tokenizer_sha256"]:
        raise ValueError("timing prefix tokenizer differs from source")
    if a.cap<INCREMENT or a.cap>TOTAL-2048:
        raise ValueError("timing continuation budget is outside the registered window")
    cases=prefix_cases(rows,raw,marker,a.anchor,a.cap)
    items=inputs_for(r.tokenizer,cases)
    partial=a.output.with_suffix(".partial.jsonl")
    if a.output.exists():
        raise FileExistsError(a.output)
    with partial.open("x") as f,r.torch.inference_mode():
        def emit(value):
            print(json.dumps(value,allow_nan=False),file=f,flush=True)
        arms=ARMS+((ZERO_CONV,) if a.zero_conv else ())
        emit({"kind":"manifest","anchor":a.anchor,"source":str(a.raw),"baseline_sha256":a.fingerprint,
              "dataset":head["dataset"],"split":head["split"],"panel_sha256":sha(a.panel),
              "worker_sha256_timing":sha(Path(__file__)),"arms":arms,"cap":a.cap,"draws":a.draws,
              "marker":marker,"turn":head["turn"],"eligible":len(items),"panel_size":len(rows),**r.manifest})
        start=time.monotonic()
        for i in range(0,len(items),a.batch):
            if time.time()>=a.deadline-150:
                raise TimeoutError("timing shard reached reserved end window")
            group=items[i:i+a.batch]
            estimates=preview(r,group)
            evidence={x["row"]["id"]:{**m,**{k:x[k] for k in ("prefix_tokens","recurrence","repeat16")},
                      "kind":"feature","id":x["row"]["id"],"prefix_sha256":packed_hash(x["completion_prefix"])}
                      for x,m in zip(group,estimates,strict=True)}
            for row in evidence.values():
                emit(row)
            question=[x|{"presence_seed_ids":[],"no_think_ids":r.tokenizer.encode(
                     render(r.tokenizer,x["row"],"no-think"),add_special_tokens=False)} for x in group]
            width=max(max(len(x["native_ids"]),len(x["no_think_ids"])) for x in question)
            packed=pack_at(r,[x["no_think_ids"] for x in question],width)
            for draw in range(a.draws):
                if time.time()>=a.deadline-150:
                    raise TimeoutError("timing draw reached reserved end window")
                outs,audit=sample(r,group,draw,a.cap-1,a.canary,swap_arms=(CONV,),question_only=False,
                                  zero_conv=a.zero_conv)
                outs[DIRECT]=r.generate(question,"no_think",draw,a.cap,packed=packed,start_after_marker=True)
                emit({"kind":"audit","draw":draw,"cache":audit,"ids":list(evidence)})
                for arm in arms:
                    for trace in outs[arm]:
                        emit(trace|{"arm":arm,"prefix_sha256":evidence[trace["id"]]["prefix_sha256"]})
        emit({"kind":"complete","seconds":time.monotonic()-start})
    partial.replace(a.output)


def report(path,fingerprint,panel,model):
    from transformers import AutoTokenizer
    if sha(path)!=fingerprint:
        raise ValueError("timing raw digest changed")
    with path.open() as f:
        h,*body,end=map(json.loads,f)
    if (h["worker_sha256_timing"]!=sha(Path(__file__)) or h["panel_sha256"]!=sha(panel)
            or end["kind"]!="complete"):
        raise ValueError("timing worker or panel proof changed")
    _,rows,base,_=verified(Path(h["source"]),h["baseline_sha256"],panel)
    budget=h["cap"]
    recorded=tuple(h["arms"])
    if recorded not in (ARMS,(*ARMS,ZERO_CONV)):
        raise ValueError("unknown timing replay arm")
    cases={x["row"]["id"]:x for x in prefix_cases(rows,base,h["marker"],h["anchor"],budget)}
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
    features={r["id"]:r for r in body if r["kind"]=="feature"}
    if features.keys()!=cases.keys() or len(features)!=h["eligible"]:
        raise ValueError("timing feature proof incomplete")
    raw={}
    for r in body:
        if r["kind"]!="rollout":
            continue
        p,d,a=r["id"],r["draw"],r["arm"]
        if ((p,d,a) in raw or len(tokens(r))>budget or r["post_seed"]!=post_seed(p,d)
               or r["seed"]!=(post_seed(p,d) if a==DIRECT else seed_for(p,d))
               or r["prefix_sha256"]!=packed_hash(cases[p]["completion_prefix"])
               or (a!=DIRECT and r["tokens"][0]!=h["marker"])
               or (r["eos_token"] is not None and r["eos_token"] not in h["eos_ids"])):
            raise ValueError("invalid timing branch")
        raw[p,d,a]=r
    if raw.keys()!={(p,d,a) for p in cases for d in range(h["draws"]) for a in recorded}:
        raise ValueError("timing raw matrix incomplete")
    windows=sorted({512,1024,INCREMENT,budget})
    cells,counts={},{str(c):{} for c in windows}
    for p,case in cases.items():
        learned={k:features[p][k] for k in FEATURES}
        at=case["prefix_tokens"]
        entry={"prefix_tokens":at,"features":learned,"arms":{}}
        for cap in windows:
            arms={}
            for a in recorded:
                arms[a]=[]
                for draw in range(h["draws"]):
                    r=raw[p,draw,a]
                    ids,eos=cut(r,"total",cap)
                    out=score_policy(rows[p],r|{"tokens":ids,"eos_token":r["eos_token"] if eos else None},tok,h["turn"])
                    arms[a].append(out|{"tokens":len(ids)+int(eos),"cost_from_start":at+len(ids)+int(eos),
                                         "terminal_exact":terminal_cycle(ids[1:] if a!=DIRECT else ids) is not None})
            n=base[p,"native",0]
            ids,eos=cut(n,"total",at+cap)
            arms["native_suffix"]=[score_policy(rows[p],n|{"tokens":ids,"eos_token":n["eos_token"] if eos else None},tok,h["turn"])
                                    |{"cost_from_start":len(ids)+int(eos)}]
            entry["arms"][str(cap)]=arms
        cells[p]=entry
    for k in counts:
        for a in (*recorded,"native_suffix"):
            v=[r for y in cells.values() for r in y["arms"][k][a]]
            counts[k][a]={"n":len(v),**{m:sum(bool(r[m]) for r in v) for m in ("correct_eos","wrong_eos","missing_eos","cap")},
                           "tokens_from_start":sum(r["cost_from_start"] for r in v)}
        counts[k]["conv_minus_keep"]=paired_interval({p:[int(x["correct_eos"])-int(y["correct_eos"])
                         for x,y in zip(v["arms"][k][CONV],v["arms"][k][NATIVE],strict=True)] for p,v in cells.items()}) if cells else None
        if ZERO_CONV in recorded:
            counts[k]["zero_minus_keep"]=paired_interval({p:[int(x["correct_eos"])-int(y["correct_eos"])
                         for x,y in zip(v["arms"][k][ZERO_CONV],v["arms"][k][NATIVE],strict=True)] for p,v in cells.items()}) if cells else None
    return {"header":h,"eligible":len(cases),"excluded":len(rows)-len(cases),"counts":counts,
            "items":cells,"worker_seconds":end["seconds"],"raw_sha256":fingerprint}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("run","score"))
    for name in ("panel","model","checkpoint","data","raw","output"):
        p.add_argument("--"+name,type=Path)
    p.add_argument("--fingerprint")
    p.add_argument("--anchor",choices=ANCHORS)
    for name,default in (("draws",2),("batch",4),("deadline",None),("cap",INCREMENT)):
        p.add_argument("--"+name,type=int,default=default)
    p.add_argument("--canary",action="store_true")
    p.add_argument("--zero-conv",action="store_true")
    a=p.parse_args()
    if a.action=="run":
        run(a)
    else:
        if a.output.exists():
            raise FileExistsError(a.output)
        save(a.output,report(a.raw,a.fingerprint,a.panel,a.model))


if __name__=="__main__":
    main()
