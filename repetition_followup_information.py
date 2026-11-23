"""Factor checkpoint values and subsequent operations, with a fresh textual-evidence control."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from repetition_baseline import astra_instruction, math_answer, render, sha
from repetition_budget_run import packed_hash, tokens
from repetition_cache_factorial import BOTH, CONV, DROP, MATRIX, NATIVE, ZERO_CONV, sample
from repetition_canonical import paired_interval
from repetition_followup_baseline import runtime_for
from repetition_mediation_queue import load, save
from repetition_sampler_reach import pack_at
from repetition_state_data import keyed
from repetition_state_value import THINK_END, inputs_for

EVIDENCE="fresh_evidence"
ARMS=(NATIVE,CONV,MATRIX,BOTH,DROP,EVIDENCE)
CELLS=(("v0","m0"),("v1","m0"),("v0","m1"),("v1","m1"))
FAMILIES=(
    ("ledger","A ledger checkpoint exists in the scratchpad.","Ledger checkpoint = {v}.",
     ("Add 17 to the checkpoint.","Triple the checkpoint, then subtract 7."),((1,17),(3,-7))),
    ("register","A calculator register was saved in the scratchpad.","Saved register value: {v}.",
     ("Double the saved register and add 31.","Subtract 23 from the saved register."),((2,31),(1,-23))),
    ("counter","An integer counter snapshot was recorded in the scratchpad.","Counter snapshot is {v}.",
     ("Add 39 to the snapshot.","Multiply the snapshot by 4 and subtract 13."),((1,39),(4,-13))),
)


def cell(tokenizer,panel_id,family,variant,op,value):
    name,intro,draft,operations,formulas=family
    statement=intro+" "+operations[op]+" If the checkpoint value is unavailable, write UNKNOWN."
    instruction=astra_instruction("canonical")
    v,m=formulas[op]
    row={"id":panel_id,"task":"math","prompt":statement+"\n"+instruction,
         "gold":str(v*value+m),"astra_prompt_arm":"canonical","boxed_answer_required":True}
    text=render(tokenizer,row,"think")
    base=tokenizer.encode(text,add_special_tokens=False)
    observed=draft.format(v=value)+"\n"
    full=tokenizer.encode(text+observed,add_special_tokens=False)
    if full[:len(base)]!=base or not full[len(base):] or THINK_END in full[len(base):]:
        return None
    disclosed=row|{"prompt":statement+"\nVerified record: "+draft.format(v=value)+"\n"+instruction}
    return {"row":row,"completion_prefix":full[len(base):],"value":value,"operation":op,
            "variant":variant,"render_sha256":hashlib.sha256(text.encode()).hexdigest(),"evidence_row":disclosed}


def make_pair(tokenizer,panel_id,family,v0,v1):
    cells={}
    for vk,mk in CELLS:
        variant=vk+"_"+mk
        item=cell(tokenizer,panel_id,family,variant,int(mk[1]),v0 if vk=="v0" else v1)
        if item is None:
            return None
        cells[variant]=item
    for op in ("m0","m1"):
        first,second=(cells[f"{v}_{op}"] for v in ("v0","v1"))
        if (first["row"]["prompt"]!=second["row"]["prompt"] or first["row"]["gold"]==second["row"]["gold"]
                or len(first["completion_prefix"])!=len(second["completion_prefix"])):
            return None
    if len({item["row"]["gold"] for item in cells.values()})!=len(CELLS):
        return None
    return {"id":panel_id,"family":family[0],"cells":cells}


def freeze(tokenizer,path,excluded=None):
    if path.exists():
        raise FileExistsError(path)
    prior = ({c["value"] for rows in load(excluded)["panels"].values()
              for item in rows for c in item["cells"].values()} if excluded else set())
    values=sorted(set(range(201,993))-prior,key=lambda x:keyed(str(x),"wsp-followup-information-v1"))
    plans={"dev":[],"eval":[]}
    layout=(("dev",12,FAMILIES[0]),("eval",48,FAMILIES[1]),("eval",48,FAMILIES[2]))
    offset=0
    for split,target,family in layout:
        found=0
        while found<target:
            a,b=values[offset:offset+2]
            offset+=2
            if a>b:
                a,b=b,a
            prefix="checkpoint-cross-v2" if excluded else "checkpoint-cross"
            row=make_pair(tokenizer,f"{prefix}:{family[0]}:{found:03d}",family,a,b)
            if row is not None:
                plans[split].append(row)
                found+=1
    save(path,{"panels":plans,"held_out_families":[x[0] for x in FAMILIES[1:]],"dev_family":FAMILIES[0][0],
               "contrasts":["conv_only-native","matrix_only-native","both-native","fresh_evidence-question_only"],
               "excluded_panel_sha256":sha(excluded) if excluded else None,
               "scope":"disjoint synthesized values and template families across splits"})


def check_question_only(dropped):
    for start in range(0,len(dropped),len(CELLS)):
        block=dropped[start:start+len(CELLS)]
        if len(block)!=len(CELLS) or any(tokens(block[j])!=tokens(block[j+1]) for j in (0,2)):
            raise ValueError("question-only arm leaked the hidden value")


def run(a):
    cases=load(a.panel)["panels"][a.split][a.start:a.start+a.count]
    if len(cases)!=a.count:
        raise ValueError("controlled information shard does not match frozen range")
    r=runtime_for(a)
    partial=a.output.with_suffix(".partial.jsonl")
    if a.output.exists():
        raise FileExistsError(a.output)
    with partial.open("x") as stream,r.torch.inference_mode():
        def emit(value):
            print(json.dumps(value,allow_nan=False),file=stream,flush=True)
        arms=ARMS + ((ZERO_CONV,) if a.zero_conv else ()) + (("self_copy",) if a.canary else ())
        emit({"kind":"manifest","panel_sha256":sha(a.panel),"worker_sha256_information":sha(Path(__file__)),
              "split":a.split,"start":a.start,"count":a.count,"arms":arms,"total_cap":a.cap,"draws":a.draws,**r.manifest})
        tick=time.monotonic()
        for draw in range(a.draws):
            for i in range(0,len(cases),2):
                if time.time()>=a.deadline-120:
                    raise TimeoutError("controlled information shard nearing deadline")
                source=[x["cells"][f"{v}_{m}"] for x in cases[i:i+2] for v,m in CELLS]
                items=inputs_for(r.tokenizer,source)
                traces,audit=sample(r,items,draw,a.cap-1,a.canary,zero_conv=a.zero_conv)
                fresh=inputs_for(r.tokenizer,[{"row":x["evidence_row"],"completion_prefix":[]} for x in items])
                width=max(len(x["native_ids"]) for x in [*items,*fresh])
                packed=pack_at(r,[x["native_ids"] for x in fresh],width)
                traces[EVIDENCE]=r.generate(fresh,"e0",draw,a.cap-1,packed=packed)
                check_question_only(traces[DROP])
                emit({"kind":"audit","draw":draw,"ids":list(dict.fromkeys(x["row"]["id"] for x in items)),"cache":audit})
                for arm,generated in traces.items():
                    for raw,item in zip(generated,items,strict=True):
                        emit(raw|{"arm":arm,"variant":item["variant"],"source_prefix_sha256":packed_hash(item["completion_prefix"])})
        emit({"kind":"complete","seconds":time.monotonic()-tick})
    partial.replace(a.output)


def report(path,fingerprint,panel,model):
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
    if sha(path)!=fingerprint:
        raise ValueError("controlled information shard checksum differs")
    with path.open() as stream:
        head,*body,end=map(json.loads,stream)
    if (head["panel_sha256"]!=sha(panel) or head["worker_sha256_information"]!=sha(Path(__file__))
            or end["kind"]!="complete"):
        raise ValueError("controlled information receipt differs")
    cases=load(panel)["panels"][head["split"]][head["start"]:head["start"]+head["count"]]
    ids={item["id"]:item for item in cases}
    cells={}
    for r in body:
        if r["kind"]=="audit":
            continue
        p,d,a,v=r["id"],r["draw"],r["arm"],r["variant"]
        item=ids[p]["cells"][v]
        if ((p,d,a,v) in cells or not r["tokens"] or r["tokens"][0]!=THINK_END
                or len(tokens(r))>head["total_cap"] or r["source_prefix_sha256"]!=packed_hash(item["completion_prefix"])
                or (r["eos_token"] is not None and r["eos_token"] not in head["eos_ids"])):
            raise ValueError("invalid controlled information trajectory")
        eos=r["eos_token"] is not None
        scored=math_answer(item["row"],tok.decode(r["tokens"],skip_special_tokens=False))
        cells[p,d,a,v]={"correct_eos":eos and scored["numeric_correct"],"wrong_eos":eos and scored["answer_found"] and not scored["numeric_correct"],
                        "missing_eos":eos and not scored["answer_found"],"cap":not eos,"tokens":len(tokens(r))}
    draws=range(head["draws"])
    if cells.keys()!={(p,d,a,v+"_"+m) for p in ids for d in draws for a in head["arms"] for v,m in CELLS}:
        raise ValueError("incomplete controlled information matrix")
    arms={}
    pairs={}
    for arm in head["arms"]:
        rows=[y for (p,d,a,v),y in cells.items() if a==arm]
        pairs[arm]={p:[all(cells[p,d,arm,v+"_"+m]["correct_eos"] for v in ("v0","v1"))
                       for d in draws for m in ("m0","m1")] for p in ids}
        arms[arm]={k:sum(x[k] for x in rows) for k in ("correct_eos","wrong_eos","missing_eos","cap","tokens")}
        arms[arm].update(n=len(rows),both_values_correct=sum(sum(x) for x in pairs[arm].values()),
                         all_four_correct=sum(all(cells[p,d,arm,v+"_"+m]["correct_eos"] for v,m in CELLS)
                                              for p in ids for d in draws))
    contrasts={a:paired_interval({p:[int(x)-int(y) for x,y in zip(pairs[a][p],pairs[NATIVE][p],strict=True)]
                                  for p in ids}) for a in head["arms"] if a!=NATIVE}
    return {"header":head,"template_counts":dict(Counter(x["family"] for x in cases)),"arms":arms,
            "paired_both_values_vs_keep":contrasts,"pairs":pairs,"seconds":end["seconds"],"raw_sha256":fingerprint}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("freeze","run","score"))
    for name in ("root","panel","model","checkpoint","data","raw","output","exclude-panel"):
        p.add_argument("--"+name,type=Path)
    p.add_argument("--split")
    p.add_argument("--fingerprint")
    for name,default in (("start",0),("count",12),("cap",256),("draws",2),("deadline",None)):
        p.add_argument("--"+name,type=int,default=default)
    p.add_argument("--canary",action="store_true")
    p.add_argument("--zero-conv",action="store_true")
    a=p.parse_args()
    if a.action=="freeze":
        from transformers import AutoTokenizer
        freeze(AutoTokenizer.from_pretrained(a.model,local_files_only=True),a.panel,a.exclude_panel)
    elif a.action=="run":
        run(a)
    else:
        if a.output.exists():
            raise FileExistsError(a.output)
        save(a.output,report(a.raw,a.fingerprint,a.panel,a.model))


if __name__=="__main__":
    main()
