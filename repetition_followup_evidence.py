"""Change the evidence instruction while retaining the exact original evidence text."""
import argparse
import json
from pathlib import Path
import re
import time

from repetition_baseline import math_answer, sha
from repetition_budget_run import packed_hash, tokens
from repetition_canonical import paired_interval, seed_for
from repetition_followup_baseline import runtime_for
from repetition_mediation_queue import load, save
from repetition_sampler_reach import pack_at
from repetition_state_value import inputs_for, post_seed

ORIGINAL = "fresh_evidence"
RECORD = "verified_record"
INTRO = re.compile(r"^(?:A ledger checkpoint exists|A calculator register was saved|An integer counter snapshot was recorded) in the scratchpad\. ")


def read_source(path, fingerprint, panel):
    if sha(path) != fingerprint:
        raise ValueError("evidence comparison source checksum changed")
    with path.open() as stream:
        head, *body, end = map(json.loads, stream)
    if head["panel_sha256"] != sha(panel) or ORIGINAL not in head["arms"] or end["kind"] != "complete":
        raise ValueError("original evidence matrix is incomplete")
    groups = load(panel)["panels"][head["split"]][head["start"]:head["start"]+head["count"]]
    rows = [v for item in groups for v in item["cells"].values()]
    old = {(r["id"],r["draw"],r["variant"]):r for r in body if r.get("arm")==ORIGINAL}
    if len(groups)!=head["count"] or len(old)!=len(rows)*head["draws"]:
        raise ValueError("original evidence has missing outcomes")
    return head, rows, old


def new_row(item):
    old = item["evidence_row"]
    revised, count = INTRO.subn("Use the verified checkpoint record below. ", old["prompt"])
    if count != 1 or "Verified record:" not in revised:
        raise ValueError("original evidence prompt has the wrong scaffold")
    return {"row": old | {"prompt":revised}, "completion_prefix":[]}


def run(args):
    head, rows, old = read_source(args.raw,args.fingerprint,args.panel)
    r = runtime_for(args)
    [marker] = r.groups["think"]
    if head["tokenizer_sha256"]!=r.manifest["tokenizer_sha256"]:
        raise ValueError("evidence replay tokenizer differs")
    cap=head["total_cap"]
    temp=args.output.with_suffix(".partial.jsonl")
    if args.output.exists():
        raise FileExistsError(args.output)
    with temp.open("x") as stream, r.torch.inference_mode():
        def emit(item):
            print(json.dumps(item,allow_nan=False),file=stream,flush=True)
        emit({"kind":"manifest","worker_sha256_evidence":sha(Path(__file__)),
              "source":str(args.raw),"source_raw_sha256":args.fingerprint,
              "panel_sha256":sha(args.panel),"marker":marker,"cap":cap,"count":head["count"],
              "split":head["split"],"draws":head["draws"],**r.manifest})
        t=time.monotonic()
        for draw in range(head["draws"]):
            for start in range(0,len(rows),8):
                if time.time()>=args.deadline-150:
                    raise TimeoutError("evidence prompt shard nearing end window")
                group=rows[start:start+8]
                original=inputs_for(r.tokenizer,[{"row":x["evidence_row"],"completion_prefix":[]} for x in group])
                retained=inputs_for(r.tokenizer,group)
                revised=inputs_for(r.tokenizer,list(map(new_row,group)))
                old_width=max(len(x["native_ids"]) for x in (*original,*retained))
                new_width=max(old_width,max(len(x["native_ids"]) for x in revised))
                batch=pack_at(r,[x["native_ids"] for x in revised],new_width)
                for item, new, trace in zip(group,revised,r.generate(revised,"e0",draw,cap-1,packed=batch),strict=True):
                    key=(item["row"]["id"],draw,item["variant"])
                    if (trace["seed"],trace["post_seed"])!=(old[key]["seed"],old[key]["post_seed"]):
                        raise ValueError("paired evidence prompt seeds differ")
                    emit(trace | {"variant":item["variant"],"old_width":old_width,"new_width":new_width,
                                  "input_sha256":packed_hash(new["native_ids"])})
        emit({"kind":"complete","seconds":time.monotonic()-t})
    temp.replace(args.output)


def report(path,fingerprint,panel,model):
    from transformers import AutoTokenizer
    if sha(path)!=fingerprint:
        raise ValueError("revised evidence shard checksum mismatch")
    with path.open() as stream:
        head,*records,end=map(json.loads,stream)
    if (head["panel_sha256"]!=sha(panel) or head["worker_sha256_evidence"]!=sha(Path(__file__))
            or end["kind"]!="complete"):
        raise ValueError("revised evidence actor mismatch")
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
    original,rows,old=read_source(Path(head["source"]),head["source_raw_sha256"],panel)
    gold={(x["row"]["id"],x["variant"]):x["row"] for x in rows}
    if original["total_cap"]!=head["cap"] or original["draws"]!=head["draws"]:
        raise ValueError("revised evidence budget differs")
    new={}
    for trace in records:
        p,d,v=(trace[x] for x in ("id","draw","variant"))
        if (p,d,v) in new or len(tokens(trace))>head["cap"] or trace["seed"]!=seed_for(p,d) or trace["post_seed"]!=post_seed(p,d):
            raise ValueError("invalid revised evidence trajectory")
        new[p,d,v]=trace
    if new.keys()!=old.keys():
        raise ValueError("revised evidence outcomes differ from frozen panel")
    cells={ORIGINAL:{},RECORD:{}}
    for arm,raw in ((ORIGINAL,old),(RECORD,new)):
        for (p,d,v),trace in sorted(raw.items()):
            if trace["tokens"][0]!=head["marker"] or (trace["eos_token"] is not None and trace["eos_token"] not in head["eos_ids"]):
                raise ValueError("evidence marker or EOS is invalid")
            answer=tok.decode(trace["tokens"][1:],skip_special_tokens=False)
            result=math_answer(gold[p,v],answer)
            eos=trace["eos_token"] is not None
            cells[arm].setdefault(p,[]).append({"draw":d,"variant":v,"correct_eos":eos and result["numeric_correct"],
                "wrong_eos":eos and result["answer_found"] and not result["numeric_correct"],
                "missing_eos":eos and not result["answer_found"],"cap":not eos,
                "unknown":not result["numeric_correct"] and re.search(r"\bunknown\b",answer,re.I) is not None})
    metrics={a:{k:sum(x[k] for group in by_id.values() for x in group)
                for k in ("correct_eos","wrong_eos","missing_eos","cap","unknown")} for a,by_id in cells.items()}
    both={}
    for arm,by_id in cells.items():
        both[arm]={}
        for p,items in by_id.items():
            q={(r["draw"],r["variant"]):r["correct_eos"] for r in items}
            both[arm][p]=[q[d,"v0_"+m] and q[d,"v1_"+m] for d in range(head["draws"]) for m in ("m0","m1")]
    diffs={p:[int(a)-int(b) for a,b in zip(q,both[ORIGINAL][p],strict=True)] for p,q in both[RECORD].items()}
    return {"header":head,"raw_sha256":fingerprint,"n":len(old),"arms":metrics,"pairs":both,
            "same_padding":sum(x["old_width"]==x["new_width"] for x in new.values()),
            "verified_record_minus_original":paired_interval(diffs),"seconds":end["seconds"]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("run","score"))
    for name in ("panel","raw","model","data","checkpoint","output"):
        p.add_argument("--"+name,type=Path)
    p.add_argument("--fingerprint")
    p.add_argument("--deadline",type=int)
    args=p.parse_args()
    if args.action=="run":
        run(args)
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        save(args.output,report(args.raw,args.fingerprint,args.panel,args.model))


if __name__=="__main__":
    main()
