"""Isolate newline placement around the prefixed reasoning close token."""
import argparse
from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time

from repetition_baseline import render, sha
from repetition_budget_run import packed_hash, tokens
from repetition_budget_score import cut
from repetition_cache_swap_score import first_box
from repetition_canonical import paired_interval
from repetition_mediation_queue import Supervisor, check_devices, load, save
from repetition_restart import score_policy
from repetition_sampler_reach import pack_at
from repetition_state_value import Runtime, THINK_END, post_seed

ARM_FORMAT = {"1+0":(1,0),"1+2":(1,2),"2+0":(2,0),"2+2":(2,2)}
EFFECTS = {
    "full_no_think_scaffold_minus_minimal":{"2+2":1,"1+0":-1},
    "main_pre_newline":{"2+0":.5,"1+0":-.5,"2+2":.5,"1+2":-.5},
    "main_post_newline":{"1+2":.5,"1+0":-.5,"2+2":.5,"2+0":-.5},
    "interaction":{"2+2":1,"2+0":-1,"1+2":-1,"1+0":1},
}
CAP,BATCH = 1536,8


def freeze(source, parent, output):
    subset = load(source)["cases"]
    dev = load(parent)["pilot"][:BATCH]
    if (len(subset)!=32 or len(dev)!=BATCH or {x["row"]["id"] for x in subset}&{x["row"]["id"] for x in dev}
            or output.exists()):
        raise ValueError("need the frozen depth panel and separate template canary")
    save(output,{"cases":[x["row"] for x in subset],"canary":[x["row"] for x in dev],
                 "source_cohort_sha256":sha(source),"pilot_source_cohort_sha256":sha(parent),
                 "cap":CAP,"arms":ARM_FORMAT,"initial_presence_ids":[THINK_END]})


def make_inputs(tokenizer, rows):
    def encode(text):
        return tokenizer.encode(text,add_special_tokens=False)

    result = []
    for row in rows:
        thinking = render(tokenizer,row,"think")
        prefix,opener,suffix = thinking.rpartition("<think>")
        if opener!="<think>" or suffix!="\n":
            raise ValueError("native thinking template changed")
        core = prefix+opener
        variants = {name:core+"\n"*before+"</think>"+"\n"*after for name,(before,after) in ARM_FORMAT.items()}
        if variants["2+2"]!=render(tokenizer,row,"no-think"):
            raise ValueError("the 2+2 scaffold does not match the no-thinking template bytes")
        packed = {name:encode(text) for name,text in variants.items()}
        start = encode(thinking)
        core_ids = encode(core)
        if (packed["1+0"]!=start+[THINK_END] or any(ids[:len(core_ids)]!=core_ids for ids in packed.values())
                or encode("</think>")!=[THINK_END]):
            raise ValueError("scaffold and forced marker do not have the intended token sequence")
        result.append({"row":row,"pending":False,"completion_prefix":[],"presence_seed_ids":[THINK_END],
                       "think_prompt":start,"prefixes":packed,"tails":{n:ids[len(core_ids):] for n,ids in packed.items()}})
    return result


@contextmanager
def capture_first_response(runtime, forced):
    first = []

    def hook(_module,_args,kwargs,out):
        if first:
            return
        ids = kwargs.get("input_ids")
        if forced and (ids is None or ids.shape[1]!=1 or not bool((ids==THINK_END).all())):
            return
        first.append(out.logits[:,-1].float().detach().clone())

    handle = runtime.model.register_forward_hook(hook,with_kwargs=True)
    try:
        yield first
    finally:
        handle.remove()
    if len(first)!=1:
        raise ValueError("could not read first response logits")


def run(a):
    path = a.root/"cohort.json"
    plan = load(path)
    canary = a.phase=="smoke"
    rows = plan["canary"] if canary else plan["cases"][a.part::a.partitions]
    cap = 64 if canary else plan["cap"]
    r = Runtime(a)
    items = make_inputs(r.tokenizer,rows)
    with a.output.open("x") as out,r.torch.inference_mode():
        def emit(row):
            print(json.dumps(row,allow_nan=False),file=out,flush=True)
        emit({"kind":"manifest","phase":a.phase,"draw":a.draw,"part":a.part,"cap":cap,
              "ids":[x["id"] for x in rows],"cohort_sha256":sha(path),"scaffold_worker_sha256":sha(Path(__file__)),
              "arms":ARM_FORMAT,"initial_presence_ids":[THINK_END],"rng":"post_seed at first answer token",**r.manifest})
        for start in range(0,len(items),BATCH):
            if time.time() >= a.deadline-120:
                raise TimeoutError("scaffold queue reached its deadline")
            group = items[start:start+BATCH]
            width = max(len(ids) for x in group for ids in x["prefixes"].values())
            outcomes = {}
            for arm in ARM_FORMAT:
                pack = pack_at(r,[x["prefixes"][arm] for x in group],width)
                probe = capture_first_response(r,False) if canary and arm=="1+0" else nullcontext()
                with probe as raw:
                    outcomes[arm] = r.generate(group,"no_think",a.draw,cap,packed=pack,start_after_marker=True)
                if raw is not None:
                    prefilled_logits = raw[0]
                for row,item in zip(outcomes[arm],group,strict=True):
                    emit(row | {"kind":"rollout","arm":arm,"prefix_sha256":packed_hash(item["prefixes"][arm]),
                                "scaffold_tokens":item["tails"][arm]})
            if canary:
                reference = [x | {"pending":True,"presence_seed_ids":[]} for x in group]
                pack = pack_at(r,[x["think_prompt"] for x in group],width-1)
                with capture_first_response(r,True) as logits:
                    forced = r.generate(reference,"e0",a.draw,cap,packed=pack)
                tolerance = (logits[0]-prefilled_logits).abs().amax(dim=-1).tolist()
                exact = [tokens(f)[1:]==tokens(v) for f,v in zip(forced,outcomes["1+0"],strict=True)]
                emit({"kind":"canary","matched_input_ids":len(group),"first_answer_logit_max_abs":tolerance,
                      "exact_post_token_trace":exact})
            print(f"scaffold {a.phase} draw={a.draw} prompts={start+len(group)}/{len(items)}",flush=True)
        emit({"kind":"complete"})


def report(root, tokenizer):
    cohort = root/"cohort.json"
    plan = load(cohort)
    prompts = {x["id"]:x for x in plan["cases"]}
    [turn] = tokenizer.encode("<|im_end|>",add_special_tokens=False)
    raw,receipts = {},[]
    done = sorted((root/"queue").glob("draw*-done.json"))
    for receipt in done:
        for filename,expected in load(receipt)["outputs"]:
            p = Path(filename)
            if sha(p)!=expected:
                raise ValueError("scaffold shard checksum mismatch")
            with p.open() as f:
                head,*rows,footer = map(json.loads,f)
            if (head["cohort_sha256"]!=sha(cohort) or head["phase"]!="eval" or head["cap"]!=CAP
                    or footer["kind"]!="complete" or head["initial_presence_ids"]!=[THINK_END]):
                raise ValueError("scaffold rollout contract changed")
            for row in rows:
                p,d,arm = row["id"],row["draw"],row["arm"]
                key = p,d,arm
                if (row["kind"]!="rollout" or key in raw or p not in head["ids"] or d!=head["draw"]
                        or row["seed"]!=post_seed(p,d) or row["post_seed"]!=row["seed"] or len(tokens(row))>CAP
                        or (row["eos_token"] is not None and row["eos_token"] not in head["eos_ids"])):
                    raise ValueError("scaffold trajectory lost its matching seed or budget")
                raw[key] = row
            receipts.append([filename,expected])
    if not done or set(raw)!={(p,d,a) for p in prompts for d in range(len(done)) for a in ARM_FORMAT}:
        raise ValueError("incomplete scaffold factorial")
    windows = {}
    for horizon in (256,512,CAP):
        y,boxes = {},{}
        for k,row in raw.items():
            ids,eos = cut(row,"total",horizon)
            outcome = row | {"tokens":ids,"eos_token":row["eos_token"] if eos else None,"arm":"no_think"}
            y[k] = score_policy(prompts[k[0]],outcome,tokenizer,turn)
            boxes[k] = first_box(ids[:ids.index(turn)] if turn in ids else ids,tokenizer)
        table = {}
        for arm in ARM_FORMAT:
            coords = [k for k in y if k[2]==arm]
            hits = [boxes[k] for k in coords if boxes[k] is not None]
            table[arm] = {"n":len(coords),**{m:sum(y[k][m] for k in coords) for m in
                          ("correct_eos","eos","wrong_eos","missing_eos","cap","visible_boxed")},
                          "first_box_n":len(hits),"first_box_median":statistics.median(hits) if hits else None}
        effects = {name:paired_interval({
            p:[sum(w*int(y[p,d,a]["correct_eos"]) for a,w in effect.items()) for d in range(len(done))]
            for p in prompts
        }) for name,effect in EFFECTS.items()}
        windows[str(horizon)] = {"arms":table,"effects":effects}
    tails = {a:sorted({tuple(r["scaffold_tokens"]) for k,r in raw.items() if k[2]==a}) for a in ARM_FORMAT}
    return {"prompts":len(prompts),"draws":len(done),"windows":windows,"scaffold_token_ids":tails,"source_shards":receipts}


class ScaffoldQueue(Supervisor):
    def __init__(self,args):
        args.role,args.output = "queue",args.root
        super().__init__(args)
        self.worker = Path(__file__).resolve()
        self.env["PYTHONPATH"] = args.runtime_path

    def plan(self,i,phase,draw,parts):
        a = self.args
        out = self.lane/f"{phase}-{draw}-{i}.jsonl"
        return i,out,[self.worker,"run","--phase",phase,"--root",self.root,"--model",a.model,"--data",a.data,
                      "--output",out,"--draw",draw,"--part",i,"--partitions",parts,"--deadline",a.deadline]

    def supervise(self):
        try:
            self.status(cohort_sha256=sha(self.root/"cohort.json"))
            (canary,checksum), = self.run_batch("canary",[self.plan(0,"smoke",0,1)])
            with canary.open() as f:
                h,*rows,end = map(json.loads,f)
            checked = [a for a in rows if a["kind"]=="canary"]
            if (h["phase"]!="smoke" or end["kind"]!="complete" or len(checked)!=1
                    or checked[0]["matched_input_ids"]!=BATCH or len(h["ids"])!=BATCH):
                raise ValueError("scaffold canary did not cover the fixed matching check")
            self.status("canary_passed",canary_sha256=checksum,canary=checked[0])
            completed,reserve = 0,900
            for draw in range(2):
                if not self.bounded(reserve):
                    break
                start = time.monotonic()
                outputs = self.run_batch(f"draw{draw}",[self.plan(i,"eval",draw,2) for i in range(2)])
                save(self.lane/f"draw{draw}-done.json",{"outputs":[[str(p),h] for p,h in outputs]})
                completed += 1
                reserve = 1.35*(time.monotonic()-start)+120
            if completed:
                target = self.root/"summary.json"
                argv = [sys.executable,"-B",self.worker,"score","--root",self.root,"--model",self.args.model,"--output",target]
                subprocess.run(argv,env={**self.env,"CUDA_VISIBLE_DEVICES":""},check=True,timeout=150)
                self.status("completed" if completed==2 else "budget_complete",summary_sha256=sha(target),completed_eval_draws=completed)
            else:
                self.status("budget_complete",completed_eval_draws=0)
        except Exception as exc:
            self.status("failed",error=str(exc))
            raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("run","score","launch","supervise"))
    p.add_argument("--phase",choices=("smoke","eval"))
    for name in ("root","model","data","output","overlay","stable","source"):
        p.add_argument("--"+name,type=Path)
    for name in ("runtime-path","devices","host","cohort-sha256"):
        p.add_argument("--"+name)
    for name,default in (("part",0),("partitions",1),("draw",0),("deadline",None)):
        p.add_argument("--"+name,type=int,default=default)
    a=p.parse_args()
    if a.action=="run":
        run(a)
        return
    if a.action=="score":
        from transformers import AutoTokenizer
        out=report(a.root,AutoTokenizer.from_pretrained(a.model,local_files_only=True))
        if a.output.exists():
            raise FileExistsError(a.output)
        save(a.output,out)
        return
    q=ScaffoldQueue(a)
    if socket.gethostname()!=a.host or len(q.devices)!=2 or len(set(q.devices))!=2 or not q.bounded(1200):
        raise ValueError("wrong host, device assignment, or scaffold budget")
    if (sha(a.root/"cohort.json")!=a.cohort_sha256
            or any(sha(q.worker.parent/n)!=h for n,h in load(a.source).items())):
        raise ValueError("scaffold input or source changed")
    if a.action=="supervise":
        q.supervise()
        return
    check_devices(q.devices)
    q.lane.mkdir()
    with (q.lane/"supervisor.log").open("x") as log:
        process=subprocess.Popen([sys.executable,"-B",__file__,"supervise",*sys.argv[2:]],
                                 env=q.env,start_new_session=True,stdout=log,stderr=subprocess.STDOUT)
    print(json.dumps({"supervisor":process.pid,"root":str(q.root),"deadline":a.deadline,"devices":q.devices,
                      "source_sha256":sha(a.source)}))


if __name__=="__main__":
    main()
