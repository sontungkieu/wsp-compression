"""Split convolution and recurrent matrix effects at native Qwen3.5 reasoning states."""
import argparse
import json
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time

from repetition_baseline import sha
from repetition_budget_run import tokens
from repetition_budget_score import assess_window, cut
from repetition_cache_swap import FIELDS, marker_hook, prefill_donor, state_digest
from repetition_cache_swap_score import first_box
from repetition_canonical import paired_interval, seed_for
from repetition_mediation_queue import Supervisor, check_devices, load, save
from repetition_restart import reaches_switch, score_policy
from repetition_restart_score import read_run
from repetition_sampler_reach import pack_at
from repetition_state_value import Runtime, THINK_END, inputs_for, post_seed

NATIVE, CONV, MATRIX, BOTH, DROP = ("native", "conv_only", "matrix_only", "both", "question_only")
ARMS = (NATIVE, CONV, MATRIX, BOTH, DROP)
SWAP = {CONV:(FIELDS[0],), MATRIX:(FIELDS[1],), BOTH:FIELDS}
SWITCH, CAP, BATCH = 512, 1535, 8
EFFECTS = {
    "both_minus_native":{BOTH:1,NATIVE:-1}, "matrix_minus_conv":{MATRIX:1,CONV:-1},
    "main_conv":{CONV:.5,NATIVE:-.5,BOTH:.5,MATRIX:-.5},
    "main_matrix":{MATRIX:.5,NATIVE:-.5,BOTH:.5,CONV:-.5},
    "conv_matrix_interaction":{BOTH:1,CONV:-1,MATRIX:-1,NATIVE:1},
    "both_minus_question_only":{BOTH:1,DROP:-1},
}


def freeze(parent, excluded, output):
    status = load(parent/"queue"/"status.json")
    if (status["stage"]!="completed" or status["completed_eval_draws"]!=2
            or sha(parent/"eval-summary.json")!=status["eval_summary_sha256"]):
        raise ValueError("factorial requires two complete parent draws")
    plan,prompts,raw,batches,receipts,draws = read_run(parent,"eval")
    _,dev,dev_raw,dev_batches,dev_receipts,dev_draws = read_run(parent,"pilot")
    if plan["switch"]!=SWITCH or plan["budget"]!=SWITCH+1+CAP or draws!=2 or dev_draws!=1:
        raise ValueError("parent budgets differ from the fixed factorial")
    reserved = {c["row"]["id"] for c in load(excluded)["cases"]}
    ids = sorted(prompts.keys()-reserved)
    if len(ids)!=32 or reserved-set(prompts) or set(dev)&set(prompts) or output.exists():
        raise ValueError("require 32 distinct IDs outside the depth and transfer panel")

    def inputs(rows, metadata, source, selected, draw):
        hashes = {p:h for b in metadata for p,h in zip(b["ids"],b["render_sha256"],strict=True)}
        batch = []
        for p in selected:
            trace = source[p,draw,NATIVE]
            if not reaches_switch(trace,SWITCH):
                raise ValueError("parent did not reach the fixed switch landmark")
            batch.append({"row":rows[p],"completion_prefix":trace["tokens"][:SWITCH],"render_sha256":hashes[p]})
        return batch

    pilot = [c["row"]["id"] for c in plan["pilot"][:BATCH]]
    save(output,{
        "draw_inputs":[inputs(prompts,batches,raw,ids,d) for d in range(draws)],
        "canary":inputs(dev,dev_batches,dev_raw,pilot,0),
        "parent":str(parent),"parent_cohort_sha256":sha(parent/"cohort.json"),"parent_summary_sha256":status["eval_summary_sha256"],
        "parent_eval_receipts":receipts,"parent_pilot_receipts":dev_receipts,"excluded_cohort_sha256":sha(excluded),
        "scope":"matched canonical MATH-500 numeric states reserved for the cache factorial",
        "post_cap":CAP,"spent_thinking":SWITCH,
    })


def sample(runtime, group, draw, cap, canary, *, swap_arms=tuple(SWAP), question_only=True):
    torch = runtime.torch
    group = [x | {"presence_seed_ids":[]} for x in group]
    question,donor = prefill_donor(runtime,group)
    packed = runtime.pack([x["native_ids"] for x in group])
    width = packed["input_ids"].shape[1]
    before = state_digest(donor,FIELDS,torch)
    traces,audits = {},{}
    arms = (NATIVE,*swap_arms,"self_copy") if canary else (NATIVE,*swap_arms)
    for arm in arms:
        mode = "native" if arm==NATIVE else "self" if arm=="self_copy" else "swap"
        with marker_hook(runtime,width,mode,donor,digest_kv=True,fields=SWAP.get(arm,FIELDS)) as evidence:
            traces[arm] = runtime.generate(group,"e0",draw,cap,packed=packed)
        audits[arm] = evidence
    if len({a["kv_sha256"] for a in audits.values()})!=1 or len({a["mask_sha256"] for a in audits.values()})!=1:
        raise ValueError("factorial receiver cache or mask differs before the intervention")
    if state_digest(donor,FIELDS,torch)!=before:
        raise ValueError("factorial altered the source donor tensors")
    if canary:
        if not torch.equal(audits[NATIVE]["logits"],audits["self_copy"]["logits"]) or any(
            tokens(a)!=tokens(b) for a,b in zip(traces[NATIVE],traces["self_copy"],strict=True)
        ):
            raise ValueError("factorial self-copy differs from the native trace")
    for arm in swap_arms:
        actual = audits[arm]["layers_changed"]
        if tuple(actual)!=SWAP[arm] or min(actual.values())<=0:
            raise ValueError(f"factorial swap did not execute as declared: {arm}")
        audits[arm]["max_logit_change"] = (audits[arm]["logits"]-audits[NATIVE]["logits"]).abs().amax(dim=-1).tolist()
    for data in audits.values():
        data.pop("logits")
    if question_only:
        traces[DROP] = runtime.generate(question,"e0",draw,cap,
                                        packed=pack_at(runtime,[x["native_ids"] for x in question],width))
    return traces,audits


def run(a):
    cohort = a.root/"cohort.json"
    plan = load(cohort)
    canary = a.phase=="smoke"
    cap = 64 if canary else plan["post_cap"]
    items = plan["canary"] if canary else plan["draw_inputs"][a.draw]
    runtime = Runtime(a)
    examples = inputs_for(runtime.tokenizer,items)
    with a.output.open("x") as out,runtime.torch.inference_mode():
        def emit(row):
            print(json.dumps(row,allow_nan=False),file=out,flush=True)
        emit({"kind":"manifest","phase":a.phase,"draw":a.draw,"cap":cap,"ids":[x["row"]["id"] for x in items],
              "cohort_sha256":sha(cohort),"source_sha256_factorial":sha(Path(__file__)),**runtime.manifest})
        for i in range(0,len(examples),BATCH):
            if time.time() >= a.deadline-120:
                raise TimeoutError("factorial queue is at its hard deadline")
            current = examples[i:i+BATCH]
            traces,audit = sample(runtime,current,a.draw,cap,canary)
            emit({"kind":"audit","ids":[x["row"]["id"] for x in current],"arms":audit})
            for arm,values in traces.items():
                for row in values:
                    emit(row | {"arm":arm})
            print(f"factorial {a.phase} draw={a.draw} prompts={i+len(current)}/{len(examples)}",flush=True)
        emit({"kind":"complete"})


def report(root, tokenizer):
    cohort = root/"cohort.json"
    plan = load(cohort)
    prompts = {x["row"]["id"]:x["row"] for x in plan["draw_inputs"][0]}
    parent = Path(plan["parent"])
    _,_,prior,_,parent_receipts,draws = read_run(parent,"eval")
    if parent_receipts!=plan["parent_eval_receipts"] or sha(parent/"cohort.json")!=plan["parent_cohort_sha256"]:
        raise ValueError("parent trace archive changed")
    [turn] = tokenizer.encode("<|im_end|>",add_special_tokens=False)
    rows,receipts = {},[]
    for filename,fingerprint in load(root/"queue"/"draws-done.json")["outputs"]:
        path = Path(filename)
        if sha(path)!=fingerprint:
            raise ValueError("factorial rollout shard checksum mismatch")
        with path.open() as f:
            head,*records,footer = map(json.loads,f)
        if (head["cohort_sha256"]!=sha(cohort) or head["cap"]!=CAP or head["phase"]!="eval"
                or set(head["ids"])!=set(prompts) or footer["kind"]!="complete"):
            raise ValueError("incorrect factorial rollout receipt")
        for row in records:
            if row["kind"]=="audit":
                continue
            p,d,arm = row["id"],row["draw"],row["arm"]
            if ((p,d,arm) in rows or d!=head["draw"] or row["tokens"][0]!=THINK_END
                    or row["seed"]!=seed_for(p,d) or row["post_seed"]!=post_seed(p,d)
                    or len(tokens(row))>CAP+1 or (row["eos_token"] is not None and row["eos_token"] not in head["eos_ids"])):
                raise ValueError("unexpected factorial trace")
            rows[p,d,arm] = row
        receipts.append([filename,fingerprint])
    if set(rows)!={(p,d,a) for p in prompts for d in range(draws) for a in ARMS}:
        raise ValueError("incomplete factorial outcomes")
    windows = {}
    outcomes = ("eos_correct","eos","wrong_parsed_eos","missing_answer_eos","unfinished",
                "correct_without_eos","terminal_cycle","visible_exact_format")
    for cap in (512,CAP):
        y = {k:assess_window(prompts[k[0]],r,"post",cap,tokenizer,turn) for k,r in rows.items()}
        arms = {}
        for arm in ARMS:
            data = [v for k,v in y.items() if k[2]==arm]
            answer_ids = []
            for k,row in rows.items():
                if k[2]!=arm:
                    continue
                ids,_ = cut(row,"post",cap)
                ids = ids[1:]
                answer_ids.append(ids[:ids.index(turn)] if turn in ids else ids)
            events = {"first_numeric_box": [v for ids in answer_ids if (v:=first_box(ids,tokenizer)) is not None],
                      "second_close": [ids.index(THINK_END)+1 for ids in answer_ids if THINK_END in ids]}
            arms[arm] = {"n":len(data),**{m:sum(r[m] for r in data) for m in outcomes},
                         **{name:{"n":len(v),"median":statistics.median(v) if v else None}
                            for name,v in events.items()}}
        effects = {name:paired_interval({
            p:[sum(w*int(y[p,d,arm]["eos_correct"]) for arm,w in contrast.items()) for d in range(draws)]
            for p in prompts
        }) for name,contrast in EFFECTS.items()}
        window = {"arms":arms,"effects":effects}
        if cap==CAP:
            direct = {(p,d):score_policy(prompts[p],prior[p,d,"no_think"],tokenizer,turn)
                      for p in prompts for d in range(draws)}
            window["parent_no_think"] = {m:sum(r[m] for r in direct.values()) for m in
                                         ("correct_eos","eos","wrong_eos","missing_eos","cap")}
            window["parent_no_think_minus_both"] = paired_interval({
                p:[int(direct[p,d]["correct_eos"])-int(y[p,d,BOTH]["eos_correct"]) for d in range(draws)] for p in prompts})
        windows[str(cap)] = window
    return {"n_prompts":len(prompts),"draws":draws,"source_shards":receipts,"parent_shards":parent_receipts,
            "scope":plan["scope"],"windows":windows}


class FactorialQueue(Supervisor):
    def __init__(self,args):
        args.role,args.output = "queue",args.root
        super().__init__(args)
        self.worker = Path(__file__).resolve()
        self.env["PYTHONPATH"] = args.runtime_path

    def plan(self,i,phase,draw):
        out = self.lane/f"{phase}-{draw}.jsonl"
        argv = [self.worker,"run","--phase",phase,"--root",self.root,"--model",self.args.model,"--data",self.args.data,
                "--output",out,"--draw",draw,"--deadline",self.args.deadline]
        return i,out,argv

    def supervise(self):
        try:
            self.status(cohort_sha256=self.args.cohort_sha256)
            (smoke,fingerprint), = self.run_batch("canary",[self.plan(0,"smoke",0)])
            with smoke.open() as f:
                head,*rows,footer = map(json.loads,f)
            audits = [a for a in rows if a["kind"]=="audit"]
            if (head["cap"]!=64 or head["cohort_sha256"]!=self.args.cohort_sha256 or len(head["ids"])!=BATCH
                    or footer["kind"]!="complete" or len(audits)!=1 or "self_copy" not in audits[0]["arms"]):
                raise ValueError("cache factorial canary incomplete")
            self.status("canary_passed",canary_sha256=fingerprint)
            if not self.bounded(1800):
                self.status("budget_complete")
                return
            results = self.run_batch("eval", [self.plan(i,"eval",i) for i in range(2)])
            save(self.lane/"draws-done.json",{"outputs":[[str(p),h] for p,h in results]})
            summary = self.root/"summary.json"
            argv = [sys.executable,"-B",self.worker,"score","--root",self.root,"--model",self.args.model,"--output",summary]
            subprocess.run(argv,env={**self.env,"CUDA_VISIBLE_DEVICES":""},check=True,timeout=150)
            self.status("completed",summary_sha256=sha(summary))
        except Exception as exc:
            self.status("failed",error=str(exc))
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("run","score","launch","supervise"))
    p.add_argument("--phase",choices=("smoke","eval"))
    for name in ("root","model","data","output","overlay","stable","source"):
        p.add_argument("--"+name,type=Path)
    for name in ("runtime-path","devices","host","cohort-sha256"):
        p.add_argument("--"+name)
    p.add_argument("--draw",type=int,default=0)
    p.add_argument("--deadline",type=int)
    a = p.parse_args()
    if a.action=="run":
        run(a)
        return
    if a.action=="score":
        from transformers import AutoTokenizer
        output = report(a.root,AutoTokenizer.from_pretrained(a.model,local_files_only=True))
        if a.output.exists():
            raise FileExistsError(a.output)
        save(a.output,output)
        return
    q = FactorialQueue(a)
    if socket.gethostname()!=a.host or len(q.devices)!=2 or len(set(q.devices))!=2 or not q.bounded(2100):
        raise ValueError("incorrect host, assigned devices or factorial budget")
    if (sha(a.root/"cohort.json")!=a.cohort_sha256
            or any(sha(q.worker.parent/n)!=h for n,h in load(a.source).items())):
        raise ValueError("factorial cohort or staged sources differ")
    if a.action=="supervise":
        q.supervise()
        return
    check_devices(q.devices)
    q.lane.mkdir()
    with (q.lane/"supervisor.log").open("x") as log:
        job = subprocess.Popen([sys.executable,"-B",__file__,"supervise",*sys.argv[2:]],
                               start_new_session=True,env=q.env,stdout=log,stderr=subprocess.STDOUT)
    print(json.dumps({"supervisor":job.pid,"root":str(q.root),"deadline":a.deadline,"devices":q.devices,
                      "source_sha256":sha(a.source)}))


if __name__=="__main__":
    main()
