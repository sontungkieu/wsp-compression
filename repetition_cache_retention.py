"""Test whether a cache swap preserves information available only in the scratchpad."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

from repetition_baseline import astra_instruction, math_answer, render, sha
from repetition_budget_run import packed_hash, tokens
from repetition_cache_factorial import ARMS, BOTH, CONV, DROP, MATRIX, NATIVE, sample
from repetition_canonical import paired_interval, seed_for
from repetition_mediation_queue import Supervisor, check_devices, load, save
from repetition_state_data import keyed
from repetition_state_value import Runtime, THINK_END, inputs_for, post_seed

TASKS = ("copy","math")
CONTENTS = ("plus","minus")
CAP, BLOCK, EVAL = 192, 4, 24


def candidates(task):
    if task=="copy":
        return [a+b+c for a,b,c in itertools.product("mnpbrst","aeiou","lnmrkst")]
    return [str(i) for i in range(11,96)]


def question(task, ident):
    if task=="copy":
        return {"id":ident,"task":"copy","prompt":"Read the lowercase word on the checkpoint line in your scratchpad. "
                "Write exactly that word three times separated by a single space. If there is no checkpoint line, write UNKNOWN."}
    return {"id":ident,"task":"math","prompt":"Read the integer on the checkpoint line in your scratchpad. "
            "Add 7 to that integer, then double the result. If the checkpoint line is missing, write UNKNOWN.\n"
            +astra_instruction("canonical"),"astra_prompt_arm":"canonical","boxed_answer_required":True}


def content_pair(tokenizer, task, ident, values):
    row = question(task,ident)
    raw = render(tokenizer,row,"think")
    base = tokenizer.encode(raw,add_special_tokens=False)
    variants = {}
    for name,value in zip(CONTENTS,values,strict=True):
        draft = f"Draft record.\ncheckpoint: {value}\nUse the checkpoint entry above for the requested answer."
        with_draft = tokenizer.encode(raw+draft,add_special_tokens=False)
        if with_draft[:len(base)]!=base or value in row["prompt"]:
            return None
        target = " ".join([value]*3) if task=="copy" else str((int(value)+7)*2)
        variants[name] = {"row":row | {"gold":target},"completion_prefix":with_draft[len(base):],
                          "value":value,"render_sha256":hashlib.sha256(raw.encode()).hexdigest()}
    sv = [variants[c]["completion_prefix"] for c in CONTENTS]
    if len(sv[0])!=len(sv[1]) or not sv[0] or THINK_END in sv[0] or THINK_END in sv[1]:
        return None
    locations = [i for i,(a,b) in enumerate(zip(*sv,strict=True)) if a!=b]
    if not locations or locations!=list(range(locations[0],locations[-1]+1)):
        return None
    return {"id":ident,"task":task,"scaffold_changed_indices":locations,"variants":variants}


def freeze(tokenizer, output):
    panels = {}
    for task in TASKS:
        # Tokenization alone determines eligibility. Evaluation values are unique within each task.
        ordered = sorted(candidates(task),key=lambda x:keyed(x,"cache-retention-v1:"+task))
        pending = []
        examples = []
        for value in ordered:
            match = None
            for a in pending:
                ident = f"cache-retention:{task}:{len(examples)}"
                item = content_pair(tokenizer,task,ident,(a,value))
                if item is not None:
                    match = a
                    examples.append(item)
                    break
            if match is None:
                pending.append(value)
            else:
                pending.remove(match)
            if len(examples)==EVAL+4:
                break
        if len(examples)!=EVAL+4:
            raise ValueError("insufficient matched tokenizations for the frozen synthetic cohort")
        panels[task] = {"canary":examples[:4],"eval":examples[4:]}
    if output.exists():
        raise FileExistsError(output)
    save(output,{"panels":panels,"cap":CAP,"scope":"controlled synthetic information only present in the scratchpad"})


def run(a):
    cohort=a.root/"cohort.json"
    plan=load(cohort)
    canary=a.phase=="smoke"
    r=Runtime(a)
    chosen=TASKS if canary else (a.task,)
    with a.output.open("x") as f,r.torch.inference_mode():
        def emit(row):
            print(json.dumps(row,allow_nan=False),file=f,flush=True)
        emit({"kind":"manifest","phase":a.phase,"tasks":chosen,"draw":a.draw,"cap":CAP,
              "cohort_sha256":sha(cohort),"worker_sha256_retention":sha(Path(__file__)),**r.manifest})
        for task in chosen:
            panel=plan["panels"][task]["canary" if canary else "eval"]
            for i in range(0,len(panel),BLOCK):
                if time.time()>=a.deadline-120:
                    raise TimeoutError("cache retention queue reached its deadline")
                block=panel[i:i+BLOCK]
                batch=inputs_for(r.tokenizer,[x["variants"][c] for x in block for c in CONTENTS])
                trajectories,audits=sample(r,batch,a.draw,CAP,canary)
                drop=trajectories[DROP]
                if any(tokens(drop[j])!=tokens(drop[j+1]) for j in range(0,len(drop),2)):
                    raise ValueError("question-only reference differs across identical questions and post seeds")
                emit({"kind":"audit","task":task,"ids":[x["id"] for x in block],
                      "paired_content_matches_question_only":len(block),"arms":audits})
                for arm,outcomes in trajectories.items():
                    for j,(row,x) in enumerate(zip(outcomes,batch,strict=True)):
                        emit(row | {"arm":arm,"task":task,"content":CONTENTS[j%2],
                                    "scratchpad_sha256":packed_hash(x["completion_prefix"])})
                print(f"retention {a.phase} {task} draw={a.draw} prompts={i+len(block)}/{len(panel)}",flush=True)
        emit({"kind":"complete"})


def correctness(example, row, tokenizer):
    if example["task"]=="copy":
        answer = tokenizer.decode(row["tokens"][1:],skip_special_tokens=False)
        value = answer.strip()
        correct = value==example["gold"]
        expected_word = example["gold"].split()[0]
        mentions_value = re.search(r"\b"+re.escape(expected_word)+r"\b",answer) is not None
    else:
        text = tokenizer.decode(row["tokens"],skip_special_tokens=False)
        answer = text.rpartition("</think>")[-1]
        assessed = math_answer(example,text)
        value,correct = assessed["parsed_answer"],assessed["numeric_correct"]
    eos = row["eos_token"] is not None
    assessed = {"correct":correct,"correct_eos":correct and eos,"eos":eos,
                "abstained":answer.strip().upper()=="UNKNOWN","value":value}
    if example["task"]=="copy":
        assessed["mentions_value_eos"] = mentions_value and eos
    return assessed


def read_receipts(root,paths,phase,tokenizer):
    cohort=root/"cohort.json"
    plan=load(cohort)
    expected={x["id"]:x for task in TASKS for x in plan["panels"][task]["canary" if phase=="smoke" else "eval"]}
    raw,proof,audit=[],[],[]
    for filename,fingerprint in paths:
        path=Path(filename)
        if sha(path)!=fingerprint:
            raise ValueError("retention raw shard checksum mismatch")
        with path.open() as f:
            header,*body,end=map(json.loads,f)
        if (header["phase"]!=phase or header["cap"]!=CAP or header["cohort_sha256"]!=sha(cohort)
                or end["kind"]!="complete"):
            raise ValueError("retention worker protocol differs")
        for row in body:
            if row["kind"]=="audit":
                audit.append(row)
                continue
            p,d,arm,c=row["id"],row["draw"],row["arm"],row["content"]
            item=expected[p]["variants"][c]
            eos=header["copy_eos_ids"] if row["task"]=="copy" else header["eos_ids"]
            if (row["kind"]!="rollout" or expected[p]["task"]!=row["task"] or row["task"] not in header["tasks"]
                    or d!=header["draw"] or row["seed"]!=seed_for(p,d) or row["post_seed"]!=post_seed(p,d)
                    or row["tokens"][0]!=THINK_END or len(tokens(row))>CAP+1
                    or row["scratchpad_sha256"]!=packed_hash(item["completion_prefix"])
                    or (row["eos_token"] is not None and row["eos_token"] not in eos)):
                raise ValueError("retention trajectory changed its protocol")
            raw.append(((p,d,arm,c),correctness(item["row"],row,tokenizer)))
        proof.append([filename,fingerprint])
    cells=dict(raw)
    arms=(*ARMS,"self_copy") if phase=="smoke" else ARMS
    draws=len({k[1] for k in cells})
    if len(cells)!=len(raw) or set(cells)!={(p,d,a,c) for p in expected for d in range(draws) for a in arms for c in CONTENTS}:
        raise ValueError("incomplete paired synthetic retention matrix")
    return expected,cells,arms,draws,proof,audit


def report(root,paths,phase,tokenizer):
    places,cells,arms,draws,proof,audits=read_receipts(root,paths,phase,tokenizer)
    output={}
    for task in TASKS:
        ids=[p for p,row in places.items() if row["task"]==task]
        table={}
        def pair(p,d,a):
            return all(cells[p,d,a,c]["correct_eos"] for c in CONTENTS)
        for arm in arms:
            group=[cells[p,d,arm,c] for p in ids for d in range(draws) for c in CONTENTS]
            both=sum(pair(p,d,arm) for p in ids for d in range(draws))
            changed=sum(cells[p,d,arm,"plus"]["value"]!=cells[p,d,arm,"minus"]["value"] for p in ids for d in range(draws))
            table[arm]={"trajectories":len(group),"pairs":len(ids)*draws,"both_values_correct_eos":both,
                        "prediction_changed":changed,**{m:sum(r[m] for r in group) for m in
                                                       ("correct","correct_eos","eos","abstained")}}
            if task=="copy":
                table[arm]["mentions_value_eos"] = sum(r["mentions_value_eos"] for r in group)
            table[arm]["paired_both_values_minus_raw"]=paired_interval({
                p:[int(pair(p,d,arm))-int(pair(p,d,NATIVE)) for d in range(draws)] for p in ids})
        output[task]={"prompts":len(ids),"cells":table}
    return {"draws":draws,"tasks":output,"verified_shards":proof,
            "matched_question_only_pairs_by_batch":[len(a["ids"]) for a in audits]}


class RetentionQueue(Supervisor):
    def __init__(self,args):
        args.role,args.output="queue",args.root
        super().__init__(args)
        self.worker=Path(__file__).resolve()
        self.env["PYTHONPATH"]=args.runtime_path

    def plan(self,i,phase,draw,task):
        a=self.args
        out=self.lane/f"{phase}-{draw}-{task}.jsonl"
        params=[self.worker,"run","--phase",phase,"--root",self.root,"--model",a.model,"--data",a.data,
                "--output",out,"--task",task,"--draw",draw,"--deadline",a.deadline]
        return i,out,params

    def score(self,phase):
        target=self.root/f"{phase}-summary.json"
        argv=[sys.executable,"-B",self.worker,"score","--phase",phase,"--root",self.root,
              "--model",self.args.model,"--output",target]
        subprocess.run(argv,env={**self.env,"CUDA_VISIBLE_DEVICES":""},check=True,timeout=140)
        return target,load(target)

    def supervise(self):
        try:
            self.status(cohort_sha256=self.args.cohort_sha256)
            if self.args.canary_receipt is not None:
                receipt = load(self.args.canary_receipt)
                self.status("reusing_canary",canary_receipt_sha256=self.args.canary_receipt_sha256)
            else:
                (out,fingerprint),=self.run_batch("canary",[self.plan(0,"smoke",0,"copy")])
                receipt = {"outputs":[[str(out),fingerprint]]}
            save(self.lane/"smoke-done.json",receipt)
            canary,cells=self.score("smoke")
            for task in TASKS:
                arms=cells["tasks"][task]["cells"]
                if arms[NATIVE]["correct_eos"]<=arms[DROP]["correct_eos"]:
                    self.status("canary_floor",canary_sha256=sha(canary),floor_task=task)
                    return
            self.status("canary_passed",canary_sha256=sha(canary))
            completed,reserve=0,900
            for draw in range(2):
                if not self.bounded(reserve):
                    break
                start=time.monotonic()
                values=self.run_batch(f"draw{draw}",[self.plan(i,"eval",draw,task) for i,task in enumerate(TASKS)])
                save(self.lane/f"draw{draw}-done.json",{"outputs":[[str(p),h] for p,h in values]})
                completed+=1
                reserve=1.35*(time.monotonic()-start)+120
            if completed:
                score,_=self.score("eval")
                self.status("completed" if completed==2 else "budget_complete",
                            summary_sha256=sha(score),completed_eval_draws=completed)
            else:
                self.status("budget_complete",completed_eval_draws=0)
        except Exception as exc:
            self.status("failed",error=str(exc))
            raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("run","score","launch","supervise"))
    p.add_argument("--phase",choices=("smoke","eval"))
    p.add_argument("--task",choices=TASKS)
    for name in ("root","model","data","output","overlay","stable","source","canary-receipt"):
        p.add_argument("--"+name,type=Path)
    for name in ("runtime-path","devices","host","cohort-sha256","canary-receipt-sha256"):
        p.add_argument("--"+name)
    p.add_argument("--draw",type=int,default=0)
    p.add_argument("--deadline",type=int)
    a=p.parse_args()
    if a.action=="run":
        run(a)
        return
    if a.action=="score":
        from transformers import AutoTokenizer
        manifest=[a.root/"queue"/"smoke-done.json"] if a.phase=="smoke" else sorted((a.root/"queue").glob("draw*-done.json"))
        paths=[pair for file in manifest for pair in load(file)["outputs"]]
        report_data=report(a.root,paths,a.phase,AutoTokenizer.from_pretrained(a.model,local_files_only=True))
        if a.output.exists():
            raise FileExistsError(a.output)
        save(a.output,report_data)
        return
    q=RetentionQueue(a)
    if socket.gethostname()!=a.host or len(q.devices)!=2 or len(set(q.devices))!=2 or not q.bounded(1200):
        raise ValueError("wrong retention node, assigned GPUs or budget")
    if (sha(a.root/"cohort.json")!=a.cohort_sha256
            or any(sha(q.worker.parent/n)!=h for n,h in load(a.source).items())):
        raise ValueError("retention source or cohort hash changed")
    if a.canary_receipt is not None and sha(a.canary_receipt)!=a.canary_receipt_sha256:
        raise ValueError("the retained canary receipt changed")
    if a.action=="supervise":
        q.supervise()
        return
    check_devices(q.devices)
    q.lane.mkdir()
    with (q.lane/"supervisor.log").open("x") as log:
        process=subprocess.Popen([sys.executable,"-B",__file__,"supervise",*sys.argv[2:]],env=q.env,
                                 start_new_session=True,stdout=log,stderr=subprocess.STDOUT)
    print(json.dumps({"supervisor":process.pid,"root":str(a.root),"deadline":a.deadline,"devices":q.devices,
                      "source_sha256":sha(a.source)}))


if __name__=="__main__":
    main()
