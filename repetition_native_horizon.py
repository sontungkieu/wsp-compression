"""Observe native Qwen3.5 reasoning across nested 2K to 32K token horizons."""
import argparse
from decimal import Decimal
import json
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time

from repetition_baseline import BOXED, DECIMAL, first_recurrent, sha
from repetition_budget_run import tokens
from repetition_budget_score import cut
from repetition_canonical import seed_for
from repetition_mediation_queue import Supervisor, check_devices, load, save
from repetition_phase_trace import before_marker, first_long_cycle
from repetition_replay import terminal_cycle
from repetition_restart import score_policy
from repetition_state_data import keyed
from repetition_state_value import Runtime, THINK_END, inputs_for, post_seed

HORIZONS = (2048,4096,8192,16384,32768)
PILOT = 4096


def freeze(subset, parent, output):
    cases = sorted(load(subset)["cases"],key=lambda x:keyed(x["row"]["id"],"native-long-horizon-v1"))[:12]
    dev = load(parent)["pilot"][:2]
    if len(cases)!=12 or len(dev)!=2 or output.exists() or {x["row"]["id"] for x in cases}&{x["row"]["id"] for x in dev}:
        raise ValueError("horizon evaluation requires 12 fixed questions and a separate timing pilot")
    save(output,{"cases":[x["row"] for x in cases],"pilot":[x["row"] for x in dev],
                 "source_cohort_sha256":sha(subset),"pilot_source_cohort_sha256":sha(parent),
                 "horizons":HORIZONS,"draw":0})


def run(a):
    cohort = a.root/"cohort.json"
    plan = load(cohort)
    rows = plan["pilot"] if a.phase=="timing" else plan["cases"][a.part::a.partitions]
    if not rows or (a.phase=="timing" and a.cap!=PILOT) or (a.phase=="rollout" and a.cap not in HORIZONS[-2:]):
        raise ValueError("unexpected horizon size or missing worker shard")
    r = Runtime(a)
    items = inputs_for(r.tokenizer,[{"row":row,"completion_prefix":[]} for row in rows])
    part = a.output.with_suffix(".partial.jsonl")
    if a.output.exists():
        raise FileExistsError(a.output)
    with part.open("x") as out,r.torch.inference_mode():
        def emit(row):
            print(json.dumps(row,allow_nan=False),file=out,flush=True)
        emit({"kind":"manifest","phase":a.phase,"part":a.part,"cap":a.cap,"ids":[x["id"] for x in rows],
              "cohort_sha256":sha(cohort),"worker_sha256_horizon":sha(Path(__file__)),**r.manifest})
        started = time.monotonic()
        records = r.generate(items,"native",plan["draw"],a.cap)
        r.torch.cuda.synchronize()
        elapsed = time.monotonic()-started
        for rec,item in zip(records,items,strict=True):
            emit(rec | {"render_sha256":item["render_sha256"]})
        emit({"kind":"complete","seconds":elapsed,"longest_tokens":max(len(tokens(r)) for r in records)})
    part.replace(a.output)


def first_correct_box(ids, tokenizer, gold):
    def has_gold(prefix):
        s = tokenizer.decode(prefix,skip_special_tokens=False)
        for box in BOXED.finditer(s):
            value = box.group(1).strip()
            if DECIMAL.fullmatch(value) and Decimal(value.replace(",",""))==gold:
                return True
        return False
    if not has_gold(ids):
        return None
    lo,hi = 1,len(ids)
    while lo<hi:
        mid = (lo+hi)//2
        if has_gold(ids[:mid]):
            hi = mid
        else:
            lo = mid+1
    return lo


def summarize(root, tokenizer):
    cohort = root/"cohort.json"
    plan, settings = load(cohort),load(root/"selected-cap.json")
    rows = {r["id"]:r for r in plan["cases"]}
    records,receipts = {},[]
    [marker] = tokenizer.encode("</think>",add_special_tokens=False)
    [turn] = tokenizer.encode("<|im_end|>",add_special_tokens=False)
    for file,fingerprint in load(root/"queue"/"rollout-done.json")["outputs"]:
        path = Path(file)
        if sha(path)!=fingerprint:
            raise ValueError("native long-rollout checksum mismatch")
        with path.open() as f:
            head,*body,end = map(json.loads,f)
        if (head["cohort_sha256"]!=sha(cohort) or head["phase"]!="rollout" or head["cap"]!=settings["cap"]
                or end["kind"]!="complete"):
            raise ValueError("unexpected horizon protocol")
        for record in body:
            p,draw = record["id"],record["draw"]
            if (record["kind"]!="rollout" or p in records or p not in head["ids"] or draw!=plan["draw"]
                    or record["seed"]!=seed_for(p,draw) or record["post_seed"]!=post_seed(p,draw)
                    or len(tokens(record))>settings["cap"] or
                    (record["eos_token"] is not None and record["eos_token"] not in head["eos_ids"])):
                raise ValueError("unexpected native long-rollout record")
            records[p]=record
        receipts.append([file,fingerprint])
    if records.keys()!=rows.keys():
        raise ValueError("missing native long-rollout questions")
    events = {}
    for p,r in records.items():
        ids = r["tokens"]
        before,closed,_ = before_marker(r,marker,settings["cap"])
        close = len(before)+1 if closed else None
        answer = ids[close:] if closed else []
        if turn in answer:
            answer = answer[:answer.index(turn)]
        first = first_correct_box(answer,tokenizer,Decimal(rows[p]["gold"]))
        events[p] = {"first_close_token":close,"eos_token_position":len(ids)+1 if r["eos_token"] is not None else None,
                     "first_exact_cycle_before_close":first_long_cycle(before),"first_recurrent16_before_close":first_recurrent(before),
                     "first_correct_box_token":close+first if first is not None else None}
    windows = {}
    for budget in (n for n in HORIZONS if n<=settings["cap"]):
        outcomes = []
        for p,r in records.items():
            ids,eos = cut(r,"total",budget)
            y = score_policy(rows[p],r | {"tokens":ids,"eos_token":r["eos_token"] if eos else None},tokenizer,turn)
            before,closed,_ = before_marker(r,marker,budget)
            v = events[p]
            def observed(name):
                return v[name] is not None and v[name]<=budget
            outcomes.append(y | {"closed":closed,"unfinished_thinking":not closed and not eos,
                                 "unfinished_answering":closed and not eos,
                                 "first_exact_cycle_before_close":observed("first_exact_cycle_before_close"),
                                 "first_recurrent16_before_close":observed("first_recurrent16_before_close"),
                                 "terminal_cycle":terminal_cycle(ids) is not None,
                                 "terminal_cycle_before_close":terminal_cycle(before) is not None,
                                 "ever_correct_box_after_close":observed("first_correct_box_token"),
                                 "correct_without_eos":y["numeric_correct"] and not eos})
        windows[str(budget)] = {"n":len(outcomes),**{k:sum(bool(y[k]) for y in outcomes) for k in outcomes[0] if k!="switched"}}
    return {"selected_cap":settings,"windows":windows,"events_by_prompt":events,"verified_shards":receipts}


class HorizonQueue(Supervisor):
    def __init__(self,args):
        args.role,args.output = "queue",args.root
        super().__init__(args)
        self.worker = Path(__file__).resolve()
        self.env["PYTHONPATH"] = args.runtime_path

    def plan(self,i,phase,cap,parts):
        a = self.args
        out = self.lane/f"{phase}-{i}.jsonl"
        return i,out,[self.worker,"run","--phase",phase,"--root",self.root,"--model",a.model,"--data",a.data,
                      "--output",out,"--cap",cap,"--part",i,"--partitions",parts,"--deadline",a.deadline]

    def supervise(self):
        try:
            self.status(cohort_sha256=self.args.cohort_sha256)
            (canary,fingerprint), = self.run_batch("timing",[self.plan(0,"timing",PILOT,1)])
            with canary.open() as f:
                head,*rows,tail = map(json.loads,f)
            if (head["cohort_sha256"]!=self.args.cohort_sha256 or head["cap"]!=PILOT or len(rows)!=2
                    or tail["kind"]!="complete" or not 0<tail["longest_tokens"]<=PILOT):
                raise ValueError("timing did not complete on its two fixed questions")
            per_token = max(.03,tail["seconds"]/tail["longest_tokens"])
            estimates = {cap:1.75*per_token*cap+300 for cap in HORIZONS[-2:]}
            eligible = [cap for cap,eta in estimates.items() if self.bounded(max(2700,eta))]
            if not eligible:
                self.status("budget_complete",timing_sha256=fingerprint)
                return
            cap = max(eligible)
            save(self.root/"selected-cap.json",{"cap":cap,"timing_sha256":fingerprint,
                                                 "pilot_seconds":tail["seconds"],"pilot_longest_tokens":tail["longest_tokens"],
                                                 "conservative_wall_seconds":estimates})
            self.status("horizon_frozen",cap=cap,timing_sha256=fingerprint)
            results = self.run_batch("rollout",[self.plan(i,"rollout",cap,2) for i in range(2)])
            save(self.lane/"rollout-done.json",{"outputs":[[str(p),h] for p,h in results]})
            output = self.root/"summary.json"
            cmd = [sys.executable,"-B",self.worker,"score","--root",self.root,"--model",self.args.model,"--output",output]
            subprocess.run(cmd,env={**self.env,"CUDA_VISIBLE_DEVICES":""},check=True,timeout=150)
            self.status("completed",summary_sha256=sha(output))
        except Exception as exc:
            self.status("failed",error=str(exc))
            raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("run","score","launch","supervise"))
    p.add_argument("--phase",choices=("timing","rollout"))
    for name in ("root","model","data","output","overlay","stable","source"):
        p.add_argument("--"+name,type=Path)
    for name in ("runtime-path","devices","host","cohort-sha256"):
        p.add_argument("--"+name)
    for name,default in (("part",0),("partitions",1),("cap",None),("deadline",None)):
        p.add_argument("--"+name,type=int,default=default)
    a=p.parse_args()
    if a.action=="run":
        run(a)
        return
    if a.action=="score":
        from transformers import AutoTokenizer
        result = summarize(a.root,AutoTokenizer.from_pretrained(a.model,local_files_only=True))
        if a.output.exists():
            raise FileExistsError(a.output)
        save(a.output,result)
        return
    q=HorizonQueue(a)
    if socket.gethostname()!=a.host or len(q.devices)!=2 or len(set(q.devices))!=2 or not q.bounded(2700):
        raise ValueError("wrong horizon node, GPU list or insufficient wall time")
    if (sha(a.root/"cohort.json")!=a.cohort_sha256
            or any(sha(q.worker.parent/n)!=expected for n,expected in load(a.source).items())):
        raise ValueError("horizon cohort or sources changed")
    if a.action=="supervise":
        q.supervise()
        return
    check_devices(q.devices)
    q.lane.mkdir()
    with (q.lane/"supervisor.log").open("x") as log:
        child=subprocess.Popen([sys.executable,"-B",__file__,"supervise",*sys.argv[2:]],
                               env=q.env,start_new_session=True,stdout=log,stderr=subprocess.STDOUT)
    print(json.dumps({"supervisor":child.pid,"root":str(a.root),"deadline":a.deadline,"devices":q.devices,
                      "source_sha256":sha(a.source)}))


if __name__=="__main__":
    main()
