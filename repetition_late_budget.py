"""Compare larger answer budgets at the frozen late Qwen3.5 landmarks."""
import argparse
from decimal import Decimal
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
import time

from repetition_baseline import render, sha
from repetition_budget_run import packed_hash, tokens
from repetition_budget_score import cut
from repetition_cache_factorial import BOTH, CONV, DROP, NATIVE, sample
from repetition_canonical import paired_interval, seed_for
from repetition_late_replay import CONFIRM, ends_in_motif
from repetition_mediation import SAMPLING, presence_scores
from repetition_mediation_queue import Supervisor, check_devices, load, save, smi
from repetition_native_horizon import first_correct_box
from repetition_replay import terminal_cycle
from repetition_restart import score_policy
from repetition_sampler_reach import pack_at
from repetition_state_value import Runtime, THINK_END, inputs_for, post_seed

CAPS, SMOKE, DRAWS, BLOCK = (512,1024,2048,4096),128,2,3
STATES = (CONFIRM,"open_16384")
DIRECT = "no_think"
ARMS = (NATIVE,CONV,BOTH,DROP,DIRECT)


def freeze(parent,output):
    summary, cohort = parent/"summary.json", parent/"cohort.json"
    receipt = load(parent/"queue/status.json")
    original = load(summary)
    if (output.exists() or receipt["stage"]!="completed" or sha(summary)!=receipt["summary_sha256"]
            or sha(cohort)!=original["cohort_sha256"]):
        raise ValueError("late-budget cohort requires verified original late replay")
    verified = original["verified_shards"]
    if any(sha(Path(p))!=h for p,h in verified):
        raise ValueError("original late replay traces changed")
    chosen = [c for c in load(cohort)["cases"] if c["state"] in STATES]
    if (sum(c["state"]==CONFIRM for c in chosen),sum(c["state"]=="open_16384" for c in chosen))!=(7,3):
        raise ValueError("late-budget cap sensitivity requires the original seven exact and three open cases")
    save(output,{"cases":chosen,"draws":DRAWS,"caps":CAPS,"source_summary_sha256":sha(summary),
                 "source_cohort_sha256":sha(cohort),"source_shards":verified,
                 "scope":"cap sensitivity on the already-observed late-replay panel"})


def probability_product(probabilities):
    total = math.fsum(math.log(p) for p in probabilities if p>0) if all(p>0 for p in probabilities) else None
    return {"product":math.exp(total) if total is not None else 0.,
            "log_probability":total,"min_step_probability":min(probabilities),
            "singleton_steps":sum(p==1 for p in probabilities),"support_steps":sum(p>0 for p in probabilities)}


def motif_probe(runtime,group):
    selected = [x for x in group if x["state"]==CONFIRM]
    if not selected:
        return []
    motifs = [x["cycle"]["motif"] for x in selected]
    if any(not motif or runtime.eos.intersection(motif) or THINK_END in motif for motif in motifs):
        raise ValueError("motif contains a stopping or closing token")
    sampled = [[] for _ in selected]
    tr,torch = runtime.transformers,runtime.torch
    top_k = tr.TopKLogitsWarper(top_k=SAMPLING["top_k"])
    top_p = tr.TopPLogitsWarper(top_p=SAMPLING["top_p"])
    base = runtime.policy

    class ForcedMotif(base):
        def __call__(self,ids,scores):
            history = ids[:,self.width:]
            step = history.shape[1]
            adjusted = presence_scores(scores,self.seen_mask(history,scores))
            probs = top_p(ids,top_k(ids,adjusted)).softmax(dim=-1)
            targets = [motif[step%len(motif)] for motif in motifs]
            row = torch.arange(len(targets),device=scores.device)
            weights = probs[row,targets].tolist()
            for i,(weight,motif) in enumerate(zip(weights,motifs,strict=True)):
                if step<len(motif):
                    sampled[i].append(weight)
            if self.first_step is None:
                raw = adjusted.softmax(dim=-1)[:,THINK_END]
                self.first_step = {"think":torch.stack((raw,raw,probs[:,THINK_END]),dim=-1).tolist()}
            selected_tokens = scores.new_tensor(targets,dtype=torch.long).unsqueeze(-1)
            return torch.full_like(scores,-torch.inf).scatter_(1,selected_tokens,0.)

    records = runtime.generate(selected,"native",0,max(map(len,motifs)),policy=ForcedMotif)
    results = []
    for item,motif,record,probabilities in zip(selected,motifs,records,sampled,strict=True):
        if record["tokens"][:len(motif)]!=motif or len(probabilities)!=len(motif) or record["eos_token"] is not None:
            raise ValueError("motif probe did not execute the fixed tokens through the model")
        results.append({"kind":"probe","id":item["row"]["id"],"state":item["state"],"history":"keep",
                        "motif":motif,"probabilities":probabilities,**probability_product(probabilities)})
    return results


def run(a):
    cohort = a.root/"cohort.json"
    plan = load(cohort)
    states,arms,caps = tuple(plan.get("states",STATES)),tuple(plan.get("arms",ARMS)),tuple(plan["caps"])
    smoke = a.phase=="smoke"
    cases = plan["cases"]
    if smoke:
        cases = cases[:2] if len(states)==1 else [next(x for x in cases if x["state"]==s) for s in states]
    else:
        cases = sorted(cases,key=lambda x:(x["cut"],x["row"]["id"]))[a.part::2]
    r = Runtime(a)
    items = inputs_for(r.tokenizer,cases)
    cap = SMOKE if smoke else caps[-1]
    partial = a.output.with_suffix(".partial.jsonl")
    if a.output.exists():
        raise FileExistsError(a.output)
    with partial.open("x") as stream,r.torch.inference_mode():
        def emit(value):
            print(json.dumps(value,allow_nan=False),file=stream,flush=True)
        emit({"kind":"manifest","phase":a.phase,"part":a.part,"cap":cap,"arms":arms,
              "cohort_sha256":sha(cohort),"late_budget_worker_sha256":sha(Path(__file__)),**r.manifest})
        started = time.monotonic()
        for draw in range(1 if smoke else DRAWS):
            for i in range(0,len(items),BLOCK):
                if time.time() >= a.deadline-100:
                    raise TimeoutError("late-budget worker reached deadline")
                group = items[i:i+BLOCK]
                if draw==0 and plan.get("probe_motif",True):
                    for record in motif_probe(r,group):
                        emit(record)
                traces,audit = sample(r,group,draw,cap-1,smoke,swap_arms=tuple(a for a in (CONV,BOTH) if a in arms),
                                       question_only=DROP in arms)
                emit({"kind":"audit","ids":[x["row"]["id"] for x in group],"draw":draw,"cache":audit})
                question = [x | {"no_think_ids":r.tokenizer.encode(render(r.tokenizer,x["row"],"no-think"),
                                                                  add_special_tokens=False),
                                 "presence_seed_ids":[]} for x in group]
                width = max(max(len(x["native_ids"]),len(x["no_think_ids"])) for x in question)
                if any(THINK_END not in x["no_think_ids"] for x in question):
                    raise ValueError("direct baseline did not prefill the closing marker")
                pack = pack_at(r,[x["no_think_ids"] for x in question],width)
                traces[DIRECT] = r.generate(question,"no_think",draw,cap,packed=pack,start_after_marker=True)
                for arm in arms:
                    for x,record in zip(group,traces[arm],strict=True):
                        emit(record | {"arm":arm,"state":x["state"],"landmark":x["cut"],
                                       "source_prefix_sha256":packed_hash(x["completion_prefix"]),
                                       "render_sha256":x["render_sha256"]})
        emit({"kind":"complete","seconds":time.monotonic()-started})
    partial.replace(a.output)


def read_shard(path,fingerprint,cohort,phase,cases):
    if sha(path)!=fingerprint:
        raise ValueError("late-budget shard checksum changed")
    with path.open() as stream:
        head,*rows,end = map(json.loads,stream)
    plan = load(cohort)
    arms = tuple(plan.get("arms",ARMS))
    cap = SMOKE if phase=="smoke" else plan["caps"][-1]
    if (head["phase"]!=phase or head["cohort_sha256"]!=sha(cohort) or tuple(head["arms"])!=arms
            or head["cap"]!=cap or head["late_budget_worker_sha256"]!=sha(Path(__file__)) or end["kind"]!="complete"):
        raise ValueError("unexpected late-budget worker receipt")
    raw,probes = [],[]
    for r in rows:
        if r["kind"]=="audit":
            continue
        case = cases[r["id"],r["state"]]
        if r["kind"]=="probe":
            p = r["probabilities"]
            if (case["cycle"]["motif"]!=r["motif"] or len(p)!=case["cycle"]["period"] or
                    any(not 0<=q<=1 for q in p) or r["history"]!="keep"):
                raise ValueError("invalid late-budget motif probe")
            probes.append(r)
            continue
        arm,d,p = r["arm"],r["draw"],r["id"]
        if (r["kind"]!="rollout" or arm not in arms or len(tokens(r))>cap
                or r["post_seed"]!=post_seed(p,d) or r["seed"]!=(post_seed(p,d) if arm==DIRECT else seed_for(p,d))
                or r["landmark"]!=case["cut"] or r["render_sha256"]!=case["render_sha256"]
                or packed_hash(case["completion_prefix"])!=r["source_prefix_sha256"]
                or (r["eos_token"] is None and len(tokens(r))!=cap)
                or (r["eos_token"] is not None and r["eos_token"] not in head["eos_ids"])
                or (arm!=DIRECT and (r["think_step"]!=0 or r["tokens"][0]!=THINK_END))):
            raise ValueError("invalid late-budget trajectory")
        raw.append(r)
    return raw,probes,end["seconds"]


def summarize(root,tokenizer):
    cohort = root/"cohort.json"
    plan = load(cohort)
    states,arms,caps = tuple(plan.get("states",STATES)),tuple(plan.get("arms",ARMS)),tuple(plan["caps"])
    cases = {(x["row"]["id"],x["state"]):x for x in plan["cases"]}
    receipts = load(root/"queue/eval-done.json")["outputs"]
    [turn] = tokenizer.encode("<|im_end|>",add_special_tokens=False)
    raw,probes,times = {},{},[]
    for filename,fingerprint in receipts:
        rows,payload,seconds = read_shard(Path(filename),fingerprint,cohort,"eval",cases)
        times.append(seconds)
        for r in rows:
            key = tuple(r[x] for x in ("id","state","draw","arm"))
            if key in raw:
                raise ValueError("duplicate late-budget outcome")
            raw[key] = r
        for p in payload:
            if p["id"] in probes:
                raise ValueError("duplicate late-budget motif probe")
            probes[p["id"]] = p
    if (raw.keys()!={(p,s,d,a) for p,s in cases for d in range(DRAWS) for a in arms} or
            set(probes)!={p for p,s in cases if s==CONFIRM and plan.get("probe_motif",True)}):
        raise ValueError("incomplete late-budget outcomes")

    windows = {}
    for cap in caps:
        rows = {}
        for (p,s,d,a),r in raw.items():
            case = cases[p,s]
            ids,eos = cut(r,"total",cap)
            view = ids if a==DIRECT else ids[1:]
            if turn in view:
                view = view[:view.index(turn)]
            y = score_policy(case["row"],r | {"tokens":ids,"eos_token":r["eos_token"] if eos else None},tokenizer,turn)
            first = first_correct_box(view,tokenizer,Decimal(case["row"]["gold"]))
            cycle = case["cycle"]
            rows[p,s,d,a] = y | {"first_correct_box_token":first+(a!=DIRECT) if first is not None else None,
                                 "terminal_cycle":terminal_cycle(view) is not None,
                                 "original_motif":ends_in_motif(view,cycle["motif"]) if cycle else None}
        table = {}
        for state in states:
            pids = [p for p,s in cases if s==state]
            metrics = ("correct_eos","eos","wrong_eos","missing_eos","cap","numeric_correct","strict_boxed",
                       "terminal_cycle","first_correct_box_token")
            cells,delta = {},{}
            for arm in arms:
                vals = [rows[p,state,d,arm] for p in pids for d in range(DRAWS)]
                cells[arm] = {m:sum(bool(r[m]) for r in vals) for m in metrics} | {
                    "original_motif":sum(r["original_motif"] for r in vals) if state==CONFIRM else None}
            for a,b in ((CONV,NATIVE),(BOTH,NATIVE),(CONV,DIRECT),(NATIVE,DIRECT),(DROP,DIRECT)):
                if a not in arms or b not in arms:
                    continue
                delta[a+"-"+b] = paired_interval({p:[int(rows[p,state,d,a]["correct_eos"])-
                                                     int(rows[p,state,d,b]["correct_eos"]) for d in range(DRAWS)] for p in pids})
            table[state] = {"prompts":len(pids),"rows_per_arm":len(pids)*DRAWS,"arms":cells,"paired_correct_eos":delta}
        windows[str(cap)] = {"summary":table,"cells":[{"id":p,"state":s,"draw":d,"arm":a,**r}
                                                     for (p,s,d,a),r in rows.items()]}

    mapping = {NATIVE:"keep",CONV:"swap_conv",BOTH:"swap_both",DROP:"question_only"}
    parity = {a:{"same":0,"compared":0} for a in mapping if a in arms}
    parity_sources = [] if plan.get("replication") else plan["source_shards"]
    for filename,expected in parity_sources:
        path = Path(filename)
        if sha(path)!=expected:
            raise ValueError("prior 512-token comparison shard changed")
        with path.open() as stream:
            for r in map(json.loads,stream):
                if r["kind"]!="rollout" or (r["id"],r["state"]) not in cases:
                    continue
                for current,old in mapping.items():
                    if current in parity and r["arm"]==old:
                        ids,eos = cut(raw[r["id"],r["state"],r["draw"],current],"total",caps[0])
                        parity[current]["compared"] += 1
                        parity[current]["same"] += (ids==r["tokens"] and eos==(r["eos_token"] is not None))
    result = {"verified_shards":receipts,"cohort_sha256":sha(cohort),"worker_seconds":times,"windows":windows,
              "motif_probes":list(probes.values()),"source_summary_sha256":plan["source_summary_sha256"],
              "scope":plan["scope"]}
    if parity_sources:
        result.update(archived_cut512_parity=parity,parity_note="batch rows changed after removing the other landmarks")
    if plan.get("replication"):
        result["primary_nested_cap"] = 1024
        result["conv_then_direct"] = fallback_policy(raw,cases,caps[-1],tokenizer,turn)
    return result


def fallback_policy(raw,cases,cap,tokenizer,turn):
    """Use the remaining budget on a fresh no-thinking trajectory if conv has not stopped by 1024."""
    switch = 1024
    data = []
    full_cost = {CONV:0,DIRECT:0}
    for (p,s),case in cases.items():
        for draw in range(DRAWS):
            conv,direct = raw[p,s,draw,CONV],raw[p,s,draw,DIRECT]
            full_cost[CONV] += len(tokens(conv))
            full_cost[DIRECT] += len(tokens(direct))
            _,conv_stopped = cut(conv,"total",switch)
            trajectory,budget,spent = (conv,switch,0) if conv_stopped else (direct,cap-switch,switch)
            ids,eos = cut(trajectory,"total",budget)
            row = score_policy(case["row"],trajectory | {"tokens":ids,
                "eos_token":trajectory["eos_token"] if eos else None},tokenizer,turn)
            data.append({"id":p,"state":s,"draw":draw,**row,"used":spent+len(ids)+int(eos),
                         "restarted":not conv_stopped})
    metrics = ("correct_eos","wrong_eos","missing_eos","cap","eos","used","restarted")
    return {"charges_before_restart":switch,"budget":cap,"rows":len(data),
            **{m:sum(r[m] for r in data) for m in metrics},"full_arm_total_tokens":full_cost,"cells":data}


class BudgetQueue(Supervisor):
    def __init__(self,args):
        args.role,args.output = "queue",args.root
        super().__init__(args)
        self.worker = Path(__file__).resolve()
        self.env["PYTHONPATH"] = args.runtime_path

    def plan(self,i,phase):
        a = self.args
        out = self.lane/f"{phase}-{i}.jsonl"
        return i,out,[self.worker,"run","--phase",phase,"--root",self.root,"--model",a.model,
                      "--data",a.data,"--output",out,"--part",i,"--deadline",a.deadline]

    def supervise(self):
        try:
            self.status(cohort_sha256=self.args.cohort_sha256)
            ((path,fingerprint),) = self.run_batch("smoke",[self.plan(0,"smoke")])
            plan = load(self.root/"cohort.json")
            cases = {(x["row"]["id"],x["state"]):x for x in plan["cases"]}
            data,probes,seconds = read_shard(path,fingerprint,self.root/"cohort.json","smoke",cases)
            if len(data)!=len(plan.get("arms",ARMS))*2 or len(probes)!=int(plan.get("probe_motif",True)):
                raise ValueError("late-budget canary matrix is incomplete")
            self.status("canary_passed",canary_sha256=fingerprint,canary_seconds=seconds)
            if not self.bounded(max(1200,seconds*plan["caps"][-1]/SMOKE*4)):
                self.status("budget_complete")
                return
            for attempt in range(10):
                try:
                    check_devices(self.devices)
                    break
                except RuntimeError:
                    busy = {device for device,_ in smi("--query-compute-apps=gpu_uuid,pid")}
                    if busy.intersection(self.devices) or attempt==9:
                        raise
                    time.sleep(2)
            results = self.run_batch("eval",[self.plan(i,"eval") for i in range(2)])
            save(self.lane/"eval-done.json",{"outputs":[[str(p),h] for p,h in results]})
            out = self.root/"summary.json"
            cmd = [sys.executable,"-B",self.worker,"score","--root",self.root,"--model",self.args.model,"--output",out]
            subprocess.run(cmd,env={**self.env,"CUDA_VISIBLE_DEVICES":""},check=True,timeout=180)
            self.status("completed",summary_sha256=sha(out))
        except Exception as exc:
            self.status("failed",error=str(exc))
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("freeze","run","score","launch","supervise"))
    p.add_argument("--phase",choices=("smoke","eval"))
    for name in ("root","model","data","output","overlay","stable","source"):
        p.add_argument("--"+name,type=Path)
    for name in ("runtime-path","devices","host","cohort-sha256"):
        p.add_argument("--"+name)
    p.add_argument("--part",type=int,default=0)
    p.add_argument("--deadline",type=int)
    a = p.parse_args()
    if a.action=="freeze":
        freeze(a.root,a.output)
        return
    if a.action=="run":
        run(a)
        return
    if a.action=="score":
        if a.output.exists():
            raise FileExistsError(a.output)
        from transformers import AutoTokenizer
        save(a.output,summarize(a.root,AutoTokenizer.from_pretrained(a.model,local_files_only=True)))
        return
    q = BudgetQueue(a)
    if socket.gethostname()!=a.host or len(q.devices)!=2 or len(set(q.devices))!=2 or not q.bounded(1200):
        raise ValueError("unexpected late-budget allocation")
    if (sha(a.root/"cohort.json")!=a.cohort_sha256 or
            any(sha(q.worker.parent/n)!=h for n,h in load(a.source).items())):
        raise ValueError("late-budget source or cohort changed")
    if a.action=="supervise":
        q.supervise()
        return
    check_devices(q.devices)
    q.lane.mkdir()
    with (q.lane/"supervisor.log").open("x") as log:
        child = subprocess.Popen([sys.executable,"-B",__file__,"supervise",*sys.argv[2:]],env=q.env,
                                 start_new_session=True,stdout=log,stderr=subprocess.STDOUT)
    print(json.dumps({"supervisor":child.pid,"root":str(a.root),"deadline":a.deadline,
                      "devices":q.devices,"source_sha256":sha(a.source)}))


if __name__=="__main__":
    main()
