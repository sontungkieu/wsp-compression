"""Measure high-budget thinking in two Qwen3.5 checkpoints on new numeric problems."""
import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time

from repetition_baseline import DECIMAL, first_recurrent, sha
from repetition_budget_run import packed_hash, tokens
from repetition_budget_score import cut
from repetition_canonical import paired_interval, seed_for
from repetition_mediation_queue import Supervisor, check_devices, load, save, smi
from repetition_native_horizon import first_correct_box
from repetition_phase_trace import first_long_cycle, repeated_ngram_fraction
from repetition_replay import terminal_cycle
from repetition_restart import BATCH, prepare_items, score_policy
from repetition_restart_cohort import canonical_problem, read_source
from repetition_restart_transfer import Qwen3Runtime
from repetition_sampler_reach import pack_at
from repetition_state_data import keyed
from repetition_state_value import Runtime, post_seed

CAPS = (2048,4096,8192,16384)
TIMING, DIRECT_CAP = 4096,4096
MODELS = ("Qwen3.5-0.8B","Qwen3.5-2B")
ARMS = ("native","no_think")


def freeze(parent,output):
    cohort = load(parent)
    source = Path(cohort["source_path"])
    if output.exists() or cohort["dataset"]!="math500" or sha(source)!=cohort["source_sha256"]:
        raise ValueError("numeric horizon source differs from the verified MATH-500 mirror")
    excluded = {x["row"]["id"] for split in ("pilot","eval") for x in cohort[split]}
    if len(excluded)!=80:
        raise ValueError("expected to exclude the full prior MATH-500 panel")
    pool = []
    for item in read_source(source):
        problem,gold = item["problem"],str(item["answer"]).strip()
        label = "math500:"+hashlib.sha256(problem.encode()).hexdigest()
        if label in excluded or not DECIMAL.fullmatch(gold):
            continue
        row = {"id":label,"task":"math","prompt":canonical_problem(problem),"gold":gold.replace(",",""),
               "boxed_answer_required":True,"astra_prompt_arm":"canonical"}
        pool.append({"row":row})
    pids = {x["row"]["id"] for x in pool}
    if len(pool)!=len(pids) or len(pool)+len(excluded)!=317:
        raise ValueError("numeric mirror or prior cohort count changed")
    picked = sorted(pool,key=lambda x:keyed(x["row"]["id"],"qwen35-horizon-size-v1"))[:20]
    save(output,{"pilot":picked[:4],"eval":picked[4:],"source_sha256":sha(source),
                 "excluded_cohort_sha256":sha(parent),"excluded_ids":sorted(excluded),
                 "native_budgets":CAPS,"direct_cap":DIRECT_CAP,"draw":0,
                 "scope":"hash-selected numeric MATH-500 subset outside the entire earlier question-level study"})


def run(a):
    config = load(a.root/"models.json")[a.part]
    a.model = Path(config["path"])
    a.checkpoint = Path(config["provenance"]) if config.get("provenance") else None
    if a.checkpoint and sha(a.checkpoint)!=config["provenance_sha256"]:
        raise ValueError("2B checkpoint provenance changed before starting")
    runtime = Qwen3Runtime(a) if a.checkpoint else Runtime(a)
    cohort = a.root/"cohort.json"
    plan = load(cohort)
    pilot = a.phase=="pilot"
    samples = prepare_items(runtime,plan[a.phase])
    arms = ("native",) if pilot else ARMS
    [marker] = runtime.groups["think"]
    [turn] = runtime.tokenizer.encode("<|im_end|>",add_special_tokens=False)
    part = a.output.with_suffix(".partial.jsonl")
    if a.output.exists():
        raise FileExistsError(a.output)
    with part.open("x") as stream,runtime.torch.inference_mode():
        def emit(x):
            print(json.dumps(x,allow_nan=False),file=stream,flush=True)
        emit({"kind":"manifest","part":a.part,"phase":a.phase,"model":config["label"],"native_cap":a.cap,
              "direct_cap":None if pilot else DIRECT_CAP,"arms":arms,"think_end_id":marker,"chat_end_id":turn,
              "cohort_sha256":sha(cohort),"horizon_worker_sha256":sha(Path(__file__)),**runtime.manifest})
        started = time.monotonic()
        total_native_max = 0
        for i in range(0,len(samples),BATCH):
            if time.time() >= a.deadline-120:
                raise TimeoutError("Qwen3.5 horizon reached its compute deadline")
            group = samples[i:i+BATCH]
            width = max(max(len(x["native_ids"]),len(x["no_think_ids"])) for x in group)
            if any(marker in x["native_ids"] or marker not in x["no_think_ids"] for x in group):
                raise ValueError("model found the closing marker in the wrong template")
            for arm in arms:
                ids = [x["native_ids" if arm=="native" else "no_think_ids"] for x in group]
                packed = pack_at(runtime,ids,width)
                cap = a.cap if arm=="native" else DIRECT_CAP
                generated = runtime.generate(group,arm,0,cap,packed=packed)
                if arm=="native":
                    total_native_max += max(len(tokens(r)) for r in generated)
                for r,x,input_ids in zip(generated,group,ids,strict=True):
                    emit(r | {"render_sha256":x["render_sha256"],"prefix_sha256":packed_hash(input_ids),
                              "total_cap":cap})
        emit({"kind":"complete","seconds":time.monotonic()-started,"native_max_batch_tokens":total_native_max})
    part.replace(a.output)


def read_shard(path,fingerprint,cohort,phase,cap,selected):
    if sha(path)!=fingerprint:
        raise ValueError("horizon shard checksum changed")
    with path.open() as stream:
        head,*body,end = map(json.loads,stream)
    arms = ("native",) if phase=="pilot" else ARMS
    if (head["phase"]!=phase or head["native_cap"]!=cap or head["cohort_sha256"]!=sha(cohort)
            or head["horizon_worker_sha256"]!=sha(Path(__file__)) or tuple(head["arms"])!=arms
            or end["kind"]!="complete"):
        raise ValueError("horizon worker protocol changed")
    records = {}
    for r in body:
        key = (r["id"],r["arm"])
        limit = cap if r["arm"]=="native" else DIRECT_CAP
        if (r["kind"]!="rollout" or key in records or r["draw"]!=0 or r["seed"]!=seed_for(r["id"],0)
                or r["post_seed"]!=post_seed(r["id"],0) or r["total_cap"]!=limit or len(tokens(r))>limit
                or (r["eos_token"] is None and len(tokens(r))!=limit)
                or (r["eos_token"] is not None and r["eos_token"] not in head["eos_ids"])):
            raise ValueError("invalid horizon trajectory")
        records[key] = r
    if records.keys()!={(p,a) for p in selected for a in arms}:
        raise ValueError("incomplete horizon panel")
    return head,records,end


def measure(row,raw,marker,turn,cap,tokenizer,first_cycle):
    ids,eos = cut(raw,"total",cap)
    t = ids.index(marker) if marker in ids else None
    y = score_policy(row,raw | {"tokens":ids,"eos_token":raw["eos_token"] if eos else None},tokenizer,turn)
    is_native = raw["arm"]=="native"
    thought = ids[:t] if t is not None and is_native else ids
    answer = ids[t+1:] if t is not None and is_native else [] if is_native else ids
    if turn in answer:
        answer = answer[:answer.index(turn)]
    result = y | {"generated_close":t is not None,"first_exact_before_close":first_cycle if first_cycle and first_cycle<=cap else None,
                  "terminal_exact":terminal_cycle(thought) is not None,"recurrent16":first_recurrent(thought) is not None,
                  "correct_box":first_correct_box(answer,tokenizer,Decimal(row["gold"])) is not None}
    if len(thought)>=256:
        for n in (8,16):
            result[f"early_{n}"] = repeated_ngram_fraction(thought[:128],n)
            result[f"late_{n}"] = repeated_ngram_fraction(thought[-128:],n)
    return result


def summarize(root):
    from transformers import AutoTokenizer
    cohort = root/"cohort.json"
    rows = {c["row"]["id"]:c["row"] for c in load(cohort)["eval"]}
    config = load(root/"models.json")
    cap = load(root/"selected-cap.json")["cap"]
    receipts = load(root/"queue/eval-done.json")["outputs"]
    per_model,opening = {},{}
    for filename,fingerprint in receipts:
        part = int(Path(filename).stem.rsplit("-",1)[1])
        spec = config[part]
        tokenizer = AutoTokenizer.from_pretrained(spec["path"],local_files_only=True)
        [marker] = tokenizer.encode("</think>",add_special_tokens=False)
        [turn] = tokenizer.encode("<|im_end|>",add_special_tokens=False)
        head,raw,end = read_shard(Path(filename),fingerprint,cohort,"eval",cap,rows)
        if (head["model"]!=spec["label"] or head["think_end_id"]!=marker or head["chat_end_id"]!=turn
                or head["part"]!=part):
            raise ValueError("horizon model or tokenizer mismatched receipt")
        firsts = {}
        for key,r in raw.items():
            ids = r["tokens"]
            before = ids[:ids.index(marker)] if r["arm"]=="native" and marker in ids else ids
            firsts[key] = first_long_cycle(before)
        windows = {}
        for budget in (b for b in CAPS if b<=cap):
            arms = [a for a in ARMS if a=="native" or budget<=DIRECT_CAP]
            cell = {(p,a):measure(rows[p],raw[p,a],marker,turn,budget,tokenizer,firsts[p,a])
                    for p in rows for a in arms}
            opening[spec["label"],budget] = {p:cell[p,"native"]["correct_eos"] for p in rows}
            metrics = ("correct_eos","wrong_eos","missing_eos","cap","generated_close","terminal_exact",
                       "recurrent16","correct_box","first_exact_before_close")
            table = {}
            for arm in arms:
                outcomes = [cell[p,arm] for p in rows]
                common = [y for y in outcomes if not y["generated_close"] and "late_16" in y]
                table[arm] = {"n":len(outcomes),**{m:sum(bool(y[m]) for y in outcomes) for m in metrics},
                              "open_nonoverlap_n":len(common),
                              "first_exact_tokens":sorted(y["first_exact_before_close"] for y in outcomes
                                                          if y["first_exact_before_close"] is not None),
                              "open_repeat_density":{k:statistics.fmean(y[k] for y in common) if common else None
                                                     for k in ("early_8","late_8","early_16","late_16")}}
            windows[budget] = {"arms":table,"cases":[{"id":p,"arm":a,**y} for (p,a),y in cell.items()]}
        gain = paired_interval({p:[int(opening[spec["label"],cap][p])-int(opening[spec["label"],4096][p])]
                                for p in rows})
        per_model[spec["label"]] = {"manifest":head,"windows":windows,"worker_seconds":end["seconds"],
                                   "native_long_minus4k":gain}
    growth = None
    if set(per_model)==set(MODELS):
        lo,hi = MODELS
        growth = paired_interval({p:[int(opening[hi,cap][p])-int(opening[hi,4096][p])-
                                      int(opening[lo,cap][p])+int(opening[lo,4096][p])] for p in rows})
    return {"selected_cap":cap,"direct_cap":DIRECT_CAP,"cohort_sha256":sha(cohort),"models":per_model,
            "difference_in_native_growth_descriptive":growth,"verified_shards":receipts}


class HorizonQueue(Supervisor):
    def __init__(self,args):
        args.role,args.output = "queue",args.root
        super().__init__(args)
        self.worker = Path(__file__).resolve()
        self.env["PYTHONPATH"] = args.runtime_path

    def plan(self,i,phase,cap):
        a = self.args
        out = self.lane/f"{phase}-{i}.jsonl"
        return i,out,[self.worker,"run","--phase",phase,"--root",self.root,"--data",a.data,
                      "--output",out,"--part",i,"--cap",cap,"--deadline",a.deadline]

    def supervise(self):
        try:
            self.status(cohort_sha256=self.args.cohort_sha256)
            cohort=self.root/"cohort.json"
            pilot={c["row"]["id"] for c in load(cohort)["pilot"]}
            timing=self.run_batch("pilot",[self.plan(i,"pilot",TIMING) for i in range(2)])
            rates,proof=[],[]
            for p,h in timing:
                head,_,end=read_shard(p,h,cohort,"pilot",TIMING,pilot)
                rates.append(max(.02,end["seconds"]/max(1,end["native_max_batch_tokens"])))
                proof.append({"model":head["model"],"seconds":end["seconds"],"sha256":h,
                              "max_batch_tokens":end["native_max_batch_tokens"]})
            estimates={number:2.8*max(rates)*number+480 for number in CAPS[-2:]}
            available=[number for number,eta in estimates.items() if self.bounded(max(1500,eta))]
            if not available:
                self.status("budget_complete",timing=proof)
                return
            cap=max(available)
            save(self.root/"selected-cap.json",{"cap":cap,"direct_cap":DIRECT_CAP,"timing":proof,
                                                 "conservative_wall_seconds":estimates})
            self.status("long_cap_frozen",cap=cap,timing=proof)
            for index in range(10):
                try:
                    check_devices(self.devices)
                    break
                except RuntimeError:
                    busy={u for u,_ in smi("--query-compute-apps=gpu_uuid,pid")}
                    if busy.intersection(self.devices) or index==9:
                        raise
                    time.sleep(2)
            results=self.run_batch("eval",[self.plan(i,"eval",cap) for i in range(2)],tolerate_failure=True)
            if not results:
                raise RuntimeError("no complete Qwen3.5 horizon worker")
            save(self.lane/"eval-done.json",{"outputs":[[str(p),h] for p,h in results]})
            out=self.root/"summary.json"
            cmd=[sys.executable,"-B",self.worker,"score","--root",self.root,"--output",out]
            subprocess.run(cmd,env={**self.env,"CUDA_VISIBLE_DEVICES":""},check=True,timeout=300)
            self.status("completed" if len(results)==2 else "partial",summary_sha256=sha(out),cap=cap)
        except Exception as exc:
            self.status("failed",error=str(exc))
            raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("freeze","run","score","launch","supervise"))
    p.add_argument("--phase",choices=("pilot","eval"))
    for name in ("root","data","output","overlay","stable","source"):
        p.add_argument("--"+name,type=Path)
    for name in ("runtime-path","devices","host","cohort-sha256","models-sha256"):
        p.add_argument("--"+name)
    for name,default in (("part",0),("cap",None),("deadline",None)):
        p.add_argument("--"+name,type=int,default=default)
    a=p.parse_args()
    if a.action=="freeze":
        freeze(a.root,a.output)
        return
    if a.action=="run":
        run(a)
        return
    if a.action=="score":
        if a.output.exists():
            raise FileExistsError(a.output)
        save(a.output,summarize(a.root))
        return
    q=HorizonQueue(a)
    if socket.gethostname()!=a.host or len(q.devices)!=2 or len(set(q.devices))!=2 or not q.bounded(1500):
        raise ValueError("unexpected long horizon allocation")
    if (sha(a.root/"cohort.json")!=a.cohort_sha256 or sha(a.root/"models.json")!=a.models_sha256
            or any(sha(q.worker.parent/n)!=h for n,h in load(a.source).items()) or
            tuple(x["label"] for x in load(a.root/"models.json"))!=MODELS):
        raise ValueError("long horizon source or cohort changed")
    if a.action=="supervise":
        q.supervise()
        return
    check_devices(q.devices)
    q.lane.mkdir()
    with (q.lane/"supervisor.log").open("x") as log:
        child=subprocess.Popen([sys.executable,"-B",__file__,"supervise",*sys.argv[2:]],env=q.env,
                               start_new_session=True,stdout=log,stderr=subprocess.STDOUT)
    print(json.dumps({"supervisor":child.pid,"root":str(a.root),"deadline":a.deadline,
                      "devices":q.devices,"source_sha256":sha(a.source)}))


if __name__=="__main__":
    main()
