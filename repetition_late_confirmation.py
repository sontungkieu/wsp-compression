"""Run a budget-matched whole-panel confirmation with the pinned horizon and replay workers."""
import argparse
from collections import Counter
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import re
import socket
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import repetition_late_budget as replay
import repetition_qwen35_horizon as horizon
from repetition_baseline import DECIMAL, sha
from repetition_budget_run import tokens
from repetition_budget_score import cut
from repetition_canonical import paired_interval
from repetition_late_replay import CONFIRM, cycle_landmarks, ends_in_motif
from repetition_mediation_queue import Supervisor, check_devices, load, save, smi
from repetition_phase_trace import first_long_cycle
from repetition_replay import terminal_cycle
from repetition_restart import score_policy
from repetition_restart_cohort import canonical_problem, read_source
from repetition_state_data import keyed

TOTAL, INCREMENT, PRIMARY = 16384, 2048, 1024
LATEST = TOTAL-INCREMENT
BASE_CUTS, REPLAY_CUTS = (4096,8192,TOTAL), (512,PRIMARY,INCREMENT)
ARMS = (replay.NATIVE,replay.CONV,replay.DIRECT)
FULL_DIRECT = "no_think"
ID_PATTERN = re.compile(rb"math500:[0-9a-f]{64}(?![0-9a-f])")
PILOT_SIZE = 8
METRICS = ("correct_eos","wrong_eos","missing_eos","eos","cap")


def freeze(root,previous,archive,worker_pins):
    previous_plan = load(previous)
    source = Path(previous_plan["source_path"])
    if previous_plan["dataset"]!="math500" or sha(source)!=previous_plan["source_sha256"]:
        raise ValueError("numeric MATH-500 mirror changed")
    excluded,exposure_receipts,scanned = set(),[],0
    for path in sorted(archive.rglob("*")):
        if path.suffix not in {".json",".jsonl"} or root in path.parents or path.is_symlink() or not path.is_file():
            continue
        scanned += 1
        ids = set()
        with path.open("rb") as stream:
            for line in stream:
                ids.update(match.decode("ascii") for match in ID_PATTERN.findall(line))
        if ids:
            excluded.update(ids)
            exposure_receipts.append([str(path.relative_to(archive)),len(ids)])
    prior_ids = {x["row"]["id"] for split in ("pilot","eval") for x in previous_plan[split]}
    if not prior_ids<=excluded:
        raise ValueError("archive did not cover the previous pilot and evaluation cohort")
    numeric = {}
    for item in read_source(source):
        problem,gold = item["problem"],str(item["answer"]).strip()
        if DECIMAL.fullmatch(gold):
            label = "math500:"+hashlib.sha256(problem.encode()).hexdigest()
            if label in numeric:
                raise ValueError("duplicate math problem in source")
            numeric[label] = {"row":{"id":label,"task":"math","prompt":canonical_problem(problem),
                                      "gold":gold.replace(",",""),"boxed_answer_required":True,
                                      "astra_prompt_arm":"canonical"}}
    if len(numeric)!=317:
        raise ValueError("numeric source cardinality changed")
    clean = sorted(numeric.keys()-excluded,key=lambda p:keyed(p,"late-confirmation-panel-v1"))
    if len(clean)<48+PILOT_SIZE:
        raise ValueError(f"only {len(clean)} unexposed numeric IDs in the archived WSP scope")
    save(root/"selection.json",{"source":str(source),"source_sha256":sha(source),
         "prior_cohort_sha256":sha(previous),"archive":str(archive),"archive_files_scanned":scanned,
         "archive_files_with_ids":exposure_receipts,"excluded_ids":sorted(excluded&numeric.keys()),
         "archived_ids_outside_numeric_source":sorted(excluded-numeric.keys()),"archive_clean_count":len(clean),
         "pilot":[numeric[p] for p in clean[:PILOT_SIZE]],"ordered_eval":[numeric[p] for p in clean[PILOT_SIZE:PILOT_SIZE+64]],
         "total":TOTAL,"latest_trigger":LATEST,"increment":INCREMENT,"mechanism_primary":PRIMARY,
         "code":worker_pins,"full_policy_draw":0})


def cache_dtype_audit(runtime,file):
    reports = {}
    layers = [m for m in runtime.model.modules() if "GatedDeltaNet" in type(m).__name__]
    kernels = {}
    if layers:
        for name in ("chunk_gated_delta_rule","recurrent_gated_delta_rule","causal_conv1d_fn","causal_conv1d_update"):
            fn = getattr(layers[0],name,None)
            kernels[name] = [getattr(fn,"__module__",None),getattr(fn,"__name__",None)]
    versions = {}
    for package in ("torch","transformers","flash-linear-attention","fla-core","causal-conv1d"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None

    def hook(_model,_inputs,kwargs,output):
        try:
            inp = kwargs.get("input_ids")
            cache = getattr(output,"past_key_values",None)
            if inp is None or cache is None:
                return
            phase = "prefill" if inp.shape[-1]>1 else "cached_decode"
            if phase in reports:
                return
            fields = {}
            for name in ("recurrent_states","conv_states","key_cache","value_cache"):
                values = getattr(cache,name,())
                fields[name] = dict(Counter(str(x.dtype) for x in values if x is not None and hasattr(x,"dtype")))
            reports[phase] = {"cache_type":type(cache).__name__,"stored_tensor_dtypes":fields}
            if len(reports)==2:
                handle.remove()
                save(file,{"cache":reports,"versions":versions,"kernels":kernels,
                           "attention":getattr(runtime.model.config,"_attn_implementation",None),
                           "checkpoint":runtime.manifest.get("checkpoint")})
        except (AttributeError,RuntimeError,TypeError) as error:
            handle.remove()
            save(file,{"error":str(error),"cache":reports,"versions":versions,"kernels":kernels})

    handle = runtime.model.register_forward_hook(hook,with_kwargs=True)


def baseline_worker(args):
    horizon.DIRECT_CAP = args.direct_cap
    job = SimpleNamespace(root=args.panel,part=0,phase="eval",data=args.data,cap=args.native_cap,
                          deadline=args.deadline,output=args.output)
    if args.output.name.startswith("timing-"):
        original = horizon.Runtime

        def audit(model_args):
            runtime = original(model_args)
            cache_dtype_audit(runtime,args.output.with_suffix(".dtype.json"))
            return runtime

        with patch.object(horizon,"Runtime",audit):
            horizon.run(job)
    else:
        horizon.run(job)


def baseline_read(path,digest,panel,native_cap,direct_cap):
    horizon.DIRECT_CAP = direct_cap
    cohort = panel/"cohort.json"
    rows = {x["row"]["id"] for x in load(cohort)["eval"]}
    head,data,end = horizon.read_shard(path,digest,cohort,"eval",native_cap,rows)
    if head["direct_cap"]!=direct_cap or head["model"]!="Qwen3.5-0.8B":
        raise ValueError("wrong full-panel baseline contract")
    return head,data,end


def first_trigger(raw,marker,limit=LATEST):
    ids = raw["tokens"][:limit]
    if marker in ids:
        ids = ids[:ids.index(marker)]
    return first_long_cycle(ids)


def estimate_seconds(n,direct_cap,native_rate,replay_rate):
    baseline_batches = (n+2*horizon.BATCH-1)//(2*horizon.BATCH)
    replay_batches = (n+2*replay.BLOCK-1)//(2*replay.BLOCK)
    return (baseline_batches*(TOTAL+direct_cap)*native_rate
            + replay_batches*replay.DRAWS*len(ARMS)*INCREMENT*replay_rate+1200)


def summarize_rows(rows):
    return {"n":len(rows),**{m:sum(bool(r[m]) for r in rows) for m in METRICS},
            "generated_tokens_total":sum(r["generated_tokens"] for r in rows),
            "mean_generated_tokens":statistics.fmean(r["generated_tokens"] for r in rows) if rows else None}


def score_cut(row,raw,budget,tokenizer,turn,spent=0):
    ids,eos = cut(raw,"total",budget)
    full = score_policy(row,raw|{"tokens":ids,"eos_token":raw["eos_token"] if eos else None},tokenizer,turn)
    return full|{"generated_tokens":spent+len(ids)+int(eos)}


def paired(rows_a,rows_b):
    return paired_interval({p:[int(a["correct_eos"])-int(b["correct_eos"])
                              for a,b in zip(rows_a[p],rows_b[p],strict=True)] for p in rows_a})


class Confirmation(Supervisor):
    def __init__(self,args):
        args.role,args.output = "queue",args.root
        super().__init__(args)
        self.worker = Path(__file__).resolve()
        self.env["PYTHONPATH"] = args.runtime_path

    def settle(self):
        for attempt in range(10):
            try:
                check_devices(self.devices)
                return
            except RuntimeError:
                others = {u for u,_ in smi("--query-compute-apps=gpu_uuid,pid")}
                if others.intersection(self.devices) or attempt==9:
                    raise
                time.sleep(2)

    def baseline(self,label,items,native_cap,direct_cap):
        plans,locations = [],[]
        for part in range(2):
            path = self.root/f"{label}-{part}"
            path.mkdir()
            save(path/"cohort.json",{"eval":items[part::2],"selection_sha256":self.args.selection_sha256})
            save(path/"models.json",[{"label":"Qwen3.5-0.8B","path":str(self.args.model),"provenance":None}])
            out = self.lane/f"{label}-{part}.jsonl"
            plans.append((part,out,[self.worker,"baseline-worker","--panel",path,"--output",out,
                         "--data",self.args.data,"--deadline",self.args.deadline,"--native-cap",native_cap,
                         "--direct-cap",direct_cap]))
            locations.append(path)
        self.settle()
        receipts = self.run_batch(label,plans)
        return [[str(path),digest,str(panel)] for (path,digest),panel in zip(receipts,locations,strict=True)]

    def read_baseline(self,receipts,native_cap,direct_cap):
        data,meta,times,sources = {},[],[],{}
        for filename,digest,panel in receipts:
            head,rows,end = baseline_read(Path(filename),digest,Path(panel),native_cap,direct_cap)
            if data.keys()&rows.keys():
                raise ValueError("duplicate baseline record across GPUs")
            data.update(rows)
            sources.update({key:digest for key in rows})
            meta.append(head)
            instances = load(Path(panel)/"cohort.json")["eval"]
            counts = [sum(max(len(tokens(rows[x["row"]["id"],arm])) for x in instances[i:i+horizon.BATCH])
                          for arm in horizon.ARMS) for i in range(0,len(instances),horizon.BATCH)]
            times.append(end|{"sampled_batch_steps":sum(counts)})
        if len({(m["tokenizer_sha256"],m["think_end_id"],m["chat_end_id"],tuple(m["eos_ids"])) for m in meta})!=1:
            raise ValueError("baseline models disagree across devices")
        return data,meta[0],times,sources

    def replay_stage(self,label,cases,smoke=False):
        phase = "smoke" if smoke else "eval"
        plans = []
        indices = range(1) if smoke else range(2)
        for part in indices:
            out = self.lane/f"{label}-{part}.jsonl"
            plans.append((part,out,[Path(replay.__file__),"run","--phase",phase,"--root",self.root,
                      "--model",self.args.model,"--data",self.args.data,"--output",out,"--part",part,
                      "--deadline",self.args.deadline]))
        self.settle()
        receipts = self.run_batch(label,plans)
        raw,times = {},[]
        for path,digest in receipts:
            rows,probes,seconds = replay.read_shard(path,digest,self.root/"cohort.json",phase,cases)
            if probes:
                raise ValueError("unexpected motif probe in confirmation")
            for r in rows:
                key = (r["id"],r["draw"],r["arm"])
                if key in raw:
                    raise ValueError("duplicate replay trajectory")
                raw[key] = r
            times.append(seconds)
        pids = [p for p,_ in cases]
        expected = pids[:2] if smoke else pids
        if raw.keys()!={(p,d,arm) for p in expected for d in range(1 if smoke else replay.DRAWS) for arm in ARMS}:
            raise ValueError("incomplete replay arm or prompt")
        return raw,times,[[str(p),d] for p,d in receipts]

    def supervise(self):
        try:
            self.campaign()
        except Exception as error:
            self.status("failed",error=str(error))
            raise

    def campaign(self):
        plan = load(self.root/"selection.json")
        self.status(selection_sha256=self.args.selection_sha256,archive_clean=plan["archive_clean_count"])
        timing = self.baseline("timing",plan["pilot"],4096,4096)
        trows,_,timings,_ = self.read_baseline(timing,4096,4096)
        rate = max(.025,1.55*max(x["seconds"]/x["sampled_batch_steps"] for x in timings))
        replay_rate = max(.032,rate)
        candidates = [(n,c) for n in (64,48) for c in (TOTAL,4096) if n<=len(plan["ordered_eval"])]
        selected = next(((n,c) for n,c in candidates if self.bounded(estimate_seconds(n,c,rate,replay_rate))),None)
        dtype_files = [Path(p).with_suffix(".dtype.json") for p,_,_ in timing]
        timing_proof = {"receipts":timing,"workers":timings,"verified_trajectories":len(trows),
                        "dtype_audit":[[str(p),sha(p)] for p in dtype_files if p.is_file()],
                        "conservative_seconds_per_batch_step":rate,"replay_seconds_per_batch_step":replay_rate}
        if selected is None:
            self.status("skipped_budget",timing=timing_proof)
            return
        n,direct_cap = selected
        window = estimate_seconds(n,direct_cap,rate,replay_rate)
        locked = {"eval":plan["ordered_eval"][:n],"total_cap":TOTAL,"direct_cap":direct_cap,"timing":timing_proof,
                  "selection_sha256":self.args.selection_sha256,"baseline_and_replay_upper_estimate_seconds":window}
        save(self.root/"panel.json",locked)
        self.status("panel_committed",panel_sha256=sha(self.root/"panel.json"),prompts=n,
                    native_cap=TOTAL,direct_cap=direct_cap,eta_seconds=window,timing=timing_proof)
        receipts = self.baseline("baseline",locked["eval"],TOTAL,direct_cap)
        raw,header,elapsed,sources = self.read_baseline(receipts,TOTAL,direct_cap)
        ids = {x["row"]["id"]:x["row"] for x in locked["eval"]}
        if raw.keys()!={(p,arm) for p in ids for arm in horizon.ARMS}:
            raise ValueError("full baseline prompt coverage changed")
        [marker,turn] = [header[k] for k in ("think_end_id","chat_end_id")]
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(self.args.model,local_files_only=True)
        if tok.encode("</think>",add_special_tokens=False)!=[marker] or tok.encode("<|im_end|>",add_special_tokens=False)!=[turn]:
            raise ValueError("baseline tokenizer marker differs from scorer")
        known,eligible,baseline_scores = {},[],{}
        for p,row in ids.items():
            r = raw[p,"native"]
            confirm = first_trigger(r,marker)
            if confirm is not None:
                think = r["tokens"][:confirm]
                eligible.append({"row":row,"state":CONFIRM,"cycle":cycle_landmarks(think,confirm),
                                 "cut":confirm,"completion_prefix":think,"render_sha256":r["render_sha256"],
                                 "source_draw":0,"source_shard_sha256":sources[p,"native"]})
            for arm in horizon.ARMS:
                seq = raw[p,arm]["tokens"]
                thought = seq[:seq.index(marker)] if arm=="native" and marker in seq else seq
                known[p,arm] = first_long_cycle(thought)
        for budget in BASE_CUTS:
            reports = {}
            for arm in horizon.ARMS:
                if arm==FULL_DIRECT and budget>direct_cap:
                    continue
                table = []
                for p,row in ids.items():
                    r = raw[p,arm]
                    sc = horizon.measure(row,r,marker,turn,budget,tok,known[p,arm])
                    cost = len(tokens(r))
                    table.append(sc|{"generated_tokens":min(cost,budget)})
                reports[arm] = summarize_rows(table)|{key:sum(bool(r[key]) for r in table) for key in
                               ("generated_close","first_exact_before_close","terminal_exact","recurrent16")}
                if arm=="native":
                    reports[arm]["not_closed_by_cut"] = n-reports[arm]["generated_close"]
                    reports[arm]["still_thinking_at_cut"] = sum(not r["generated_close"] and not r["eos"] for r in table)
            baseline_scores[str(budget)] = reports
        baseline_summary = self.root/"baseline-summary.json"
        save(baseline_summary,{"panel_sha256":sha(self.root/"panel.json"),"budgets":baseline_scores,
                               "verified_shards":receipts,"workers":elapsed,"eligible":len(eligible),
                               "confirm_tokens":sorted(x["cut"] for x in eligible)})
        self.status("baseline_complete",eligible=len(eligible),baseline_sha256=sha(baseline_summary))
        remaining = max(0,self.args.deadline-time.time()-720)
        batch_time = replay.DRAWS*len(ARMS)*INCREMENT*replay_rate
        max_cases = 2*replay.BLOCK*int(remaining//batch_time)
        cohort = sorted(eligible,key=lambda x:keyed(x["row"]["id"],"late-confirmation-replay-v1"))[:max_cases]
        if cohort and len(cohort)<2:
            self.status("baseline_only",reason="fewer_than_two_replay_cases",eligible=len(eligible))
            return
        replay_rows,wall,smoke,replay_receipts = {},[],[],[]
        if cohort:
            save(self.root/"cohort.json",{"cases":cohort,"caps":REPLAY_CUTS,"states":[CONFIRM],"arms":ARMS,
                 "probe_motif":False,"draws":replay.DRAWS,"source_summary_sha256":sha(baseline_summary),
                 "eligible_pool":len(eligible),"scope":"first-confirmation states chosen without inspecting replay"})
            case_index = {(x["row"]["id"],x["state"]):x for x in cohort}
            _,_,smoke = self.replay_stage("replay_canary",case_index,smoke=True)
            self.status("canary_passed",replay_canary_receipts=smoke,replay_prompts=len(cohort),
                        replay_cohort_sha256=sha(self.root/"cohort.json"))
            replay_rows,wall,replay_receipts = self.replay_stage("replay",case_index)
        needed = {x["row"]["id"]:x for x in eligible}
        self.finish(ids,needed,cohort,raw,replay_rows,tok,turn,direct_cap,baseline_scores,
                    {"baseline":receipts,"pilot":timing,"replay_canary":smoke,"replay":replay_receipts,
                    "baseline_seconds":[x["seconds"] for x in elapsed],"replay_seconds":wall})

    def finish(self,ids,needed,cohort,baseline,raw,tok,turn,direct_cap,base_tables,evidence):
        if sha(self.root/"baseline-summary.json")!=self.state["baseline_sha256"]:
            raise ValueError("baseline changed before final score")
        mechanism = {}
        if cohort:
            pids = [x["row"]["id"] for x in cohort]
            for cap in REPLAY_CUTS:
                arms = {}
                for arm in ARMS:
                    cells = {}
                    for case in cohort:
                        p = case["row"]["id"]
                        cells[p] = []
                        for draw in range(replay.DRAWS):
                            r = raw[p,draw,arm]
                            view,_ = cut(r,"total",cap)
                            view = view if arm==FULL_DIRECT else view[1:]
                            if turn in view:
                                view = view[:view.index(turn)]
                            cells[p].append(score_cut(ids[p],r,cap,tok,turn)|
                                            {"exact_tail":terminal_cycle(view) is not None,
                                             "original_motif":ends_in_motif(view,case["cycle"]["motif"])})
                    arms[arm] = cells
                tables = {}
                for arm,cells in arms.items():
                    flat = [r for outcomes in cells.values() for r in outcomes]
                    tables[arm] = summarize_rows(flat)|{k:sum(r[k] for r in flat) for k in ("exact_tail","original_motif")}
                mechanism[str(cap)] = {"arms":tables,"conv_minus_keep":paired(arms[replay.CONV],arms[replay.NATIVE]),
                                       "conv_minus_residual_direct":paired(arms[replay.CONV],arms[replay.DIRECT]),
                                       "affected_prompts_conv_keep":sum(any(a["correct_eos"]!=b["correct_eos"] for a,b in
                                           zip(arms[replay.CONV][p],arms[replay.NATIVE][p],strict=True)) for p in pids)}
        itt,status = None,"not_estimated_for_missing_triggers"
        if len(needed)==len(cohort):
            base = {p:[score_cut(row,baseline[p,"native"],TOTAL,tok,turn)] for p,row in ids.items()}
            comparisons = {"native":base}
            for label,arm in (("trigger_then_keep",replay.NATIVE),("trigger_then_conv",replay.CONV)):
                comparisons[label] = {p:([score_cut(row,raw[p,0,arm],INCREMENT,tok,turn,spent=needed[p]["cut"])]
                                         if p in needed else base[p]) for p,row in ids.items()}
            matched_direct = direct_cap==TOTAL
            if matched_direct:
                comparisons["direct_from_start"] = {p:[score_cut(row,baseline[p,FULL_DIRECT],TOTAL,tok,turn)]
                                                     for p,row in ids.items()}
            itt = {"draw":0,"n":len(ids),"policy":{a:summarize_rows([v[0] for v in cells.values()])
                   for a,cells in comparisons.items()},"conv_minus_native":paired(comparisons["trigger_then_conv"],base),
                   "conv_minus_keep":paired(comparisons["trigger_then_conv"],comparisons["trigger_then_keep"]),
                   "conv_minus_direct":paired(comparisons["trigger_then_conv"],comparisons["direct_from_start"])
                   if matched_direct else None,"fixed_total_cap":TOTAL}
            status = "complete_draw0"
        out = self.root/"summary.json"
        save(out,{"n":len(ids),"eligible":len(needed),"replayed":len(cohort),"trigger_latest":LATEST,
                  "primary":{"contrast":"conv_minus_keep","incremental_budget":PRIMARY},
                  "secondary":{"contrast":"conv_minus_residual_direct","incremental_budget":INCREMENT},
                  "baseline":base_tables,"direct_baseline_cap":direct_cap,"mechanism":mechanism,
                  "itt_status":status,"itt":itt,"evidence":evidence,"deadline":self.args.deadline,
                  "latency_scope":"worker wall seconds by batch and device, without an online deployment measurement"})
        self.status("completed",summary_sha256=sha(out),replay_prompts=len(cohort),itt_status=status)


def verify(args):
    actual = {i:u for i,u in smi("--query-gpu=index,uuid")}
    if socket.gethostname()!=args.host or [actual.get(i) for i in ("0","1")]!=args.devices.split(","):
        raise ValueError("incorrect node or GPU allocation")
    pins = {horizon.__file__:args.horizon_sha256,replay.__file__:args.replay_sha256,
            __file__:args.worker_sha256}
    if any(sha(Path(p))!=h for p,h in pins.items()):
        raise ValueError("pinned horizon, replay or confirmation source changed")
    return pins


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("freeze","launch","supervise","baseline-worker"))
    for name in ("root","panel","model","data","output","overlay","stable","previous","archive"):
        p.add_argument("--"+name,type=Path)
    for name in ("runtime-path","devices","host","selection-sha256","worker-sha256","horizon-sha256","replay-sha256"):
        p.add_argument("--"+name)
    for name in ("deadline","access-cutoff","native-cap","direct-cap"):
        p.add_argument("--"+name,type=int)
    a = p.parse_args()
    if a.action=="baseline-worker":
        baseline_worker(a)
        return
    pins = verify(a)
    if a.action=="freeze":
        if (a.root/"selection.json").exists():
            raise FileExistsError("selection manifest already exists")
        freeze(a.root,a.previous,a.archive,pins)
        print(json.dumps({"selection_sha256":sha(a.root/"selection.json"),
                          "clean":load(a.root/"selection.json")["archive_clean_count"]}))
        return
    q = Confirmation(a)
    if (not q.bounded(1800) or sha(a.root/"selection.json")!=a.selection_sha256
            or load(a.root/"selection.json")["code"]!=pins):
        raise ValueError("confirmation allocation, selection or pins changed")
    if a.action=="supervise":
        q.supervise()
        return
    if time.time()>=a.access_cutoff:
        raise TimeoutError("new submission outside the access window")
    check_devices(q.devices)
    q.lane.mkdir()
    with (q.lane/"supervisor.log").open("x") as log:
        process = subprocess.Popen([sys.executable,"-B",__file__,"supervise",*sys.argv[2:]],env=q.env,
                                   start_new_session=True,stdout=log,stderr=subprocess.STDOUT)
    print(json.dumps({"supervisor":process.pid,"host":a.host,"devices":q.devices,"deadline":a.deadline,
                      "access_cutoff":a.access_cutoff,"selection_sha256":a.selection_sha256,"status":str(q.lane/"status.json")}))


if __name__=="__main__":
    main()
