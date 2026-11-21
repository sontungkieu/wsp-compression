"""Replay paired Qwen3.5 states around the onset of late exact repetition."""
import argparse
from decimal import Decimal
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

from repetition_baseline import BOXED, sha
from repetition_budget_run import packed_hash, tokens
from repetition_cache_swap import FIELDS, marker_hook, prefill_donor, state_digest
from repetition_canonical import paired_interval, seed_for
from repetition_mediation_queue import Supervisor, check_devices, load, save
from repetition_native_horizon import first_correct_box
from repetition_replay import terminal_cycle
from repetition_restart import score_policy
from repetition_sampler_reach import pack_at
from repetition_state_value import Runtime, THINK_END, inputs_for, post_seed

CAP = 512
DRAWS = 2
BEFORE = "before_exact_onset"
CONFIRM = "at_exact_confirmation"
STATES = (BEFORE, CONFIRM, "open_8192", "open_16384")
NATIVE, KEEP, CONV, SWAP, DROP, HISTORY = (
    "native_reset","keep","swap_conv","swap_both","question_only","native_keep_history")
ARMS = (NATIVE, KEEP, CONV, SWAP, DROP, HISTORY)


def cycle_landmarks(ids, confirm):
    """Return 0-based cut lengths, including the first whole period ending in phase at confirm."""
    prefix = ids[:confirm]
    cycle = terminal_cycle(prefix)
    if cycle is None or terminal_cycle(prefix[:-1]) is not None:
        raise ValueError("source is not the first exact confirmation")
    period, _ = cycle
    start = confirm - 1
    while start >= period and ids[start] == ids[start-period]:
        start -= 1
    raw = start + 1 - period
    aligned = raw + (confirm-raw) % period
    motif = ids[aligned:aligned+period]
    if confirm-aligned < 5*period or motif*((confirm-aligned)//period) != ids[aligned:confirm]:
        raise ValueError("cannot reconstruct the confirmed exact motif")
    return {"retrospective_start": raw, "whole_period_start": aligned, "confirmation": confirm,
            "period": period, "motif": motif}


def freeze(source, output):
    summary_path = source/"summary.json"
    status = load(source/"queue/status.json")
    fingerprint = sha(summary_path)
    summary = load(summary_path)
    if (status["stage"] != "completed" or status["summary_sha256"] != fingerprint
            or summary["selected_cap"]["cap"] != 32768 or output.exists()):
        raise ValueError("expected a completed 32K horizon and a fresh cohort")
    source_cohort = source/"cohort.json"
    rows = {r["id"]:r for r in load(source_cohort)["cases"]}
    chosen, seen = [], set()
    for file, expected in summary["verified_shards"]:
        path = Path(file)
        if sha(path) != expected:
            raise ValueError("horizon shard changed")
        with path.open() as stream:
            header, *records, end = map(json.loads, stream)
        if header["cohort_sha256"] != sha(source_cohort) or end["kind"] != "complete":
            raise ValueError("invalid source shard")
        for record in records:
            pid = record["id"]
            if record["kind"] != "rollout" or record["draw"] != 0 or pid in seen or pid not in rows:
                raise ValueError("invalid horizon record")
            seen.add(pid)
            event = summary["events_by_prompt"][pid]
            ids = record["tokens"]
            if event["first_close_token"] is not None:
                continue
            if len(ids) != 32768 or THINK_END in ids or record["eos_token"] is not None:
                raise ValueError("open landmark needs a full never-closed trajectory")
            confirm = event["first_exact_cycle_before_close"]
            evidence = cycle_landmarks(ids, confirm) if confirm is not None else None
            cuts = ((BEFORE, evidence["whole_period_start"]), (CONFIRM, confirm)) if evidence else (
                ("open_8192", 8192), ("open_16384", 16384))
            for state, stop in cuts:
                chosen.append({"row": rows[pid], "state": state, "completion_prefix": ids[:stop],
                               "render_sha256": record["render_sha256"], "cut": stop, "cycle": evidence,
                               "source_draw": record["draw"], "source_shard_sha256": expected})
    counts = {state:sum(x["state"] == state for x in chosen) for state in STATES}
    if seen != rows.keys() or tuple(counts.values()) != (7,7,3,3):
        raise ValueError("late cohort must be the seven exact and three open horizon traces")
    save(output, {"source_summary_sha256":fingerprint,"source_cohort_sha256":sha(source_cohort),
                  "states":counts,"total_cap":CAP,"draws":DRAWS,
                  "cases":sorted(chosen,key=lambda x:(x["row"]["id"],x["state"]))})


def first_probe(runtime, group, notes, arm):
    def probe(probabilities, adjusted):
        for x, p, z in zip(group, probabilities, adjusted, strict=True):
            if x["state"] != CONFIRM:
                continue
            next_cycle = x["cycle"]["motif"][0]
            measures = {}
            for label, tok in (("marker", THINK_END), ("cycle_token", next_cycle)):
                measures[label] = {"token":tok, "adjusted_logit":float(z[tok]),
                                   "rank_strict":int((z>z[tok]).sum())+1,
                                   "probability_after_filter":float(p[tok])}
            notes[x["row"]["id"],x["state"]][arm] = measures
        return probabilities
    return probe


def run_batch(runtime, group, draw, cap, smoke):
    width_inputs = [x["native_ids"] for x in group]
    packed = runtime.pack(width_inputs)
    width = packed["input_ids"].shape[1]
    native = [x | {"presence_seed_ids":[]} for x in group]
    question, donor = prefill_donor(runtime, native)
    donor_sha = state_digest(donor, FIELDS, runtime.torch) if smoke else None
    probes = {(x["row"]["id"],x["state"]):{} for x in group}
    outputs = {}
    for arm, batch in ((NATIVE,native),(HISTORY,group)):
        outputs[arm] = runtime.generate(batch,"native",draw,cap,packed=packed,
                                        first_transform=first_probe(runtime,batch,probes,arm))
    audits = {}
    for arm, mode, fields in ((KEEP,"native",FIELDS),(CONV,"swap",FIELDS[:1]),(SWAP,"swap",FIELDS)):
        with marker_hook(runtime,width,mode,donor,digest_kv=True,fields=fields) as audit:
            outputs[arm] = runtime.generate(native,"e0",draw,cap-1,packed=packed)
        audits[arm] = audit
    original = audits[KEEP]
    deltas = {}
    for arm in (CONV,SWAP):
        changed = audits[arm]
        if any(original[f] != changed[f] for f in ("kv_sha256","mask_sha256","cache_position")):
            raise ValueError("late replay changes receiver KV, mask or positions")
        deltas[arm] = (original["logits"]-changed["logits"]).abs().amax(-1).tolist()
    if smoke:
        with marker_hook(runtime,width,"self",digest_kv=True) as audit:
            self_copy = runtime.generate(native,"e0",draw,cap-1,packed=packed)
        if (audit["kv_sha256"] != original["kv_sha256"] or
                not runtime.torch.equal(audit["logits"],original["logits"]) or
                any(tokens(x)!=tokens(y) for x,y in zip(outputs[KEEP],self_copy,strict=True)) or
                donor_sha != state_digest(donor,FIELDS,runtime.torch) or
                not all(any(d>0 for d in values) for values in deltas.values())):
            raise ValueError("cache swap canary parity or routing failed")
    outputs[DROP] = runtime.generate(question,"e0",draw,cap-1,
                                     packed=pack_at(runtime,[x["native_ids"] for x in question],width))
    for i,x in enumerate(group):
        pid = x["row"]["id"]
        proof = {"kv_sha256": original["kv_sha256"],"mask_sha256":original["mask_sha256"],
                 "cache_position":original["cache_position"],
                 "max_post_marker_logit_delta":{arm:values[i] for arm,values in deltas.items()},
                 "layers_changed":{arm:audits[arm]["layers_changed"] for arm in deltas},
                 "prefix_sha256":packed_hash(x["native_ids"]),
                 "question_only_sha256":packed_hash(question[i]["native_ids"]),
                 "retained_history_size":len(set(x["completion_prefix"]))}
        for arm in ARMS:
            yield outputs[arm][i] | {"arm":arm,"state":x["state"],"cut":x["cut"],"proof":proof,
                                     "render_sha256":x["render_sha256"],
                                     "probe":probes[pid,x["state"]].get(arm)}


def emit_rollouts(args):
    path = args.root/"cohort.json"
    plan = load(path)
    cases = plan["cases"]
    smoke = args.phase == "smoke"
    if smoke:
        confirm = next(x for x in cases if x["state"] == CONFIRM)
        before = next(x for x in cases if x["state"] == BEFORE and x["row"]["id"] == confirm["row"]["id"])
        cases = [before,confirm,next(x for x in cases if x["state"] == "open_8192")]
    else:
        # Keep the two draw landmarks for a question on the same device.
        pids = sorted({x["row"]["id"] for x in cases})[args.part::2]
        cases = [x for x in cases if x["row"]["id"] in pids]
    runtime = Runtime(args)
    items = sorted(inputs_for(runtime.tokenizer,cases),key=lambda x:(x["cut"],x["row"]["id"]))
    partial = args.output.with_suffix(".partial.jsonl")
    if args.output.exists():
        raise FileExistsError(args.output)
    cap = 64 if smoke else CAP
    with partial.open("x") as stream, runtime.torch.inference_mode():
        def write(x):
            print(json.dumps(x,allow_nan=False),file=stream,flush=True)
        write({"kind":"manifest","phase":args.phase,"part":args.part,"cap":cap,"arms":ARMS,
               "cohort_sha256":sha(path),"late_worker_sha256":sha(Path(__file__)),**runtime.manifest})
        start = time.monotonic()
        for draw in range(1 if smoke else DRAWS):
            for i in range(0,len(items),3):
                if time.time() >= args.deadline-60:
                    raise TimeoutError("late replay reached the compute deadline")
                for record in run_batch(runtime,items[i:i+3],draw,cap,smoke):
                    write(record)
        write({"kind":"complete","seconds":time.monotonic()-start})
    partial.replace(args.output)


def ends_in_motif(ids, motif):
    period = len(motif)
    width = max(128,5*period)
    if len(ids) < width:
        return False
    tail = ids[-width:]
    return tail[:-period] == tail[period:] and any(
        tail[-period:] == motif[shift:]+motif[:shift] for shift in range(period))


def outcome(case, record, tokenizer, turn):
    ids = record["tokens"]
    t = record["think_step"]
    answer = ids[t+1:] if t is not None else []
    if turn in answer:
        answer = answer[:answer.index(turn)]
    box = BOXED.search(tokenizer.decode(answer,skip_special_tokens=False))
    cycle_ids = answer if t is not None else ids
    evidence = case["cycle"]
    final_motif = ends_in_motif(cycle_ids,evidence["motif"]) if evidence else None
    if evidence and t is None:
        final_motif = ends_in_motif(case["completion_prefix"][-max(128,5*evidence["period"]):]+ids,evidence["motif"])
    y = score_policy(case["row"],record,tokenizer,turn)
    return y | {"closed":t is not None,"terminal_cycle":terminal_cycle(cycle_ids) is not None,
                "original_period_at_tail":final_motif,"any_box_after_close":box is not None,
                "first_box_value":box.group(1) if box else None,
                "first_correct_box_post":first_correct_box(answer,tokenizer,Decimal(case["row"]["gold"])),
                "close_token":t+1 if t is not None else None}


def read_shard(path, sha256, cohort_sha, phase, cap, cases):
    if sha(path) != sha256:
        raise ValueError("late replay shard checksum changed")
    with path.open() as stream:
        head,*rows,end = map(json.loads,stream)
    if (head["late_worker_sha256"] != sha(Path(__file__)) or head["cohort_sha256"] != cohort_sha
            or head["phase"] != phase or head["cap"] != cap or tuple(head["arms"]) != ARMS
            or end["kind"] != "complete"):
        raise ValueError("late replay receipt does not match configured protocol")
    for r in rows:
        key = (r["id"],r["state"])
        case = cases[key]
        if (r["kind"] != "rollout" or r["arm"] not in ARMS or len(tokens(r)) > cap
                or (r["eos_token"] is not None and r["eos_token"] not in head["eos_ids"])
                or (r["eos_token"] is None and len(tokens(r)) != cap)
                or r["seed"] != seed_for(r["id"],r["draw"]) or r["post_seed"] != post_seed(r["id"],r["draw"])
                or r["cut"] != case["cut"] or r["render_sha256"] != case["render_sha256"]
                or (r["arm"] in (KEEP,CONV,SWAP,DROP) and (r["think_step"] != 0 or r["tokens"][0] != THINK_END))
                or bool(r["probe"]) != (r["state"] == CONFIRM and r["arm"] in (NATIVE,HISTORY))):
            raise ValueError("invalid late replay rollout")
    return rows,end["seconds"]


def summarize(root, tokenizer):
    cohort = root/"cohort.json"
    plan = load(cohort)
    cases = {(x["row"]["id"],x["state"]):x for x in plan["cases"]}
    [turn] = tokenizer.encode("<|im_end|>",add_special_tokens=False)
    receipts = load(root/"queue/eval-done.json")["outputs"]
    scores, bases, seconds = {}, {}, []
    for filename, fingerprint in receipts:
        rows, dur = read_shard(Path(filename),fingerprint,sha(cohort),"eval",CAP,cases)
        seconds.append(dur)
        for record in rows:
            pid, state, draw, arm = (record[x] for x in ("id","state","draw","arm"))
            key = (pid,state,draw,arm)
            if key in scores:
                raise ValueError("duplicate late replay trace")
            scores[key] = outcome(cases[pid,state],record,tokenizer,turn)
            if record["probe"]:
                bases[key] = record["probe"]
    if scores.keys() != {(p,s,d,a) for p,s in cases for d in range(DRAWS) for a in ARMS}:
        raise ValueError("incomplete late replay matrix")
    counts = {}
    metrics = ("correct_eos","wrong_eos","missing_eos","cap","closed","numeric_correct","strict_boxed",
               "terminal_cycle","original_period_at_tail","any_box_after_close","first_correct_box_post")
    for state in STATES:
        pids = [p for p,s in cases if state == s]
        arm_counts = {}
        for arm in ARMS:
            values = [scores[p,state,d,arm] for p in pids for d in range(DRAWS)]
            arm_counts[arm] = {m:(None if all(y[m] is None for y in values) and m == "original_period_at_tail"
                                   else sum(bool(y[m]) for y in values)) for m in metrics}
        contrasts = {}
        for a,b in ((KEEP,NATIVE),(CONV,KEEP),(SWAP,KEEP),(CONV,SWAP),(DROP,KEEP),(SWAP,DROP),(HISTORY,NATIVE)):
            contrasts[a+"-"+b] = {m:paired_interval({p:[int(bool(scores[p,state,d,a][m]))-
                int(bool(scores[p,state,d,b][m])) for d in range(DRAWS)] for p in pids})
                for m in ("correct_eos","cap","terminal_cycle")}
        counts[state] = {"prompts":len(pids),"rows_per_arm":len(pids)*DRAWS,"arms":arm_counts,"contrasts":contrasts}
    return {"cohort_sha256":sha(cohort),"total_cap":CAP,"draws":DRAWS,"results":counts,
            "landmarks":[{"id":p,"state":s,"cut":x["cut"],"cycle":x["cycle"]} for (p,s),x in cases.items()],
            "per_cell":[{"id":p,"state":s,"draw":d,"arm":a,**v,"probe":bases.get((p,s,d,a))}
                         for (p,s,d,a),v in scores.items()],
            "worker_seconds":seconds,"verified_shards":receipts}


class LateQueue(Supervisor):
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
            ((path,digest),) = self.run_batch("smoke",[self.plan(0,"smoke")])
            plan = load(self.root/"cohort.json")
            cases = {(x["row"]["id"],x["state"]):x for x in plan["cases"]}
            smoke,seconds = read_shard(path,digest,self.args.cohort_sha256,"smoke",64,cases)
            if len(smoke) != 3*len(ARMS) or {x["state"] for x in smoke} != {BEFORE,CONFIRM,"open_8192"}:
                raise ValueError("incomplete late replay canary")
            self.status("canary_passed",canary_sha256=digest,canary_seconds=seconds)
            results = self.run_batch("eval",[self.plan(i,"eval") for i in range(2)])
            save(self.lane/"eval-done.json",{"outputs":[[str(p),d] for p,d in results]})
            out = self.root/"summary.json"
            cmd = [sys.executable,"-B",self.worker,"score","--root",self.root,"--model",self.args.model,
                   "--output",out]
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
    if a.action == "freeze":
        freeze(a.root,a.output)
        return
    if a.action == "run":
        emit_rollouts(a)
        return
    if a.action == "score":
        if a.output.exists():
            raise FileExistsError(a.output)
        from transformers import AutoTokenizer
        save(a.output,summarize(a.root,AutoTokenizer.from_pretrained(a.model,local_files_only=True)))
        return
    q = LateQueue(a)
    if socket.gethostname() != a.host or len(set(q.devices)) != 2 or not q.bounded(900):
        raise ValueError("wrong late replay host, device allocation or remaining wall time")
    if (sha(a.root/"cohort.json") != a.cohort_sha256
            or any(sha(q.worker.parent/n) != expected for n,expected in load(a.source).items())):
        raise ValueError("late replay source or cohort changed")
    if a.action == "supervise":
        q.supervise()
        return
    check_devices(q.devices)
    q.lane.mkdir()
    with (q.lane/"supervisor.log").open("x") as log:
        child = subprocess.Popen([sys.executable,"-B",__file__,"supervise",*sys.argv[2:]],env=q.env,
                                 start_new_session=True,stdout=log,stderr=subprocess.STDOUT)
    print(json.dumps({"supervisor":child.pid,"root":str(a.root),"deadline":a.deadline,
                      "devices":q.devices,"source_sha256":sha(a.source)}))


if __name__ == "__main__":
    main()
