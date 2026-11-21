"""Test whether cache refresh preserves a verified, model-written arithmetic checkpoint."""
import argparse
import json
from pathlib import Path
import random
import re
import socket
import subprocess
import sys
import time

from repetition_baseline import BOXED, astra_instruction, render, sha
from repetition_budget_run import packed_hash, tokens
from repetition_budget_score import cut
from repetition_cache_factorial import BOTH, CONV, DROP, NATIVE, sample
from repetition_canonical import paired_interval, seed_for
from repetition_mediation_queue import Supervisor, check_devices, load, save
from repetition_restart import score_policy
from repetition_state_value import Runtime, THINK_END, inputs_for, post_seed

BUDGET, LIMIT, CHECKPOINT = 1024, 512, 3
PILOT, EVAL, DRAWS, BLOCK = 8, 32, 2, 4
NATURAL, DIRECT = "native_thinking", "no_think"
REPLAY = (NATIVE,CONV,BOTH,DROP)
CHECKPOINT_LINE = re.compile(r"^\s*State[ \t]+3[ \t]*=[ \t]*(-?[0-9]+)[ \t]*(?=\n)",re.I|re.M)
LATER_LINE = re.compile(r"^\s*State[ \t]+[456][ \t]*=",re.I|re.M)


def make_panel(count=PILOT+EVAL):
    rng = random.Random(21475)
    rows = []
    for n in range(count):
        initial = rng.randrange(36,178)
        operations = (("add",rng.randrange(13,67)),("multiply",rng.randrange(3,8)),
                      ("subtract",rng.randrange(18,97)),("add",rng.randrange(18,91)),
                      ("multiply",rng.randrange(3,8)),("subtract",rng.randrange(19,135)))
        value, states, directions = initial, [], []
        for step,(op,arg) in enumerate(operations,1):
            value = {"add":value+arg,"multiply":value*arg,"subtract":value-arg}[op]
            states.append(value)
            directions.append(f"{step}. " + {"add":f"Add {arg}.","multiply":f"Multiply by {arg}.",
                                              "subtract":f"Subtract {arg}."}[op])
        prompt = (f"Start with the integer {initial}. Apply these operations in order.\n"
                  +"\n".join(directions)+"\nWhile computing, record each intermediate result on its own line "
                  'as "State 1 = number", "State 2 = number", and so on. '
                  "Replace number with the integer obtained after that step. "
                  "The final answer is the integer after all six operations.\n"+astra_instruction("canonical"))
        row = {"id":f"written-checkpoint:{n:02d}","task":"math","prompt":prompt,"gold":str(value),
               "astra_prompt_arm":"canonical","boxed_answer_required":True}
        rows.append({"row":row,"initial":initial,"operations":operations,"states":states})
    return {"pilot":rows[:PILOT],"eval":rows[PILOT:],"budget":BUDGET,"landmark_limit":LIMIT,
            "checkpoint_step":CHECKPOINT,"draws":DRAWS}


def checkpoint(record, gold, tokenizer):
    """Select earliest completed valid checkpoint using only tokens through that cut."""
    prior = record["tokens"][:LIMIT]
    if THINK_END in prior:
        prior = prior[:prior.index(THINK_END)]
    if CHECKPOINT_LINE.search(tokenizer.decode(prior,skip_special_tokens=False)) is None:
        return None,"no_completed_step3_line"
    wrong = False
    for stop in range(1,len(prior)+1):
        text = tokenizer.decode(prior[:stop],skip_special_tokens=False)
        if BOXED.search(text) or LATER_LINE.search(text):
            return None,"passed_step3_before_verified"
        marks = tuple(CHECKPOINT_LINE.finditer(text))
        if any(int(m.group(1)) == gold for m in marks):
            return stop,"verified"
        wrong |= bool(marks)
    return None,"wrong_step3_value" if wrong else "unverified_step3_line"


def worker(a):
    cohort = a.root/"cohort.json"
    plan = load(cohort)
    smoke = a.phase=="pilot"
    panel = plan[a.phase] if smoke else plan[a.phase][a.part::2]
    draws = range(1 if smoke else plan["draws"])
    runtime = Runtime(a)
    empty = inputs_for(runtime.tokenizer,[x | {"completion_prefix":[]} for x in panel])
    direct = [x | {"no_think_ids":runtime.tokenizer.encode(render(runtime.tokenizer,x["row"],"no-think"),
                                                             add_special_tokens=False)} for x in empty]
    partial = a.output.with_suffix(".partial.jsonl")
    if a.output.exists():
        raise FileExistsError(a.output)
    with partial.open("x") as stream,runtime.torch.inference_mode():
        def emit(value):
            print(json.dumps(value,allow_nan=False),file=stream,flush=True)
        emit({"kind":"manifest","phase":a.phase,"part":a.part,"cohort_sha256":sha(cohort),
              "checkpoint_worker_sha256":sha(Path(__file__)),"budget":BUDGET,"limit":LIMIT,**runtime.manifest})
        start = time.monotonic()
        for draw in draws:
            for i in range(0,len(panel),BLOCK):
                if time.time() >= a.deadline-120:
                    raise TimeoutError("model-written checkpoint worker reached deadline")
                batch = empty[i:i+BLOCK]
                trunk = runtime.generate(batch,"native",draw,BUDGET)
                plain = runtime.generate(direct[i:i+BLOCK],"no_think",draw,BUDGET,start_after_marker=True)
                eligible = []
                for case,n,d in zip(batch,trunk,plain,strict=True):
                    stop,why = checkpoint(n,case["states"][CHECKPOINT-1],runtime.tokenizer)
                    emit({"kind":"landmark","id":n["id"],"draw":draw,"reason":why,"spent":stop,
                          "render_sha256":case["render_sha256"]})
                    emit(n | {"arm":NATURAL,"spent":0})
                    emit(d | {"arm":DIRECT,"spent":0})
                    if stop is not None:
                        eligible.append(case | {"completion_prefix":n["tokens"][:stop],"spent":stop})
                if not eligible:
                    continue
                group = inputs_for(runtime.tokenizer,eligible)
                # Generate the longest needed suffix in the batch, then charge each exact checkpoint cut.
                max_post = BUDGET-min(x["spent"] for x in group)-1
                traces,audit = sample(runtime,group,draw,max_post,smoke,swap_arms=(CONV,BOTH))
                emit({"kind":"cache_audit","ids":[x["row"]["id"] for x in group],"draw":draw,"arms":audit})
                for arm in REPLAY:
                    for case,rec in zip(group,traces[arm],strict=True):
                        ids, eos = cut(rec,"total",BUDGET-case["spent"])
                        emit(rec | {"arm":arm,"spent":case["spent"],"tokens":ids,
                                    "eos_token":rec["eos_token"] if eos else None,
                                    "source_prefix_sha256":packed_hash(case["completion_prefix"])})
        emit({"kind":"complete","seconds":time.monotonic()-start})
    partial.replace(a.output)


def report(root,phase,tokenizer):
    cohort = root/"cohort.json"
    plan = load(cohort)
    rows = {x["row"]["id"]:x for x in plan[phase]}
    files = load(root/"queue"/f"{phase}-done.json")["outputs"]
    draws = range(1 if phase=="pilot" else plan["draws"])
    check,results,seconds = {},{},[]
    [turn] = tokenizer.encode("<|im_end|>",add_special_tokens=False)
    for filename,fingerprint in files:
        path = Path(filename)
        if sha(path) != fingerprint:
            raise ValueError("checkpoint replay shard changed")
        with path.open() as stream:
            head,*records,tail = map(json.loads,stream)
        if (head["cohort_sha256"] != sha(cohort) or head["checkpoint_worker_sha256"] != sha(Path(__file__))
                or head["phase"] != phase or head["budget"] != BUDGET or tail["kind"] != "complete"):
            raise ValueError("unexpected checkpoint replay worker protocol")
        seconds.append(tail["seconds"])
        for record in records:
            if record["kind"]=="cache_audit":
                continue
            pid,d = record["id"],record["draw"]
            if pid not in rows or d not in draws:
                raise ValueError("invalid checkpoint replay row")
            if record["kind"]=="landmark":
                if (pid,d) in check:
                    raise ValueError("duplicate landmark")
                check[pid,d] = record
                continue
            arm = record["arm"]
            if ((pid,d,arm) in results or record["kind"] != "rollout"
                    or record["post_seed"] != post_seed(pid,d)
                    or record["seed"] != (post_seed(pid,d) if arm == DIRECT else seed_for(pid,d))
                    or (record["eos_token"] is not None and record["eos_token"] not in head["eos_ids"])
                    or len(tokens(record))+record["spent"]>BUDGET):
                raise ValueError("invalid checkpoint replay trajectory")
            results[pid,d,arm] = record
    population = {(p,d) for p in rows for d in draws}
    selected = {k for k,c in check.items() if c["spent"] is not None}
    expected = {(p,d,a) for p,d in population for a in (NATURAL,DIRECT)}
    expected |= {(p,d,a) for p,d in selected for a in REPLAY}
    if population != check.keys() or expected != results.keys():
        raise ValueError("incomplete checkpoint replay matrix")
    for pid,d in sorted(population):
        native = results[pid,d,NATURAL]
        stop,why = checkpoint(native,rows[pid]["states"][CHECKPOINT-1],tokenizer)
        if (stop,why) != (check[pid,d]["spent"],check[pid,d]["reason"]):
            raise ValueError("checkpoint selection did not reproduce from the recorded trace")
        if stop is not None and any(
                (rec := results[pid,d,a])["spent"] != stop or rec["tokens"][0] != THINK_END
                or rec["think_step"] != 0 or rec["source_prefix_sha256"] != packed_hash(native["tokens"][:stop])
                for a in REPLAY):
            raise ValueError("replay source or forced marker disagree with the verified checkpoint")

    def eval_at(pid,d,arm,budget):
        # A state that has not arrived within this total budget cannot trigger the policy.
        if arm in REPLAY and ((pid,d) not in selected or check[pid,d]["spent"]+1>budget):
            arm = NATURAL
        source = results[pid,d,arm]
        ids,eos = cut(source,"total",budget-source["spent"])
        return score_policy(rows[pid]["row"],source | {"tokens":ids,"eos_token":source["eos_token"] if eos else None},
                            tokenizer,turn)

    windows = {}
    for budget in (512,BUDGET):
        selection = sorted(k for k in selected if check[k]["spent"]+1 <= budget)
        groups = {"all_with_natural_fallback":sorted(population),"eligible":selection}
        table = {}
        for name,pairs in groups.items():
            y = {(p,d,a):eval_at(p,d,a,budget) for p,d in pairs for a in (NATURAL,DIRECT,*REPLAY)}
            metrics = ("correct_eos","eos","wrong_eos","missing_eos","cap","strict_boxed","numeric_correct")
            arms = {a:{m:sum(y[p,d,a][m] for p,d in pairs) for m in metrics} for a in (NATURAL,DIRECT,*REPLAY)}
            comparisons = {}
            for a,b in ((CONV,NATIVE),(BOTH,NATIVE),(CONV,BOTH),(CONV,DROP),(CONV,DIRECT),(CONV,NATURAL)):
                diffs = {}
                for pid,d in pairs:
                    diffs.setdefault(pid,[]).append(int(y[pid,d,a]["correct_eos"])-int(y[pid,d,b]["correct_eos"]))
                comparisons[a+"-"+b] = paired_interval(diffs)
            table[name] = {"prompt_draws":len(pairs),"prompts":len({p for p,_ in pairs}),
                           "arms":arms,"correct_eos_contrasts":comparisons}
        windows[str(budget)] = table
    return {"phase":phase,"draws":len(draws),"coverage":{"eligible":len(selected),"attempted":len(population)},
            "landmarks":[check[k] for k in sorted(check)],"windows":windows,
            "worker_seconds":seconds,"verified_shards":files}


class CheckpointQueue(Supervisor):
    def __init__(self,args):
        args.role,args.output = "queue",args.root
        super().__init__(args)
        self.worker = Path(__file__).resolve()
        self.env["PYTHONPATH"] = args.runtime_path

    def phase(self,phase,parts):
        a = self.args
        plans = []
        for i in range(parts):
            out = self.lane/f"{phase}-{i}.jsonl"
            plans.append((i,out,[self.worker,"run","--phase",phase,"--root",self.root,"--model",a.model,
                                "--data",a.data,"--output",out,"--part",i,"--deadline",a.deadline]))
        results = self.run_batch(phase,plans)
        save(self.lane/f"{phase}-done.json",{"outputs":[[str(p),d] for p,d in results]})
        out = self.root/f"{phase}-summary.json"
        cmd = [sys.executable,"-B",self.worker,"score","--phase",phase,"--root",self.root,
               "--model",a.model,"--output",out]
        subprocess.run(cmd,env={**self.env,"CUDA_VISIBLE_DEVICES":""},check=True,timeout=240)
        return out,load(out)

    def supervise(self):
        try:
            self.status(cohort_sha256=self.args.cohort_sha256)
            pilot,result = self.phase("pilot",1)
            self.status("pilot_completed",pilot_sha256=sha(pilot),pilot_coverage=result["coverage"])
            if result["coverage"]["eligible"] < PILOT//2:
                self.status("pilot_low_coverage")
                return
            # Eval has four times as many prompt-draws per GPU as the 8-row pilot.
            if not self.bounded(max(1200,5*max(result["worker_seconds"])+240)):
                self.status("budget_complete")
                return
            out,report = self.phase("eval",2)
            self.status("completed",summary_sha256=sha(out),eval_coverage=report["coverage"])
        except Exception as exc:
            self.status("failed",error=str(exc))
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("freeze","run","score","launch","supervise"))
    p.add_argument("--phase",choices=("pilot","eval"))
    for name in ("root","model","data","output","overlay","stable","source"):
        p.add_argument("--"+name,type=Path)
    for name in ("runtime-path","devices","host","cohort-sha256"):
        p.add_argument("--"+name)
    p.add_argument("--part",type=int,default=0)
    p.add_argument("--deadline",type=int)
    a = p.parse_args()
    if a.action == "freeze":
        if a.output.exists():
            raise FileExistsError(a.output)
        save(a.output,make_panel())
        return
    if a.action == "run":
        worker(a)
        return
    if a.action == "score":
        if a.output.exists():
            raise FileExistsError(a.output)
        from transformers import AutoTokenizer
        save(a.output,report(a.root,a.phase,AutoTokenizer.from_pretrained(a.model,local_files_only=True)))
        return
    q = CheckpointQueue(a)
    if socket.gethostname() != a.host or len(q.devices) != 2 or len(set(q.devices)) != 2 or not q.bounded(900):
        raise ValueError("unexpected checkpoint replay allocation")
    if (sha(a.root/"cohort.json") != a.cohort_sha256
            or any(sha(q.worker.parent/n) != h for n,h in load(a.source).items())):
        raise ValueError("checkpoint replay source or cohort changed")
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


if __name__=="__main__":
    main()
