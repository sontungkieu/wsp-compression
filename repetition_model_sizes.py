"""Compare small Qwen checkpoints under the same native and direct decoding budgets."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

from repetition_baseline import first_recurrent, sha
from repetition_budget_run import packed_hash, tokens
from repetition_budget_score import cut
from repetition_canonical import paired_interval, seed_for
from repetition_mediation_queue import Supervisor, check_devices, load, save
from repetition_phase_trace import first_long_cycle
from repetition_replay import terminal_cycle
from repetition_restart import BATCH, prepare_items, score_policy
from repetition_restart_transfer import Qwen3Runtime, verify_checkpoint
from repetition_sampler_reach import pack_at
from repetition_state_value import post_seed

ARMS = ("native","no_think")
CAPS = (2048,4096)
SMOKE = 128


def freeze(depth,broad,output):
    subset, parent = load(depth),load(broad)
    rows, canary = subset["cases"],parent["pilot"][:4]
    ids, dev = ({x["row"]["id"] for x in group} for group in (rows,canary))
    if output.exists() or len(ids)!=32 or len(dev)!=4 or ids&dev:
        raise ValueError("size comparison needs D's 32 IDs and a disjoint canary")
    save(output,{"cases":rows,"canary":canary,"source_depth_sha256":sha(depth),
                 "source_pilot_sha256":sha(broad),"arms":ARMS,"caps":CAPS,"draw":0})


def run(a):
    cohort = a.root/"cohort.json"
    plan = load(cohort)
    runtime = Qwen3Runtime(a)
    [marker] = runtime.groups["think"]
    part = a.output.with_suffix(".partial.jsonl")
    if a.output.exists():
        raise FileExistsError(a.output)
    with part.open("x") as stream,runtime.torch.inference_mode():
        def emit(row):
            print(json.dumps(row,allow_nan=False),file=stream,flush=True)
        emit({"kind":"manifest","model":str(a.model),"cohort_sha256":sha(cohort),
              "size_worker_sha256":sha(Path(__file__)),"draw":0,"caps":CAPS,"arms":ARMS,**runtime.manifest})
        clock = time.monotonic()
        for phase,panel,cap in (("canary",plan["canary"],SMOKE),("eval",plan["cases"],CAPS[-1])):
            items = prepare_items(runtime,panel)
            for i in range(0,len(items),BATCH):
                if time.time() >= a.deadline-120:
                    raise TimeoutError("Qwen size lane reached the deadline")
                group = items[i:i+BATCH]
                width = max(max(len(x["native_ids"]),len(x["no_think_ids"])) for x in group)
                if any(marker in x["native_ids"] or marker not in x["no_think_ids"] for x in group):
                    raise ValueError("no-thinking template did not prefill the actual close marker")
                for arm in ARMS:
                    ids = [x["native_ids" if arm == "native" else "no_think_ids"] for x in group]
                    packed = pack_at(runtime,ids,width)
                    for x,prefix,r in zip(group,ids,runtime.generate(group,arm,0,cap,packed=packed),strict=True):
                        if (len(tokens(r))>cap or (r["eos_token"] is None and len(tokens(r))!=cap)
                                or (r["eos_token"] is not None and r["eos_token"] not in runtime.eos)):
                            raise ValueError("bad close or EOS accounting in Qwen size worker")
                        emit(r | {"phase":phase,"cap_total":cap,"prefix_sha256":packed_hash(prefix),
                                  "render_sha256":x["render_sha256"],"width":width})
            emit({"kind":"phase-complete","phase":phase,"seconds":time.monotonic()-clock})
        emit({"kind":"complete","seconds":time.monotonic()-clock})
    part.replace(a.output)


def score(path,fingerprint,cohort,provenance,tokenizer):
    if sha(path)!=fingerprint:
        raise ValueError("size comparison shard changed")
    plan, cp = load(cohort), load(provenance)
    with path.open() as stream:
        head,*body,tail = map(json.loads,stream)
    [marker] = tokenizer.encode("</think>",add_special_tokens=False)
    [turn] = tokenizer.encode("<|im_end|>",add_special_tokens=False)
    if (tail["kind"]!="complete" or head["cohort_sha256"]!=sha(cohort) or head["caps"]!=list(CAPS)
            or head["size_worker_sha256"]!=sha(Path(__file__)) or head["checkpoint_provenance_sha256"]!=sha(provenance)
            or head["checkpoint_revision"]!=cp["expected_public_snapshot"]["revision"] or head["think_end_id"]!=marker
            or head["chat_end_id"]!=turn):
        raise ValueError("unexpected size comparison receipt")
    declarations = {"canary":plan["canary"],"eval":plan["cases"]}
    expected = {(phase,x["row"]["id"],arm) for phase,instances in declarations.items()
                for x in instances for arm in ARMS}
    raw = {}
    for r in body:
        if r["kind"]=="phase-complete":
            continue
        p,arm,phase = (r[k] for k in ("id","arm","phase"))
        key = (phase,p,arm)
        cap = SMOKE if phase=="canary" else CAPS[-1]
        if (key not in expected or key in raw or r["draw"]!=0 or r["seed"]!=seed_for(p,0)
                or r["post_seed"]!=post_seed(p,0) or r["cap_total"]!=cap or len(tokens(r))>cap
                or (r["eos_token"] is not None and r["eos_token"] not in head["eos_ids"])):
            raise ValueError("invalid size comparison rollout")
        raw[key] = r
    if raw.keys()!=expected:
        raise ValueError("incomplete size comparison rollout matrix")
    rows = {x["row"]["id"]:x["row"] for x in plan["cases"]}
    windows = {}
    for cap in CAPS:
        cells = {}
        for pid,row in rows.items():
            for arm in ARMS:
                r = raw["eval",pid,arm]
                ids,eos = cut(r,"total",cap)
                closed = marker in ids
                before = ids[:ids.index(marker)] if closed else ids
                inspected = before if arm=="native" else ids
                y = score_policy(row,r | {"tokens":ids,"eos_token":r["eos_token"] if eos else None},tokenizer,turn)
                cells[pid,arm] = y | {"generated_close":closed,"exact_cycle_anywhere":first_long_cycle(inspected) is not None,
                                      "terminal_cycle":terminal_cycle(inspected) is not None,
                                      "recurrent_16":first_recurrent(inspected) is not None}
        metrics = ("correct_eos","eos","wrong_eos","missing_eos","cap","strict_boxed","generated_close",
                   "exact_cycle_anywhere","terminal_cycle","recurrent_16")
        windows[str(cap)] = {"arms":{a:{m:sum(cells[p,a][m] for p in rows) for m in metrics} for a in ARMS},
                             "direct_minus_native":paired_interval({p:[int(cells[p,"no_think"]["correct_eos"])-
                                                                            int(cells[p,"native"]["correct_eos"])] for p in rows}),
                             "case_scores":[{"id":p,"arm":a,**y} for (p,a),y in cells.items()]}
    return {"manifest":head,"windows":windows,"verified_shard":[str(path),fingerprint],"worker_seconds":tail["seconds"]}


class SizeQueue(Supervisor):
    def __init__(self,args):
        args.role,args.output = "queue",args.root
        super().__init__(args)
        self.worker = Path(__file__).resolve()
        self.env["PYTHONPATH"] = args.runtime_path
        self.models = load(args.root/"models.json")

    def supervise(self):
        try:
            self.status(cohort_sha256=self.args.cohort_sha256)
            jobs = []
            for index,config in enumerate(self.models):
                out = self.lane/f"model-{index}.jsonl"
                jobs.append((index,out,[self.worker,"run","--root",self.root,"--model",config["path"],
                                       "--checkpoint",config["provenance"],"--output",out,"--deadline",self.args.deadline]))
            results = self.run_batch("eval",jobs,tolerate_failure=True)
            if not results:
                raise RuntimeError("no size comparison worker finished")
            save(self.lane/"eval-done.json",{"outputs":[[str(p),s] for p,s in results]})
            out = self.root/"summary.json"
            cmd = [sys.executable,"-B",self.worker,"score","--root",self.root,"--output",out]
            subprocess.run(cmd,env={**self.env,"CUDA_VISIBLE_DEVICES":""},check=True,timeout=240)
            self.status("completed" if len(results)==len(self.models) else "partial",
                        summary_sha256=sha(out),models_complete=len(results))
        except Exception as exc:
            self.status("failed",error=str(exc))
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("freeze","run","score","launch","supervise"))
    for name in ("root","model","output","overlay","stable","source","depth","broad","checkpoint"):
        p.add_argument("--"+name,type=Path)
    for name in ("runtime-path","devices","host","cohort-sha256","models-sha256"):
        p.add_argument("--"+name)
    p.add_argument("--deadline",type=int)
    a = p.parse_args()
    if a.action=="freeze":
        freeze(a.depth,a.broad,a.output)
        return
    if a.action=="run":
        run(a)
        return
    if a.action=="score":
        if a.output.exists():
            raise FileExistsError(a.output)
        from transformers import AutoTokenizer
        results = {}
        for filename,fingerprint in load(a.root/"queue/eval-done.json")["outputs"]:
            path = Path(filename)
            index = int(path.stem.rsplit("-",1)[1])
            cfg = load(a.root/"models.json")[index]
            tokenizer = AutoTokenizer.from_pretrained(cfg["path"],local_files_only=True)
            results[cfg["label"]] = score(path,fingerprint,a.root/"cohort.json",Path(cfg["provenance"]),tokenizer)
        save(a.output,{"models":results,"cohort_sha256":sha(a.root/"cohort.json")})
        return
    q = SizeQueue(a)
    if socket.gethostname()!=a.host or len(q.devices)!=2 or len(set(q.devices))!=2 or not q.bounded(1200):
        raise ValueError("size comparison allocation differs")
    if (sha(a.root/"cohort.json")!=a.cohort_sha256 or sha(a.root/"models.json")!=a.models_sha256
            or any(sha(q.worker.parent/n)!=h for n,h in load(a.source).items()) or len(q.models) not in (1,2)):
        raise ValueError("size comparison source, cohort or model plan changed")
    for item in q.models:
        checkpoint = Path(item["provenance"])
        data = load(checkpoint)
        if (sha(checkpoint)!=item["provenance_sha256"]
                or verify_checkpoint(Path(item["path"]),data["expected_public_snapshot"],weight_bytes=False)
                !=data["verified_files"]):
            raise ValueError("size comparison checkpoint changed after byte verification")
    if a.action=="supervise":
        q.supervise()
        return
    check_devices(q.devices)
    q.lane.mkdir()
    with (q.lane/"supervisor.log").open("x") as log:
        child=subprocess.Popen([sys.executable,"-B",__file__,"supervise",*sys.argv[2:]],
                               env=q.env,start_new_session=True,stdout=log,stderr=subprocess.STDOUT)
    print(json.dumps({"supervisor":child.pid,"root":str(a.root),"deadline":a.deadline,
                      "devices":q.devices,"source_sha256":sha(a.source),"models":[x["label"] for x in q.models]}))


if __name__=="__main__":
    main()
