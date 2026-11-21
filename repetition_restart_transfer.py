"""Replicate the restart matrix with Qwen3-1.7B and its own checkpoint identity."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

from repetition_baseline import sha
from repetition_budget_run import tokens
from repetition_mediation import SAMPLING, token_groups
from repetition_mediation_queue import check_devices, load, save
from repetition_restart import ARMS, BATCH, batch_run, prepare_items
from repetition_restart_queue import RestartQueue
from repetition_state_value import Runtime, make_timing_policy

SMOKE_BUDGET = 768


def verify_checkpoint(model, expected, *, weight_bytes):
    actual = {}
    for name, fingerprint in expected["files"].items():
        p = model/name
        data = p.read_bytes()
        content_sha = hashlib.sha256(data).hexdigest()
        if "git_blob_sha1" in fingerprint:
            ok = hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest()==fingerprint["git_blob_sha1"]
        else:
            ok = content_sha == fingerprint["sha256"]
        if not ok:
            raise ValueError(f"Qwen3 public blob digest mismatch: {name}")
        actual[name] = content_sha
    for name, fingerprint in expected["weights"].items():
        p = model/name
        info = p.stat()
        if weight_bytes and sha(p) != fingerprint:
            raise ValueError(f"Qwen3 public checkpoint digest mismatch: {name}")
        actual[name] = {"sha256":fingerprint,"bytes":info.st_size,"mtime_ns":info.st_mtime_ns}
    return actual


def freeze(source, dev, output):
    frozen, parent = load(source),load(dev)
    eval_rows, pilot_rows = frozen["cases"],parent["pilot"][:8]
    eval_ids, pilot_ids = ({x["row"]["id"] for x in rows} for rows in (eval_rows,pilot_rows))
    if len(eval_ids)!=32 or len(pilot_ids)!=8 or pilot_ids & eval_ids or output.exists():
        raise ValueError("transfer needs the same 32 evaluation IDs and disjoint canary")
    save(output, {"eval":eval_rows,"pilot":pilot_rows,"dataset":parent["dataset"],"switch":512,"budget":4096,
                  "nested_total_budgets":[2048],"source_cohort_sha256":sha(source),"dev_cohort_sha256":sha(dev),
                  "scope":"Qwen3-1.7B transfer on the same questions as the Qwen3.5 cap-sensitivity run"})


class Qwen3Runtime(Runtime):
    def __init__(self, args):
        import torch
        import tokenizers
        import transformers

        if torch.cuda.device_count()!=1:
            raise ValueError("exactly one assigned GPU is required")
        provenance = load(args.root/"checkpoint.json")
        expected = provenance["expected_public_snapshot"]
        if verify_checkpoint(args.model,expected,weight_bytes=False)!=provenance["verified_files"]:
            raise ValueError("Qwen3 checkpoint differs from the byte-verified snapshot")
        self.torch,self.transformers = torch,transformers
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model,local_files_only=True)
        self.groups = token_groups(self.tokenizer)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model,local_files_only=True,dtype=torch.bfloat16
        ).to("cuda:0").eval()
        if self.model.config.model_type!="qwen3" or type(self.model).__name__!="Qwen3ForCausalLM":
            raise ValueError("transfer requires the pinned dense Qwen3 model")
        eos = self.model.generation_config.eos_token_id
        self.eos = set(eos if isinstance(eos,(list,tuple)) else [eos])
        chat = self.tokenizer.encode("<|im_end|>",add_special_tokens=False)
        if not self.eos or None in self.eos or len(chat)!=1 or self.groups["think"][0] in self.eos:
            raise ValueError("invalid model-specific EOS or reasoning marker")
        self.pad = self.tokenizer.pad_token_id
        if self.pad is None:
            self.pad = min(self.eos)
        self.copy_eos = self.eos | set(chat)
        self.policy,self.stop_after = make_timing_policy(torch,transformers,self.groups)
        self.defaults = ({"use_model_defaults":False}
                         if "use_model_defaults" in inspect.signature(self.model.generate).parameters else {})
        self.manifest = {"runtime":{"torch":str(torch.__version__),"transformers":transformers.__version__,
                                   "tokenizers":tokenizers.__version__},
                         "checkpoint_revision":expected["revision"],"checkpoint_provenance_sha256":sha(args.root/"checkpoint.json"),
                         "eos_ids":sorted(self.eos),"chat_end_id":chat[0],"pad_id":self.pad,
                         "think_end_id":self.groups["think"][0],"sampling":SAMPLING,
                         "tokenizer_sha256":provenance["verified_files"]["tokenizer.json"]}


def run(a):
    cohort = a.root/"cohort.json"
    plan = load(cohort)
    smoke = a.phase=="smoke"
    budget,switch = (SMOKE_BUDGET if smoke else plan["budget"]),plan["switch"]
    selected = plan["pilot"] if smoke else plan["eval"][a.part::a.partitions]
    r = Qwen3Runtime(a)
    items = prepare_items(r,selected)
    with a.output.open("x") as f,r.torch.inference_mode():
        def emit(x):
            print(json.dumps(x,allow_nan=False),file=f,flush=True)
        emit({"kind":"manifest","part":a.part,"partitions":a.partitions,"phase":a.phase,
              "draw":a.draw,"arms":ARMS,"budget":budget,"switch":switch,
              "cohort_sha256":sha(cohort),"worker_sha256_policy":sha(Path(__file__)),
              "ids":[r["row"]["id"] for r in items],**r.manifest})
        for start in range(0,len(items),BATCH):
            if time.time() >= a.deadline-120:
                raise TimeoutError("insufficient transfer runtime")
            rows,meta = batch_run(r,items[start:start+BATCH],a.draw,budget,switch,smoke)
            emit(meta)
            for row in rows:
                emit(row)
            print(f"Qwen3 {a.phase} draw={a.draw} prompts={start+len(meta['ids'])}/{len(items)}",flush=True)
        emit({"kind":"complete"})


def verify_canary(path, cohort):
    with path.open() as f:
        head,*rows,footer = map(json.loads,f)
    ids = [x["row"]["id"] for x in load(cohort)["pilot"]]
    if (head["cohort_sha256"]!=sha(cohort) or head["ids"]!=ids or head["budget"]!=SMOKE_BUDGET
            or head["phase"]!="smoke" or footer["kind"]!="complete"):
        raise ValueError("transfer canary did not cover its fixed panel")
    data = [x for x in rows if x["kind"]=="rollout"]
    if {(r["id"],r["arm"]) for r in data}!={(p,a) for p in ids for a in ARMS}:
        raise ValueError("transfer canary arm coverage changed")
    restarted = [r for r in data if r["arm"]=="switch_restart" and r["switched"]]
    if not restarted or not all(r["nested_parity"] for r in restarted):
        raise ValueError("explicit Qwen3 restart does not reproduce the direct cut")
    if any(not all(r["short_trunk_exact"]) for r in rows if r["kind"]=="batch"):
        raise ValueError("Qwen3 canary common trunk mismatched")
    if any(len(tokens(r))>SMOKE_BUDGET or
           (r["eos_token"] is not None and r["eos_token"] not in head["eos_ids"]) for r in data):
        raise ValueError("transfer EOS/budget canary failed")
    return {"switched":len(restarted),"prompts":len(ids),"think_end_id":head["think_end_id"],
            "chat_end_id":head["chat_end_id"],"eos_ids":head["eos_ids"]}


class TransferQueue(RestartQueue):
    worker_args = ("run",)

    def __init__(self, args):
        super().__init__(args)
        self.worker = Path(__file__).resolve()

    def supervise(self):
        try:
            self.status(cohort_sha256=self.args.cohort_sha256)
            smoke = self.lane/"smoke.jsonl"
            argv = [self.worker,"run","--phase","smoke","--root",self.root,"--model",self.args.model,
                    "--output",smoke,"--deadline",self.args.deadline]
            (path,fingerprint), = self.run_batch("canary",[(0,smoke,argv)])
            audit = verify_canary(path,self.root/"cohort.json")
            self.status("canary_passed",canary_sha256=fingerprint,canary=audit)
            completed = 0
            reserve = 1100
            for draw in range(2):
                if not self.bounded(reserve):
                    break
                start = time.monotonic()
                self.wave("eval",draw)
                completed += 1
                reserve = 1.35*(time.monotonic()-start)+120
            if completed:
                self.score("eval")
            self.status("completed" if completed==2 else "budget_complete",completed_eval_draws=completed)
        except Exception as exc:
            self.status("failed",error=str(exc))
            raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("run","launch","supervise"))
    p.add_argument("--phase",choices=("smoke","eval"))
    for name in ("root","model","data","output","overlay","stable","source"):
        p.add_argument("--"+name,type=Path)
    for name in ("runtime-path","devices","host","cohort-sha256","checkpoint-sha256"):
        p.add_argument("--"+name)
    for name,default in (("part",0),("partitions",1),("draw",0),("deadline",None)):
        p.add_argument("--"+name,type=int,default=default)
    a=p.parse_args()
    if a.action=="run":
        run(a)
        return
    q=TransferQueue(a)
    if socket.gethostname()!=a.host or len(set(q.devices))!=len(q.devices) or len(q.devices)!=2 or not q.bounded(1500):
        raise ValueError("transfer host, device list or runtime budget differs")
    if (sha(a.root/"cohort.json")!=a.cohort_sha256 or sha(a.root/"checkpoint.json")!=a.checkpoint_sha256
            or any(sha(q.worker.parent/n)!=expected for n,expected in load(a.source).items())):
        raise ValueError("transfer source or manifest changed")
    if a.action=="supervise":
        q.supervise()
        return
    check_devices(q.devices)
    q.lane.mkdir()
    with (q.lane/"supervisor.log").open("x") as log:
        child=subprocess.Popen([sys.executable,"-B",__file__,"supervise",*sys.argv[2:]],env=q.env,
                               start_new_session=True,stdout=log,stderr=subprocess.STDOUT)
    print(json.dumps({"supervisor":child.pid,"root":str(q.root),"deadline":a.deadline,"devices":q.devices,
                      "source_sha256":sha(a.source),"checkpoint_sha256":a.checkpoint_sha256}))


if __name__=="__main__":
    main()
