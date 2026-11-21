"""Compare the published Astra BF16 edit at native and fixed replay boundaries."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

from repetition_baseline import sha
from repetition_budget_run import packed_hash, tokens
from repetition_canonical import paired_interval, seed_for
from repetition_mediation_queue import Supervisor, check_devices, load, save
from repetition_replay import terminal_cycle
from repetition_restart import reaches_switch, score_policy
from repetition_restart_score import read_run
from repetition_sampler_reach import pack_at
from repetition_state_data import THINK_END, copies, keyed
from repetition_state_value import Runtime, inputs_for, post_seed

PUBLIC_COMMIT = "4d09ba6d6a7e891dd8063974dfe81b3d78e3494f"
PATCH_SHA = "e92a92aafd1f1068b4fd94aa0fbe081ba3ddea2047ee7bcac7161d861be299e7"
MANIFEST_SHA = "83185c1db46e3dafb34af645f8c5ce033c09cc99d4512b9e1b8e3aba9d269225"
VARIANTS = ("base","astra")
MODES = ("native","fixed_switch")
SWITCH, TOTAL, BATCH = 512, 2048, 8


def patch_arrays(artifact, manifest):
    import numpy as np

    if sha(artifact)!=PATCH_SHA or sha(manifest)!=MANIFEST_SHA:
        raise ValueError("published Astra source hash mismatch")
    info = load(manifest)
    if info["changed_tensors"]!=2 or info["changed_entries_total"]!=130188:
        raise ValueError("unexpected scope of the published Astra edit")
    with np.load(artifact,allow_pickle=False) as source:
        if set(source.files)!={f"t{i}_{suffix}" for i in range(2) for suffix in ("idx","val")}:
            raise ValueError("unexpected contents in the published edit")
        arrays = []
        for i,spec in enumerate(info["tensors"]):
            indices,values = source[f"t{i}_idx"],source[f"t{i}_val"]
            count = spec["changed_entries"]
            rows,cols = spec["shape"]
            if (indices.dtype!=np.int32 or values.dtype!=np.uint16 or indices.shape!=(count,) or values.shape!=(count,)
                    or np.any(indices<0) or np.any(indices>=rows*cols) or np.unique(indices).size!=count
                    or np.unique(indices//cols).size!=spec["changed_rows"]
                    or np.unique(indices%cols).size!=spec["changed_columns"]):
                raise ValueError("published replacement indices do not match metadata")
            arrays.append((spec,indices,values))
    return arrays


def patch_tensors(torch, tensors, edits, *, apply=False):
    import numpy as np

    staged = []
    for spec,indices,values in edits:
        name = spec["name"]
        tensor = tensors[name]
        if tensor.dtype!=torch.bfloat16 or list(tensor.shape)!=spec["shape"]:
            raise ValueError(f"wrong published tensor shape or dtype: {name}")
        before = tensor.detach().contiguous().view(torch.int16).cpu().numpy().view(np.uint16)
        if hashlib.sha256(before.tobytes()).hexdigest()!=spec["base_sha256"]:
            raise ValueError(f"base tensor bytes do not match publication: {name}")
        after = before.copy()
        flat = after.reshape(-1)
        if (flat[indices]==values).any():
            raise ValueError(f"published replacement contains unchanged bits: {name}")
        flat[indices] = values
        if hashlib.sha256(after.tobytes()).hexdigest()!=spec["edited_sha256"]:
            raise ValueError(f"edited tensor bytes do not match publication: {name}")
        staged.append((tensor,after,spec))
    verified = []
    for tensor,after,spec in staged:
        if apply:
            tensor.copy_(torch.from_numpy(after.view(np.int16)).view(torch.bfloat16).to(tensor.device))
            written = tensor.detach().contiguous().view(torch.int16).cpu().numpy()
            if hashlib.sha256(written.tobytes()).hexdigest()!=spec["edited_sha256"]:
                raise ValueError("loaded BF16 weight differs from the published edit")
        verified.append({k:spec[k] for k in ("name","shape","changed_entries","base_sha256","edited_sha256")})
    return verified


def freeze(parent, subset, output):
    plan,prompts,raw,meta,receipts,draws = read_run(parent,"eval")
    _,dev,dev_raw,dev_meta,dev_receipts,_ = read_run(parent,"pilot")
    selected = [x["row"]["id"] for x in load(subset)["cases"]]
    pilot = [x["row"]["id"] for x in plan["pilot"][:4]]
    if (len(set(selected))!=32 or not set(selected)<=set(prompts) or set(selected)&set(pilot)
            or output.exists() or plan["switch"]!=SWITCH or plan["budget"]!=TOTAL or draws!=2):
        raise ValueError("published Astra comparison requires the frozen parent protocol")

    def panel(ids, cases, observations, batches, draw):
        hashes = {p:h for b in batches for p,h in zip(b["ids"],b["render_sha256"],strict=True)}
        result = []
        for p in ids:
            record = observations[p,draw,"native"]
            if not reaches_switch(record,SWITCH):
                raise ValueError("required native trunk did not reach 512 tokens")
            result.append({"row":cases[p],"render_sha256":hashes[p],"completion_prefix":record["tokens"][:SWITCH]})
        return result

    used = {row["row"]["gold"].split()[0] for row in copies("dev")+copies("test",3)}
    words = {a+b+c for a in "mnpbrst" for b in "aeiou" for c in "lnmrkst"}-used
    nonce = sorted(words,key=lambda w:keyed(w,"published-astra-copy-v1"))[:16]
    copy = [{"row":{"id":f"astra-copy:{i}","task":"copy",
                    "prompt":f"Write the word {w} exactly 3 times, separated by one space. Output only the requested text.",
                    "gold":" ".join([w]*3)},"completion_prefix":[]} for i,w in enumerate(nonce)]
    save(output,{"draw_inputs":[panel(selected,prompts,raw,meta,d) for d in range(draws)],
                 "canary":panel(pilot,dev,dev_raw,dev_meta,0),"copy":copy,"parent":str(parent),
                 "parent_cohort_sha256":sha(parent/"cohort.json"),"parent_eval_receipts":receipts,
                 "parent_pilot_receipts":dev_receipts,"subset_sha256":sha(subset),
                 "scope":"published fixed edit on the prior 32-question MATH-500 numeric subset"})


def run(a):
    cohort = a.root/"cohort.json"
    plan = load(cohort)
    canary = a.phase=="smoke"
    math_rows = plan["canary"] if canary else plan["draw_inputs"][a.draw][a.part::a.partitions]
    copy_rows = plan["copy"][:2] if canary else plan["copy"][a.part::a.partitions] if a.draw==0 else []
    r = Runtime(a)
    edits = patch_arrays(a.artifact,a.publisher_manifest)
    items = inputs_for(r.tokenizer,math_rows)
    copied = inputs_for(r.tokenizer,copy_rows)
    native_cap,forced_cap = (128,63) if canary else (TOTAL,TOTAL-SWITCH-1)
    with a.output.open("x") as out,r.torch.inference_mode():
        def emit(row):
            print(json.dumps(row,allow_nan=False),file=out,flush=True)
        emit({"kind":"manifest","phase":a.phase,"draw":a.draw,"part":a.part,
              "cohort_sha256":sha(cohort),"published_commit":PUBLIC_COMMIT,
              "patch_sha256":PATCH_SHA,"publisher_manifest_sha256":MANIFEST_SHA,
              "native_cap":native_cap,"forced_post_cap":forced_cap,"worker_sha256_published":sha(Path(__file__)),
              "ids":[x["row"]["id"] for x in items],"copy_ids":[x["row"]["id"] for x in copied],**r.manifest})
        for variant in VARIANTS:
            if variant=="astra":
                patch = patch_tensors(r.torch,dict(r.model.named_parameters()),edits,apply=True)
                emit({"kind":"edit-check","tensors":patch})
            for start in range(0,len(items),BATCH):
                if time.time() >= a.deadline-120:
                    raise TimeoutError("published Astra queue reached its deadline")
                group = [x | {"presence_seed_ids":[]} for x in items[start:start+BATCH]]
                questions = [x | {"native_ids":x["native_ids"][:-SWITCH]} for x in group]
                kept = r.pack([x["native_ids"] for x in group])
                width = kept["input_ids"].shape[1]
                plain = pack_at(r,[x["native_ids"] for x in questions],width)
                for mode in MODES:
                    batch,packed,action,cap = ((questions,plain,"native",native_cap) if mode=="native"
                                               else (group,kept,"e0",forced_cap))
                    for result,item in zip(r.generate(batch,action,a.draw,cap,packed=packed),batch,strict=True):
                        max_tokens = native_cap if mode=="native" else forced_cap+1
                        if len(tokens(result))>max_tokens:
                            raise ValueError("published rollout exceeded its continuation budget")
                        emit(result | {"kind":"rollout","variant":variant,"arm":mode,"input_sha256":packed_hash(item["native_ids"]),
                                       "spent_before_forced":0 if mode=="native" else SWITCH})
                print(f"Astra {a.phase} {variant} draw={a.draw} prompts={start+len(group)}/{len(items)}",flush=True)
            if copied:
                for result in r.generate(copied,"no_think",a.draw,192):
                    emit(result | {"kind":"copy","variant":variant})
        emit({"kind":"complete"})


def report(root,tokenizer):
    cohort = root/"cohort.json"
    plan = load(cohort)
    cases = {x["row"]["id"]:x["row"] for x in plan["draw_inputs"][0]}
    parent = Path(plan["parent"])
    _,_,b,_,verified,draws = read_run(parent,"eval")
    if verified!=plan["parent_eval_receipts"] or sha(parent/"cohort.json")!=plan["parent_cohort_sha256"]:
        raise ValueError("published Astra comparison lost its parent trace provenance")
    [turn] = tokenizer.encode("<|im_end|>",add_special_tokens=False)
    raw,copy,shards = {},{},[]
    paths = sorted((root/"queue").glob("draw*-done.json"))
    if not paths:
        raise ValueError("missing checked published rollout receipts")
    for receipt in paths:
        for filename,fingerprint in load(receipt)["outputs"]:
            file = Path(filename)
            if sha(file)!=fingerprint:
                raise ValueError("published rollout shard checksum mismatch")
            with file.open() as f:
                h,*records,end = map(json.loads,f)
            if (h["cohort_sha256"]!=sha(cohort) or h["patch_sha256"]!=PATCH_SHA or h["phase"]!="eval"
                    or h["native_cap"]!=TOTAL or h["forced_post_cap"]!=TOTAL-SWITCH-1 or end["kind"]!="complete"
                    or sum(r["kind"]=="edit-check" for r in records)!=1):
                raise ValueError("published rollout manifest mismatch")
            for row in records:
                if row["kind"]=="edit-check":
                    continue
                p,d,a,v = row["id"],row["draw"],row["arm"],row["variant"]
                if d!=h["draw"] or row["seed"]!=seed_for(p,d) or row["post_seed"]!=post_seed(p,d):
                    raise ValueError("published edit outcomes have different seeds")
                if row["kind"]=="copy":
                    if (p,v) in copy or p not in h["copy_ids"]:
                        raise ValueError("unexpected published copy outcome")
                    copy[p,v] = row
                    continue
                key = p,d,a,v
                consumed = len(tokens(row))+row["spent_before_forced"]
                if (key in raw or p not in h["ids"] or consumed>TOTAL or
                        (row["eos_token"] is not None and row["eos_token"] not in h["eos_ids"])):
                    raise ValueError("invalid published edit outcome")
                raw[key] = row
            shards.append([filename,fingerprint])
    if set(raw)!={(p,d,m,v) for p in cases for d in range(len(paths)) for m in MODES for v in VARIANTS}:
        raise ValueError("incomplete paired published rollout matrix")
    for p,d,a,v in raw:
        if raw[p,d,a,v]["input_sha256"]!=raw[p,d,a,"base"]["input_sha256"]:
            raise ValueError("base and published edit saw different token contexts")
    scored = {k:score_policy(cases[k[0]],row,tokenizer,turn) for k,row in raw.items()}
    arms,effects = {},{}
    for mode in MODES:
        arms[mode] = {}
        for variant in VARIANTS:
            selected = [k for k in scored if k[2:]==(mode,variant)]
            arms[mode][variant] = {m:sum(scored[k][m] for k in selected) for m in
                                   ("correct_eos","eos","wrong_eos","missing_eos","cap","visible_boxed")}
            arms[mode][variant].update(n=len(selected),
                                       closed=sum(THINK_END in raw[k]["tokens"] for k in selected),
                                       terminal_cycle=sum(terminal_cycle(raw[k]["tokens"]) is not None for k in selected))
        effects[mode] = paired_interval({
            p:[int(scored[p,d,mode,"astra"]["correct_eos"])-int(scored[p,d,mode,"base"]["correct_eos"])
               for d in range(len(paths))] for p in cases})
    copy_cases = {x["row"]["id"] for x in plan["copy"]}
    if set(copy)!={(p,v) for p in copy_cases for v in VARIANTS}:
        raise ValueError("missing published edit copy comparison")
    copy_status = {v:{"n":len(copy_cases),"exact_eos":sum(copy[p,v]["success"] for p in copy_cases),
                     "exact":sum(copy[p,v]["copy_exact"] for p in copy_cases),
                     "eos":sum(copy[p,v]["eos_token"] is not None for p in copy_cases)} for v in VARIANTS}
    copy_status["astra_minus_base"] = paired_interval({
        p:[int(copy[p,"astra"]["success"])-int(copy[p,"base"]["success"])] for p in copy_cases})
    direct = [score_policy(cases[p],b[p,d,"no_think"],tokenizer,turn) for p in cases for d in range(len(paths))]
    return {"arms":arms,"astra_minus_base":effects,"copy":copy_status,"draws":len(paths),"prompts":len(cases),
            "parent_no_think":{"n":len(direct),"correct_eos":sum(r["correct_eos"] for r in direct),
                               "eos":sum(r["eos"] for r in direct),"wrong_eos":sum(r["wrong_eos"] for r in direct)},
            "source_shards":shards,"publisher_sha256":PATCH_SHA,"publisher_commit":PUBLIC_COMMIT}


class PublishedQueue(Supervisor):
    def __init__(self,args):
        args.role,args.output = "queue",args.root
        super().__init__(args)
        self.worker = Path(__file__).resolve()
        self.env["PYTHONPATH"] = args.runtime_path

    def plan(self,i,phase,draw,parts):
        a = self.args
        out = self.lane/f"{phase}-{draw}-{i}.jsonl"
        argv = [self.worker,"run","--phase",phase,"--root",self.root,"--model",a.model,"--data",a.data,
                "--artifact",a.artifact,"--publisher-manifest",a.publisher_manifest,"--output",out,
                "--draw",draw,"--part",i,"--partitions",parts,"--deadline",a.deadline]
        return i,out,argv

    def supervise(self):
        try:
            self.status(cohort_sha256=self.args.cohort_sha256)
            (canary,checksum), = self.run_batch("canary",[self.plan(0,"smoke",0,1)])
            with canary.open() as f:
                h,*rows,end = map(json.loads,f)
            if (h["phase"]!="smoke" or end["kind"]!="complete" or h["cohort_sha256"]!=self.args.cohort_sha256
                    or sum(r["kind"]=="edit-check" for r in rows)!=1
                    or sum(r["kind"]=="rollout" for r in rows)!=16 or sum(r["kind"]=="copy" for r in rows)!=4):
                raise ValueError("published Astra canary did not verify all paired arms")
            self.status("canary_passed",canary_sha256=checksum)
            completed,reserve = 0,1200
            for draw in range(2):
                if not self.bounded(reserve):
                    break
                start = time.monotonic()
                result = self.run_batch(f"draw{draw}",[self.plan(i,"eval",draw,2) for i in range(2)])
                save(self.lane/f"draw{draw}-done.json",{"outputs":[[str(p),fp] for p,fp in result]})
                completed += 1
                reserve = 1.35*(time.monotonic()-start)+120
            if completed:
                score = self.root/"summary.json"
                argv = [sys.executable,"-B",self.worker,"score","--root",self.root,"--model",self.args.model,"--output",score]
                subprocess.run(argv,env={**self.env,"CUDA_VISIBLE_DEVICES":""},check=True,timeout=150)
                self.status("completed" if completed==2 else "budget_complete",summary_sha256=sha(score),completed_eval_draws=completed)
            else:
                self.status("budget_complete",completed_eval_draws=0)
        except Exception as exc:
            self.status("failed",error=str(exc))
            raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("run","score","launch","supervise"))
    p.add_argument("--phase",choices=("smoke","eval"))
    for name in ("root","model","data","output","overlay","stable","source","artifact","publisher-manifest"):
        p.add_argument("--"+name,type=Path)
    for name in ("runtime-path","devices","host","cohort-sha256"):
        p.add_argument("--"+name)
    for name,default in (("part",0),("partitions",1),("draw",0),("deadline",None)):
        p.add_argument("--"+name,type=int,default=default)
    a=p.parse_args()
    if a.action=="run":
        run(a)
        return
    if a.action=="score":
        from transformers import AutoTokenizer
        result = report(a.root,AutoTokenizer.from_pretrained(a.model,local_files_only=True))
        if a.output.exists():
            raise FileExistsError(a.output)
        save(a.output,result)
        return
    queue=PublishedQueue(a)
    if socket.gethostname()!=a.host or len(queue.devices)!=2 or len(set(queue.devices))!=2 or not queue.bounded(1600):
        raise ValueError("wrong published edit host, devices or budget")
    if (sha(a.root/"cohort.json")!=a.cohort_sha256 or sha(a.artifact)!=PATCH_SHA or sha(a.publisher_manifest)!=MANIFEST_SHA
            or any(sha(queue.worker.parent/n)!=h for n,h in load(a.source).items())):
        raise ValueError("published edit input or source changed")
    if a.action=="supervise":
        queue.supervise()
        return
    check_devices(queue.devices)
    queue.lane.mkdir()
    with (queue.lane/"supervisor.log").open("x") as log:
        child = subprocess.Popen([sys.executable,"-B",__file__,"supervise",*sys.argv[2:]],
                                 start_new_session=True,env=queue.env,stdout=log,stderr=subprocess.STDOUT)
    print(json.dumps({"supervisor":child.pid,"root":str(a.root),"deadline":a.deadline,"devices":queue.devices,
                      "source_sha256":sha(a.source),"patch_sha256":PATCH_SHA}))


if __name__=="__main__":
    main()
