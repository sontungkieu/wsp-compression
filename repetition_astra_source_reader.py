"""Cross the checkpoint that generated a scratchpad with the one that reads it."""
import argparse
from decimal import Decimal
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

from repetition_baseline import BOXED, DECIMAL, sha
from repetition_budget_run import packed_hash, tokens
from repetition_canonical import paired_interval, seed_for
from repetition_mediation_queue import Supervisor, check_devices, load, save
from repetition_published_astra import PATCH_SHA, patch_arrays, patch_tensors
from repetition_restart import reaches_switch, score_policy
from repetition_restart_score import read_run
from repetition_state_value import Runtime, THINK_END, inputs_for, post_seed

MODELS = ("base","astra")
LANDMARKS = (512,256)
TOTAL, BLOCK = 2048,4
CELLS = {(reader,source) for reader in MODELS for source in MODELS}
EFFECTS = {
    "source_mean":{("base","astra"):.5,("base","base"):-.5,("astra","astra"):.5,("astra","base"):-.5},
    "reader_mean":{("astra","base"):.5,("base","base"):-.5,("astra","astra"):.5,("base","astra"):-.5},
    "source_reader_interaction":{("astra","astra"):1,("base","astra"):-1,("astra","base"):-1,("base","base"):1},
    "source_with_base_reader":{("base","astra"):1,("base","base"):-1},
    "source_with_astra_reader":{("astra","astra"):1,("astra","base"):-1},
}


def freeze(published, parent, output):
    before = load(published/"cohort.json")
    proof = load(published/"score-recovery.json")
    if (proof["stage"]!="completed_cpu_rescore" or sha(published/"summary.json")!=proof["summary_sha256"]
            or sha(published/"queue"/"status.json")!=proof["initial_worker_status_sha256"] or output.exists()):
        raise ValueError("published scratchpad source was not verified")
    raw = {}
    for filename,digest in proof["raw_shards"]:
        p = Path(filename)
        if sha(p)!=digest:
            raise ValueError("published scratchpad shard hash changed")
        with p.open() as f:
            head,*rows,end = map(json.loads,f)
        if head["patch_sha256"]!=PATCH_SHA or head["cohort_sha256"]!=sha(published/"cohort.json") or end["kind"]!="complete":
            raise ValueError("unexpected published source protocol")
        for row in rows:
            if row["kind"]=="rollout" and row["arm"]=="native":
                key = row["id"],row["draw"],row["variant"]
                if key in raw or row["draw"]!=head["draw"] or row["seed"]!=seed_for(row["id"],row["draw"]):
                    raise ValueError("invalid or duplicate scratchpad source")
                raw[key] = row
    cohorts,excluded = {},{}
    for mark in LANDMARKS:
        cohorts[str(mark)],excluded[str(mark)] = [],[]
        for d,items in enumerate(before["draw_inputs"]):
            panel = []
            for item in items:
                p = item["row"]["id"]
                base,astra = (raw[p,d,v] for v in MODELS)
                if base["input_sha256"]!=astra["input_sha256"]:
                    raise ValueError("source checkpoints did not see the same question")
                if not all(reaches_switch(r,mark) for r in (base,astra)):
                    excluded[str(mark)].append({"id":p,"draw":d,"base_still_thinking":reaches_switch(base,mark),
                                                "astra_still_thinking":reaches_switch(astra,mark)})
                    continue
                panel.append({"row":item["row"],"render_sha256":item["render_sha256"],
                              "prefixes":{v:raw[p,d,v]["tokens"][:mark] for v in MODELS},
                              "question_token_sha256":base["input_sha256"]})
            cohorts[str(mark)].append(panel)
    if excluded["256"] or not any(cohorts["512"]):
        raise ValueError("the preregistered sensitivity landmark is not valid")
    _,dev,dev_raw,meta,receipts,_ = read_run(parent,"pilot")
    hashes = {p:h for b in meta for p,h in zip(b["ids"],b["render_sha256"],strict=True)}
    smoke = []
    for p in list(dev)[:BLOCK]:
        row = dev_raw[p,0,"native"]
        if not reaches_switch(row,512):
            raise ValueError("pilot lacks an identical-source canary state")
        scratch = row["tokens"][:512]
        smoke.append({"row":dev[p],"render_sha256":hashes[p],"prefixes":dict.fromkeys(MODELS,scratch),
                      "question_token_sha256":None})
    save(output,{"landmarks":cohorts,"exclusions":excluded,"canary":smoke,
                 "published_raw_receipts":proof["raw_shards"],"pilot_raw_receipts":receipts,
                 "published_summary_sha256":proof["summary_sha256"]})


def boxed_in_prefix(tokenizer,ids,gold):
    return any(DECIMAL.fullmatch(s.strip()) and Decimal(s.strip().replace(",",""))==gold
               for s in BOXED.findall(tokenizer.decode(ids,skip_special_tokens=False)))


def run(a):
    path = a.root/"cohort.json"
    data = load(path)
    canary = a.phase=="smoke"
    chosen = {512:data["canary"]} if canary else {m:data["landmarks"][str(m)][a.draw][a.part::a.partitions] for m in LANDMARKS}
    r = Runtime(a)
    edits = patch_arrays(a.artifact,a.publisher_manifest)
    with a.output.open("x") as out,r.torch.inference_mode():
        def emit(x):
            print(json.dumps(x,allow_nan=False),file=out,flush=True)
        emit({"kind":"manifest","phase":a.phase,"draw":a.draw,"part":a.part,"cohort_sha256":sha(path),
              "landmarks":{m:[x["row"]["id"] for x in panel] for m,panel in chosen.items()},
              "patch_sha256":PATCH_SHA,"worker_sha256_source_reader":sha(Path(__file__)),**r.manifest})
        for reader in MODELS:
            if reader=="astra":
                emit({"kind":"edit-check","tensors":patch_tensors(r.torch,dict(r.model.named_parameters()),edits,apply=True)})
            for mark,panel in chosen.items():
                cap = 64 if canary else TOTAL-mark-1
                for start in range(0,len(panel),BLOCK):
                    if time.time()>=a.deadline-120:
                        raise TimeoutError("source-reader run reached the hard deadline")
                    group = panel[start:start+BLOCK]
                    flat = [{"row":x["row"],"completion_prefix":x["prefixes"][source],"render_sha256":x["render_sha256"]}
                            for x in group for source in MODELS]
                    batch = [x | {"presence_seed_ids":[]} for x in inputs_for(r.tokenizer,flat)]
                    decoded = r.generate(batch,"e0",a.draw,cap)
                    equal,paired = 0,0
                    for j,item in enumerate(group):
                        if item["prefixes"]["base"]==item["prefixes"]["astra"]:
                            equal += 1
                            paired += tokens(decoded[2*j])==tokens(decoded[2*j+1])
                    if equal!=paired:
                        raise ValueError("identical source prefixes did not produce identical responses for one reader")
                    emit({"kind":"audit","reader":reader,"landmark":mark,"identical_inputs":equal,"identical_responses":paired})
                    for j,(row,x) in enumerate(zip(decoded,batch,strict=True)):
                        item = group[j//2]
                        q = packed_hash(x["native_ids"][:-mark])
                        if item["question_token_sha256"] is not None and q!=item["question_token_sha256"]:
                            raise ValueError("replay changed the source question tokens")
                        emit(row | {"kind":"rollout","reader":reader,"source":MODELS[j%2],"landmark":mark,
                                    "post_cap":cap,"prefix_sha256":packed_hash(x["completion_prefix"]),
                                    "correct_box_in_source":boxed_in_prefix(r.tokenizer,x["completion_prefix"],Decimal(x["row"]["gold"]))})
                    print(f"source-reader {a.phase} {reader} t={mark} draw={a.draw}: {start+len(group)}/{len(panel)}",flush=True)
        emit({"kind":"complete"})


def summarize(root,tokenizer):
    cohort = root/"cohort.json"
    config = load(cohort)
    paths = sorted((root/"queue").glob("draw*-done.json"))
    if not paths:
        raise ValueError("no complete source-reader draw")
    [turn] = tokenizer.encode("<|im_end|>",add_special_tokens=False)
    panels = {(x["row"]["id"],d,int(m)):x for m,ds in config["landmarks"].items()
              for d,rows in enumerate(ds[:len(paths)]) for x in rows}
    raw,audits,receipts = {},[],[]
    for file in paths:
        for name,digest in load(file)["outputs"]:
            p = Path(name)
            if sha(p)!=digest:
                raise ValueError("source-reader shard hash changed")
            with p.open() as f:
                h,*body,end = map(json.loads,f)
            if (h["cohort_sha256"]!=sha(cohort) or h["phase"]!="eval" or h["patch_sha256"]!=PATCH_SHA
                    or end["kind"]!="complete" or sum(r["kind"]=="edit-check" for r in body)!=1):
                raise ValueError("unexpected source-reader manifest")
            for r in body:
                if r["kind"]=="audit":
                    audits.append(r)
                    continue
                if r["kind"]=="edit-check":
                    continue
                i,d,m,v,s = r["id"],r["draw"],r["landmark"],r["reader"],r["source"]
                key=(i,d,m,v,s)
                x=panels[i,d,m]
                if (r["kind"]!="rollout" or key in raw or i not in h["landmarks"][str(m)] or d!=h["draw"]
                        or r["seed"]!=seed_for(i,d) or r["post_seed"]!=post_seed(i,d)
                        or r["prefix_sha256"]!=packed_hash(x["prefixes"][s]) or r["tokens"][0]!=THINK_END
                        or len(tokens(r))+m>TOTAL or (r["eos_token"] is not None and r["eos_token"] not in h["eos_ids"])):
                    raise ValueError("source-reader pair violates its frozen record")
                raw[key] = r
            receipts.append([name,digest])
    if set(raw)!={(*key,reader,source) for key in panels for reader,source in CELLS}:
        raise ValueError("incomplete crossed source-reader panel")
    if any(a["identical_inputs"]!=a["identical_responses"] for a in audits):
        raise ValueError("recorded identical-source parity failed")
    y={k:score_policy(panels[k[:3]]["row"],r,tokenizer,turn) for k,r in raw.items()}
    reports={}
    for mark in LANDMARKS:
        keys = [k for k in panels if k[2]==mark]
        names = sorted({p for p,_,_ in keys})
        table = {}
        for reader,source in sorted(CELLS):
            pair = [(*k,reader,source) for k in keys]
            table[f"{reader}_reader__{source}_source"] = {
                "n":len(pair), **{q:sum(y[k][q] for k in pair) for q in ("correct_eos","eos","wrong_eos","missing_eos","cap")},
                "correct_box_before_switch":sum(raw[k]["correct_box_in_source"] for k in pair)}
        effects={}
        for metric in ("correct_eos","wrong_eos","eos"):
            effects[metric]={label:paired_interval({
                p:[sum(w*int(y[p,d,mark,r,s][metric]) for (r,s),w in terms.items()) for pp,d,_ in keys if pp==p]
                for p in names}) for label,terms in EFFECTS.items()}
        reports[str(mark)]={"prompts":len(names),"pairs":len(keys),"cells":table,"effects":effects,
                            "excluded": [x for x in config["exclusions"][str(mark)] if x["draw"]<len(paths)],
                            "identical_source_pairs":sum(a["identical_inputs"] for a in audits if a["landmark"]==mark)//2}
    return {"draws":len(paths),"landmarks":reports,"source_shards":receipts,
            "published_source_shards":config["published_raw_receipts"]}


class SourceReaderQueue(Supervisor):
    def __init__(self,args):
        args.role,args.output = "queue",args.root
        super().__init__(args)
        self.worker=Path(__file__).resolve()
        self.env["PYTHONPATH"]=args.runtime_path

    def plan(self,i,phase,d,parts):
        a=self.args
        out=self.lane/f"{phase}-{d}-{i}.jsonl"
        return i,out,[self.worker,"run","--phase",phase,"--root",self.root,"--model",a.model,"--data",a.data,
                      "--output",out,"--artifact",a.artifact,"--publisher-manifest",a.publisher_manifest,
                      "--part",i,"--partitions",parts,"--draw",d,"--deadline",a.deadline]

    def supervise(self):
        try:
            self.status(cohort_sha256=self.args.cohort_sha256)
            (canary,proof),=self.run_batch("canary",[self.plan(0,"smoke",0,1)])
            with canary.open() as f:
                h,*rows,end=map(json.loads,f)
            notes=[r for r in rows if r["kind"]=="audit"]
            if (h["phase"]!="smoke" or end["kind"]!="complete" or len(notes)!=2
                    or any(n["identical_inputs"]!=BLOCK or n["identical_responses"]!=BLOCK for n in notes)):
                raise ValueError("source-reader canary had no exact identical-source control")
            self.status("canary_passed",canary_sha256=proof)
            reserve,completed=1400,0
            for d in range(2):
                if not self.bounded(reserve):
                    break
                start=time.monotonic()
                batch=self.run_batch(f"draw{d}",[self.plan(i,"eval",d,2) for i in range(2)])
                save(self.lane/f"draw{d}-done.json",{"outputs":[[str(p),h] for p,h in batch]})
                completed+=1
                reserve=1.35*(time.monotonic()-start)+120
            if completed:
                out=self.root/"summary.json"
                argv=[sys.executable,"-B",self.worker,"score","--root",self.root,"--model",self.args.model,"--output",out]
                subprocess.run(argv,env={**self.env,"CUDA_VISIBLE_DEVICES":""},check=True,timeout=150)
                self.status("completed" if completed==2 else "budget_complete",summary_sha256=sha(out),completed_eval_draws=completed)
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
        output=summarize(a.root,AutoTokenizer.from_pretrained(a.model,local_files_only=True))
        if a.output.exists():
            raise FileExistsError(a.output)
        save(a.output,output)
        return
    q=SourceReaderQueue(a)
    if socket.gethostname()!=a.host or len(q.devices)!=2 or len(set(q.devices))!=2 or not q.bounded(1700):
        raise ValueError("wrong crossed replay host, GPU assignment, or budget")
    if (sha(a.root/"cohort.json")!=a.cohort_sha256 or sha(a.artifact)!=PATCH_SHA
            or any(sha(q.worker.parent/n)!=h for n,h in load(a.source).items())):
        raise ValueError("crossed replay source changed")
    if a.action=="supervise":
        q.supervise()
        return
    check_devices(q.devices)
    q.lane.mkdir()
    with (q.lane/"supervisor.log").open("x") as log:
        child=subprocess.Popen([sys.executable,"-B",__file__,"supervise",*sys.argv[2:]],env=q.env,start_new_session=True,
                               stdout=log,stderr=subprocess.STDOUT)
    print(json.dumps({"supervisor":child.pid,"root":str(q.root),"deadline":a.deadline,"devices":q.devices,
                      "source_sha256":sha(a.source),"patch_sha256":PATCH_SHA}))


if __name__=="__main__":
    main()
