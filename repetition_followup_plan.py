"""Admit the transfer matrix over frozen public panels on the eight-GPU node."""

import argparse
from pathlib import Path

from repetition_baseline import sha
from repetition_mediation_queue import load, save

MODELS = (
    ("qwen35-08b", "Qwen3.5-0.8B", None),
    ("qwen35-2b", "Qwen3.5-2B", "repetition-model-sizes-20261122-v1/Qwen3.5-2B-checkpoint.json"),
    ("qwen3-17b", "Qwen3-1.7B", "repetition-restart-transfer-20261122-v1/checkpoint.json"),
)


def declaration(root, worker, model, data, checkpoint, dataset, panel, split, start, count, draw, cap,
                deadline, label, dev_dep, preferred, priority):
    name=f"N-{label}-{dataset}-{split}-{start:03d}-d{draw}-b{cap}"
    output=root/"jobs"/(name+".jsonl")
    stable=["--panel",str(panel),"--model",str(model)]
    run=["run",*stable,"--data",str(data),"--dataset",dataset,"--split",split,
         "--start",str(start),"--count",str(count),"--label",dict((s,l) for s,l,_ in MODELS)[label],
         "--cap",str(cap),"--deadline",str(deadline),"--consensus","--output",str(output)]
    if draw:
        run.extend(("--draw",str(draw)))
    if checkpoint:
        run.extend(("--checkpoint",str(checkpoint)))
    eta=min(12000,max(300,90+int(cap*(.095 if label=="qwen35-08b" else .135))))
    job=dict(id=name,lane="N-"+dataset,kind="gpu",worker=str(worker),extension=".jsonl",
             argv=run,deps=[dev_dep],eta=eta,preferred=preferred,priority=priority)
    score=name+"-score"
    score_args=["score",*stable,"--raw",str(output),"--fingerprint","{sha:"+name+"}",
                "--output",str(root/"jobs"/(score+".json"))]
    score_job=dict(id=score,lane=job["lane"],kind="cpu",worker=str(worker),extension=".json",
                   argv=score_args,deps=[name],eta=120,preferred=0,priority=priority+1)
    return job,score_job


def plans(root, baseline, extended, data, *, additional_power=False):
    cfg=load(root/"config.json")
    archive=root.parent
    model_dir=archive.parent.parent/"models"
    # Company paths pinned by the checkpoint receipts and baseline runtime.
    paths=(data.parent/"model",Path("/workspace/storage-shared/models/Qwen3.5-2B"),model_dir/"Qwen3-1.7B")
    plan, reserve = [], []
    for dataset in ("amc_val","svamp"):
        panel=root/("panel-"+dataset+".json")
        meta=load(panel)["panels"][dataset]
        if len(meta["eval"])<32:
            raise ValueError("evaluation panel too small after exact duplicate audit")
        for index,((label,_,checkpoint),model) in enumerate(zip(MODELS,paths,strict=True)):
            checkpoint=archive/checkpoint if checkpoint else None
            base_dep="N-canary-"+label+"-long-score"
            ready_dep=base_dep
            for split in ("dev","eval"):
                size=len(meta[split])
                for start in range(0,size,8):
                    rank=(0 if split=="dev" else 1)+(0 if dataset=="amc_val" else 2)
                    cap=16384 if dataset=="amc_val" else 8192
                    phase=plan if start<128 else reserve
                    p=10+rank+index+(0 if phase is plan else 10)
                    phase.extend(declaration(root, baseline, model, data, checkpoint, dataset, panel, split,
                        start,min(size-start,8),0,cap,cfg["deadline"],label,ready_dep,index,p))
            for draw in (2,4):
                cap=32768 if draw==4 and dataset=="amc_val" and label=="qwen3-17b" else 16384
                size=min(len(meta["eval"]),64 if draw==2 else 32)
                for start in range(0,size,8):
                    reserve.extend(declaration(root,extended,model,data,checkpoint,dataset,panel,"eval",
                        start,min(size-start,8),draw,cap,cfg["deadline"],label,ready_dep,(index+3)%8,33+draw+index))
            if additional_power:
                for draw in (6,8,10,12):
                    if dataset=="svamp" and draw==12:
                        continue
                    if dataset=="amc_val":
                        cap=32768 if draw in (8,12) and label=="qwen3-17b" else 16384
                        size=len(meta["eval"])
                    else:
                        cap=8192 if draw<10 else 16384
                        size=min(len(meta["eval"]),192 if draw<10 else 96)
                    for start in range(0,size,8):
                        reserve.extend(declaration(root,extended,model,data,checkpoint,dataset,panel,"eval",
                            start,min(size-start,8),draw,cap,cfg["deadline"],label,ready_dep,index,
                            64+draw+index))
    if additional_power:
        return [],[job for job in reserve if job["priority"]>=70]
    return plan,reserve


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ("root","baseline","extended","data"):
        cli.add_argument("--"+name,type=Path,required=True)
    cli.add_argument("--additional-power",action="store_true")
    args=cli.parse_args()
    primary,reserve=plans(args.root,args.baseline,args.extended,args.data,additional_power=args.additional_power)
    batches=(("plan-04-additional-power",reserve),) if args.additional_power else (
        ("plan-01-transfer",primary),("plan-02-power",reserve))
    for name,jobs in batches:
        dest=args.root/"plans"/(name+".json")
        if dest.exists():
            raise FileExistsError(dest)
        paths={Path(j["worker"]) for j in jobs}
        save(dest,{"pins":{str(p):sha(p) for p in paths},"jobs":jobs})
        print(name,len(jobs),round(sum(j["eta"] for j in jobs if j["kind"]=="gpu")/3600,2),flush=True)


if __name__=="__main__":
    main()
