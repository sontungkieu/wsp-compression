"""Discover disjoint scratchpads and run the frozen two-device factorial."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

from repetition_mediation_queue import Supervisor, check_devices, digest, load, save
from repetition_scratchpad import ARMS
from repetition_scratchpad_cohort import choose_panel
from repetition_state_queue import HOST

DISCOVERY_WAVES = 4
DRAWS = 4


class ScratchpadQueue(Supervisor):
    def __init__(self, args):
        args.role, args.output = "queue", args.root
        super().__init__(args)
        self.source = Path(__file__).resolve().parent
        self.env["PYTHONPATH"] = args.runtime_path
        self.worker = self.source / "repetition_scratchpad.py"

    def discover(self):
        plan = load(self.root/"discovery-plan.json")
        if plan["exposure_sha256"]!=digest(self.root/"exposure.json"):
            raise ValueError("discovery plan does not match the exposure audit")
        available = list(range(len(plan["panels"])))
        reports = []
        picked = choose_panel(self.args.baseline, self.root, self.args.data, reports)
        for i in range(DISCOVERY_WAVES):
            if picked or not self.bounded(2700):
                break
            shards = available[2*i:2*i+2]
            if not shards:
                break
            plans = []
            for lane, shard in enumerate(shards):
                out = self.lane / f"discovery-{shard:02d}.jsonl"
                worker = self.source / "repetition_baseline.py"
                argv = [worker,"--model",self.args.model,"--data",self.args.data,
                        "--output",out,"--deadline",self.args.deadline,"--mode","think",
                        "--decoder","astra","--astra-math","--shard",shard,
                        "--count",plan["panels"][shard]["n"],"--panel-manifest",plan["panels"][shard]["path"],
                        "--batch",16,"--cap",8192]
                plans.append((lane,out,argv))
            paths = self.run_batch(f"discover{i}",plans)
            reports.extend((str(path), fingerprint) for path,fingerprint in paths)
            self.status("checking_discovery", discovery_shards=len(reports))
            picked = choose_panel(self.args.baseline,self.root,self.args.data,reports)
        if picked is None:
            picked = choose_panel(self.args.baseline,self.root,self.args.data,reports,allow_short=True)
        return picked

    def supervise(self):
        try:
            exposure = self.root/"exposure.json"
            self.status("admitted", exposure_sha256=digest(exposure) if exposure.exists() else None,completed_draws=[])
            panel = load(self.root/"cohort.json") if self.args.cohort_sha256 else self.discover()
            self.status("cohort_frozen",cohort_sha256=digest(self.root/"cohort.json"),
                        cases=len(panel["cases"]),strata=panel["counts"],discovery_shards=len(panel.get("discovery_sources",[])))
            for draw in range(DRAWS):
                prior = self.state.get("last_draw_seconds")
                if not self.bounded(300 if prior is None else 1.15*prior+100):
                    self.status("budget_complete")
                    return
                name = f"draw{draw}"
                plans = []
                for lane in range(2):
                    out = self.lane / f"{name}-{lane:02d}.jsonl"
                    argv = [self.worker,"run","--root",self.root,"--model",self.args.model,
                            "--data",self.args.data,"--output",out,"--deadline",self.args.deadline,
                            "--part",lane,"--partitions",2,"--draw",draw]
                    plans.append((lane,out,argv))
                start = time.monotonic()
                records = self.run_batch(name,plans)
                save(self.lane / f"{name}-done.json",
                     {"outputs":[[str(p),fingerprint] for p,fingerprint in records]})
                self.status(name,completed_draws=list(range(draw+1)),last_draw_seconds=time.monotonic()-start)
            self.status("completed")
        except Exception as exc:
            self.status("failed",error=str(exc))
            raise


def verify_canary(canary, worker, arms, cap):
    with canary.open() as stream:
        h,*rows,end = map(json.loads,stream)
    audit = [x for x in rows if x["kind"]=="audit"]
    rollout = [x for x in rows if x["kind"]=="rollout"]
    if (h["factorial_sha256"]!=digest(worker) or tuple(h["arms"])!=arms or h["cap"]!=cap
            or len(rollout)!=len(h["prompt_ids"])*len(arms)
            or end["kind"]!="complete" or len(audit)!=1 or not audit[0]["empty_pair_exact"]
            or not audit[0]["gpu_canary"]):
        raise ValueError("scratchpad factorial canary has not passed")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("launch","supervise"))
    for name in ("root","model","data","overlay","stable","source","canary-report"):
        p.add_argument("--"+name,type=Path,required=True)
    for name in ("runtime-path","devices"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--baseline",type=Path)
    p.add_argument("--exposure-sha256")
    p.add_argument("--plan-sha256")
    p.add_argument("--cohort-sha256")
    p.add_argument("--deadline",type=int,required=True)
    a = p.parse_args()
    q = ScratchpadQueue(a)
    if socket.gethostname()!=HOST or len(set(q.devices))!=2 or a.deadline-time.time()<=420:
        raise ValueError("incorrect hostname, assigned devices, or expiry")
    hashes = load(a.source)
    if any(digest(q.source/n)!=fingerprint for n,fingerprint in hashes.items()):
        raise ValueError("source SHA mismatch")
    if a.cohort_sha256:
        if digest(q.root/"cohort.json")!=a.cohort_sha256:
            raise ValueError("fixed cohort changed")
        protocol = load(q.root/"cohort.json")
    else:
        if not (a.baseline and a.exposure_sha256 and a.plan_sha256):
            raise ValueError("discovery requires a baseline and initial plan")
        if digest(q.root/"exposure.json")!=a.exposure_sha256 or digest(q.root/"discovery-plan.json")!=a.plan_sha256:
            raise ValueError("discovery audit changed")
        protocol = {}
    if a.action=="supervise":
        q.supervise()
        return
    check_devices(q.devices)
    verify_canary(a.canary_report,q.worker,tuple(protocol.get("arms",ARMS)),protocol.get("cap",512))
    q.lane.mkdir()
    with (q.lane/"supervisor.log").open("x") as log:
        child = subprocess.Popen([sys.executable,"-B",__file__,"supervise",*sys.argv[2:]],
                                 env=q.env,start_new_session=True,stdout=log,stderr=subprocess.STDOUT)
    print(json.dumps({"supervisor":child.pid,"root":str(q.root),"deadline":a.deadline,"devices":q.devices,
                      "source_sha256":digest(a.source),"exposure_sha256":a.exposure_sha256}))


if __name__=="__main__":
    main()
