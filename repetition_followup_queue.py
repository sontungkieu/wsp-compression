"""Bound a WSP campaign with append-only, digest-checked job manifests."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

from repetition_baseline import sha
from repetition_mediation_queue import CONTROL_KEYS, load, save, smi

POLL = 8
TERMINAL = {"completed","failed","blocked","skipped_deadline"}
PLACEHOLDER = re.compile(r"^\{sha:([A-Za-z0-9_-]+)\}$")


class Campaign:
    def __init__(self,root,config):
        self.root=root
        self.cfg=config
        self.output=root/"jobs"
        self.jobs={}
        self.mark={}
        self.manifests={}
        self.live={}
        self.status="waiting"
        self.env={k:v for k,v in os.environ.items() if k.lower() not in CONTROL_KEYS}
        self.env.update(PYTHONPATH=config["runtime"], CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1",
                        TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false",OMP_NUM_THREADS="2")

    def inventory(self):
        cards=smi("--query-gpu=index,uuid,memory.used,utilization.gpu")
        processes=smi("--query-compute-apps=gpu_uuid,pid,used_memory")
        roster = [row[1] for row in cards]
        assigned = set(self.cfg["devices"])
        selected = [row for row in cards if row[1] in assigned]
        if (socket.gethostname() != self.cfg["host"]
                or roster != self.cfg.get("host_devices", self.cfg["devices"])
                or not assigned or [row[1] for row in selected] != self.cfg["devices"]):
            raise ValueError("campaign GPU or host identity changed")
        return selected, [row for row in processes if row[0] in assigned]

    def manifests_now(self):
        for file in sorted((self.root/"plans").glob("plan-*.json")):
            h=sha(file)
            if file.name in self.manifests:
                if self.manifests[file.name]!=h:
                    raise ValueError("admitted job manifest changed")
                continue
            plan=load(file)
            for path,fingerprint in plan["pins"].items():
                self.verify_source(Path(path),fingerprint)
            additions={}
            for job in plan["jobs"]:
                name=job["id"]
                worker=job["worker"]
                if (name in self.jobs or name in additions or not re.fullmatch(r"[A-Za-z0-9_-]+",name)
                        or worker not in plan["pins"] or job["kind"] not in ("gpu","cpu")
                        or not 10<=job["eta"]<=14400 or not job["argv"]):
                    raise ValueError(f"invalid job declaration {name}")
                additions[name]=job|{"sha256":plan["pins"][worker]}
            names=self.jobs.keys()|additions.keys()
            if any(name in j["deps"] or not set(j["deps"])<=names for name,j in additions.items()):
                raise ValueError("job has an undeclared dependency")
            for name,job in additions.items():
                self.jobs[name]=job
                self.mark[name]={"stage":"queued","lane":job["lane"],"kind":job["kind"],
                                 "eta":job["eta"],"worker":job["worker"],"worker_sha256":job["sha256"]}
            self.manifests[file.name]=h

    def verify_source(self,path,fingerprint):
        if not path.is_absolute() or not path.is_relative_to(self.cfg["source_root"]) or sha(path)!=fingerprint:
            raise ValueError(f"unsafe or changed campaign source {path}")

    def persist(self):
        save(self.root/"status.json",{"host":self.cfg["host"],"deadline":self.cfg["deadline"],
              "access_deadline":self.cfg["access"],"state":self.status,"updated":time.time(),
              "manifest_shas":self.manifests,"jobs":self.mark})

    def argv(self,job):
        args=[]
        for item in job["argv"]:
            match=PLACEHOLDER.fullmatch(item)
            if match:
                ident=match[1]
                if ident not in job["deps"] or self.mark[ident]["stage"]!="completed":
                    raise ValueError("requested digest of an incomplete dependency")
                item=self.mark[ident]["sha256"]
            args.append(item)
        return args

    def launch(self,name,device):
        job=self.jobs[name]
        worker=Path(job["worker"])
        self.verify_source(worker,job["sha256"])
        args=self.argv(job)
        path=self.output/(name+job.get("extension",".jsonl"))
        if str(path) not in args or path.exists():
            raise ValueError("job must declare its fresh result path")
        log=(self.output/(name+".log")).open("x")
        try:
            proc=subprocess.Popen([sys.executable,"-B",worker,*args],
                  env={**self.env,"CUDA_VISIBLE_DEVICES":device or ""},stdout=log,stderr=subprocess.STDOUT,
                  start_new_session=True)
        finally:
            log.close()
        self.live[name]=(proc,device,path)
        self.mark[name].update(stage="running",pid=proc.pid,device=device,output=str(path),started=time.time())

    def finalize(self,name,why=None):
        proc,device,path=self.live.pop(name)
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)
        result=self.mark[name]
        complete=proc.returncode==0 and path.is_file() and why is None
        result.update(stage="completed" if complete else "failed",exit_code=proc.returncode,
                      sha256=sha(path) if complete else None,ended=time.time(),reason=why)
        result["wall_seconds"]=round(result["ended"]-result["started"],3)
        save(self.output/(name+".receipt.json"),result)

    def schedule(self,cards,processes):
        for name,(proc,device,_) in list(self.live.items()):
            foreign=device is not None and any(u==device and int(pid)!=proc.pid for u,pid,_ in processes)
            if proc.poll() is not None or foreign:
                self.finalize(name,"additional_compute_process" if foreign and proc.poll() is None else None)
        for name,job in self.jobs.items():
            current=self.mark[name]
            if current["stage"]!="queued":
                continue
            if any(self.mark[d]["stage"] in TERMINAL-{"completed"} for d in job["deps"]):
                current.update(stage="blocked",reason="required job did not complete")
            elif time.time()+max(1200,job["eta"]+900)>=self.cfg["deadline"]:
                current.update(stage="skipped_deadline",reason="arrival would exceed the reserved end buffer")
        taken={d for _,d,_ in self.live.values() if d}
        occupied={u for u,_,_ in processes}
        free=[u for _,u,used,util in cards if u not in taken|occupied and int(used)<=5 and util=="0"]
        candidates=sorted((name for name,j in self.jobs.items() if self.mark[name]["stage"]=="queued"
                           and all(self.mark[d]["stage"]=="completed" for d in j["deps"])),
                          key=lambda name:(self.jobs[name].get("priority",100),name))
        cpu_active=sum(d is None for _,d,_ in self.live.values())
        for name in candidates:
            job=self.jobs[name]
            if job["kind"]=="gpu" and free:
                preferred=self.cfg["devices"][job.get("preferred",0)]
                self.launch(name,free.pop(free.index(preferred) if preferred in free else 0))
            elif job["kind"]=="cpu" and cpu_active<2:
                self.launch(name,None)
                cpu_active+=1

    def finish(self):
        self.persist()
        cards,processes=self.inventory()
        own={str(j["pid"]) for j in self.mark.values() if "pid" in j and j["kind"]=="gpu"}
        run_processes=[]
        for row in processes:
            _,pid,_=row
            if pid in own:
                try:
                    if os.fsencode(self.root) in (Path("/proc")/pid/"cmdline").read_bytes():
                        run_processes.append(row)
                except FileNotFoundError:
                    pass
        by_lane={}
        for j in self.mark.values():
            group=by_lane.setdefault(j["lane"],{"stages":{},"gpu_hours":0.,"cpu_hours":0.})
            stage=j["stage"]
            group["stages"][stage]=group["stages"].get(stage,0)+1
            key="gpu_hours" if j["kind"]=="gpu" else "cpu_hours"
            group[key]+=j.get("wall_seconds",0.)/3600
        stamp=datetime.now(timezone(timedelta(hours=7))).isoformat(timespec="seconds")
        save(self.root/"release.json",{"state":self.status,"observed_ict":stamp,"host":self.cfg["host"],
                    "allocation":cards,"compute":processes,"campaign_compute":run_processes,
                    "cutoff_epoch":self.cfg["deadline"],"status_sha256":sha(self.root/"status.json"),
                    "lane_accounting":by_lane})

    def run(self):
        try:
            while time.time()<self.cfg["deadline"]-25:
                self.manifests_now()
                cards,processes=self.inventory()
                self.schedule(cards,processes)
                sealed=(self.root/"sealed.json").is_file() or time.time()>=self.cfg["access"]
                self.status="active" if self.live else "waiting"
                self.persist()
                if self.jobs and not self.live and sealed and all(j["stage"] in TERMINAL for j in self.mark.values()):
                    self.status="completed" if all(j["stage"]=="completed" for j in self.mark.values()) else "partial"
                    break
                if sealed and not self.live and self.jobs and not any(j["stage"]=="queued" for j in self.mark.values()):
                    self.status="partial"
                    break
                time.sleep(POLL)
            else:
                self.status="deadline"
        except Exception:
            self.status="failed"
            raise
        finally:
            for name in list(self.live):
                self.finalize(name,self.status)
            if self.status=="deadline":
                for j in self.mark.values():
                    if j["stage"]=="queued":
                        j.update(stage="skipped_deadline",reason="compute allocation ended")
            self.finish()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("launch","run"))
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--config-sha256",required=True)
    a=p.parse_args()
    config=a.root/"config.json"
    if sha(config)!=a.config_sha256:
        raise ValueError("campaign configuration digest differs")
    cfg=load(config)
    q=Campaign(a.root,cfg)
    q.inventory()
    if a.action=="run":
        q.run()
        return
    if sha(Path(__file__))!=cfg["queue_sha256"] or time.time()>=cfg["access"] or cfg["deadline"]<=cfg["access"]:
        raise ValueError("source or launch window differs")
    if (a.root/"launch.json").exists():
        raise FileExistsError("campaign already submitted")
    q.output.mkdir(exist_ok=True)
    (a.root/"plans").mkdir(exist_ok=True)
    with (a.root/"supervisor.log").open("x") as log:
        proc=subprocess.Popen([sys.executable,"-B",__file__,"run",*sys.argv[2:]],env=q.env,
                              start_new_session=True,stdout=log,stderr=subprocess.STDOUT)
    receipt={"supervisor":proc.pid,"host":cfg["host"],"devices":cfg["devices"],"deadline":cfg["deadline"],
             "access":cfg["access"],"config_sha256":a.config_sha256}
    save(a.root/"launch.json",receipt)
    print(json.dumps(receipt))


if __name__=="__main__":
    main()
