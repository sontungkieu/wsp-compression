"""Test answer-budget headroom on a frozen subset of the restart evaluation."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

from repetition_baseline import sha
from repetition_budget_run import packed_hash, tokens
from repetition_budget_score import cut
from repetition_canonical import paired_interval, seed_for
from repetition_mediation import SAMPLING
from repetition_mediation_queue import Supervisor, check_devices, load, save
from repetition_restart import BATCH, output_row, prepare_items, score_policy
from repetition_sampler_reach import pack_at
from repetition_state_data import keyed
from repetition_state_value import Runtime, make_timing_policy, post_seed

CARD_SAMPLING = {"temperature":1.0, "top_p":1.0, "top_k":20, "presence_penalty":2.0}
CARD_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
BUDGET = 4096
CARD = "no_think_model_card"
DIRECT = ("no_think","question_e0")


def freeze(source, output):
    cases = sorted(load(source)["eval"], key=lambda c: keyed(c["row"]["id"],"restart-cap-4096"))[:32]
    if len({c["row"]["id"] for c in cases}) != 32 or output.exists():
        raise ValueError("require a fresh cohort of 32 distinct MATH-500 IDs")
    save(output, {"cases":cases, "source_cohort_sha256":sha(source), "budget":BUDGET,
                  "scope":"cap sensitivity on a hash-ranked subset of the earlier evaluation"})


def select_baseline(parent, cohort, output):
    status = load(parent/"queue"/"status.json")
    summary_path = parent/"eval-summary.json"
    if (status["stage"] not in ("completed","budget_complete") or not status.get("completed_eval_draws")
            or sha(parent/"cohort.json") != load(cohort)["source_cohort_sha256"]
            or sha(summary_path) != status["eval_summary_sha256"] or output.exists()):
        raise ValueError("parent restart evaluation is not verified")
    summary = load(summary_path)
    totals = summary["strata"]["all"]["arms"]
    winner = max(DIRECT, key=lambda arm: totals[arm]["correct_eos"])
    arms = ["native","no_think",CARD]
    if winner == "question_e0":
        arms.insert(2,winner)
    save(output, {"primary_direct":winner, "arms":arms, "parent_summary_sha256":sha(summary_path),
                  "shared_sampling":SAMPLING,"model_card_sampling":CARD_SAMPLING,"model_card_revision":CARD_REVISION})


def run(a):
    cohort_path, settings_path = a.root/"cohort.json", a.root/"settings.json"
    cohort, settings = load(cohort_path), load(settings_path)
    arms = settings["arms"]
    budget = 128 if a.action == "smoke" else cohort["budget"]
    cases = cohort["cases"][:4] if a.action == "smoke" else cohort["cases"][a.part::a.partitions]
    r = Runtime(a)
    official,_ = make_timing_policy(r.torch,r.transformers,r.groups,CARD_SAMPLING)
    items = prepare_items(r,cases)
    with a.output.open("x") as f, r.torch.inference_mode():
        def emit(x):
            print(json.dumps(x,allow_nan=False),file=f,flush=True)
        emit({"kind":"manifest","arms":arms,"ids":[x["row"]["id"] for x in items],"draw":a.draw,
              "cohort_sha256":sha(cohort_path),"settings_sha256":sha(settings_path),"budget":budget,
              "worker_sha256_extension":sha(Path(__file__)),**r.manifest})
        for start in range(0,len(items),BATCH):
            if time.time() >= a.deadline-120:
                raise TimeoutError("extension reached the hard deadline")
            group = items[start:start+BATCH]
            width = max(max(len(x["native_ids"]),len(x["no_think_ids"])) for x in group)
            for name in arms:
                arm = "e0" if name == "question_e0" else "no_think" if name == CARD else name
                field = "no_think_ids" if arm == "no_think" else "native_ids"
                ids = [x[field] for x in group]
                packed = pack_at(r,ids,width)
                gen = r.generate(group,arm,a.draw,budget-int(arm=="e0"),packed=packed,
                                 policy=official if name == CARD else None)
                for raw,original in zip(gen,ids,strict=True):
                    row = output_row(raw,name,budget)
                    if name == CARD:
                        row["score_as_arm"] = "no_think"
                    emit(row | {"input_sha256":packed_hash(original)})
        emit({"kind":"complete"})


def score(root, tokenizer):
    cohort, settings = load(root/"cohort.json"),load(root/"settings.json")
    prompts = {c["row"]["id"]:c["row"] for c in cohort["cases"]}
    records, receipts = {}, []
    [turn] = tokenizer.encode("<|im_end|>",add_special_tokens=False)
    for receipt in sorted((root/"queue").glob("draw*-done.json")):
        for name, fingerprint in load(receipt)["outputs"]:
            path = Path(name)
            if sha(path) != fingerprint:
                raise ValueError("extension worker output hash mismatch")
            with path.open() as f:
                header,*rows,footer = map(json.loads,f)
            if (header["cohort_sha256"] != sha(root/"cohort.json") or header["settings_sha256"] != sha(root/"settings.json")
                    or header["budget"] != BUDGET or footer["kind"] != "complete"):
                raise ValueError("extension manifest differs from the fixed contract")
            for row in rows:
                p,d,arm = row["id"],row["draw"],row["arm"]
                if (row["kind"] != "rollout" or p not in header["ids"] or d != header["draw"] or (p,d,arm) in records
                        or row["seed"] != seed_for(p,d) or row["post_seed"] != post_seed(p,d) or len(tokens(row))>BUDGET
                        or (row["eos_token"] is not None and row["eos_token"] not in header["eos_ids"])):
                    raise ValueError("extension rollout violated its cohort, seed or budget")
                records[p,d,arm] = row
            receipts.append([name,fingerprint])
    ds = {d for _,d,_ in records}
    arms = settings["arms"]
    if not ds or set(records) != {(p,d,a) for p in prompts for d in ds for a in arms}:
        raise ValueError("incomplete extension matrix")
    windows, scored = {}, {}
    for budget in (2048,BUDGET):
        y = {}
        for key,row in records.items():
            ids,eos = cut(row,"total",budget)
            sample = row | {"tokens":ids, "eos_token":row["eos_token"] if eos else None,
                            "arm":row.get("score_as_arm",row["arm"])}
            y[key] = score_policy(prompts[key[0]],sample,tokenizer,turn)
        scored[budget] = y
        table = {}
        for arm in arms:
            items = [y[p,d,arm] for p in prompts for d in ds]
            tab = {metric:sum(x[metric] for x in items) for metric in items[0] if metric != "switched"}
            table[arm] = {"n":len(items), **tab}
        controls = ((settings["primary_direct"],"native"),(CARD,"no_think"))
        effects = {f"{a}_minus_{b}":paired_interval({p:[int(y[p,d,a]["correct_eos"])-int(y[p,d,b]["correct_eos"])
                                                        for d in ds] for p in prompts}) for a,b in controls}
        windows[str(budget)] = {"arms":table,"contrasts":effects}
    gains = {a:paired_interval({
        p:[int(scored[BUDGET][p,d,a]["correct_eos"])-int(scored[2048][p,d,a]["correct_eos"]) for d in ds]
        for p in prompts}) for a in arms}
    return {"n_prompts":len(prompts),"draws":len(ds),"settings":settings,"verified_shards":receipts,
            "windows":windows,"extra_budget_effect":gains}


class ExtensionQueue(Supervisor):
    def __init__(self,args):
        args.role,args.output = "queue",args.root
        super().__init__(args)
        self.worker = Path(__file__).resolve()
        self.env["PYTHONPATH"] = args.runtime_path

    def plan(self,i,action,draw,partitions):
        out = self.lane/f"{action}-{draw}-{i}.jsonl"
        return i,out,[self.worker,action,"--root",self.root,"--model",self.args.model,"--data",self.args.data,
                      "--output",out,"--part",i,"--partitions",partitions,"--draw",draw,"--deadline",self.args.deadline]

    def supervise(self):
        try:
            self.status(cohort_sha256=sha(self.root/"cohort.json"))
            (smoke,fingerprint), = self.run_batch("canary",[self.plan(0,"smoke",0,1)])
            with smoke.open() as f:
                h,*rows,end = map(json.loads,f)
            if h["budget"] != 128 or end["kind"] != "complete" or {(r["id"],r["arm"]) for r in rows} != {
                (p,a) for p in h["ids"] for a in load(self.root/"settings.json")["arms"]}:
                raise ValueError("extension canary incomplete")
            self.status("canary_passed",canary_sha256=fingerprint)
            if not self.bounded(1200):
                self.status("budget_complete")
                return
            results = self.run_batch("draw0", [self.plan(i,"run",0,len(self.devices)) for i in range(len(self.devices))])
            save(self.lane/"draw0-done.json",{"outputs":[[str(p),fp] for p,fp in results]})
            summary = self.root/"summary.json"
            argv = [sys.executable,"-B",self.worker,"score","--root",self.root,"--model",self.args.model,"--output",summary]
            subprocess.run(argv,env={**self.env,"CUDA_VISIBLE_DEVICES":""},check=True,timeout=100)
            self.status("completed",summary_sha256=sha(summary))
        except Exception as exc:
            self.status("failed",error=str(exc))
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("launch","supervise","run","smoke","score"))
    for name in ("root","model","data","output","overlay","stable","source"):
        p.add_argument("--"+name,type=Path)
    for name in ("runtime-path","devices","host","cohort-sha256","settings-sha256"):
        p.add_argument("--"+name)
    for name,default in (("part",0),("partitions",1),("draw",0),("deadline",None)):
        p.add_argument("--"+name,type=int,default=default)
    a = p.parse_args()
    if a.action in ("run","smoke"):
        run(a)
        return
    if a.action == "score":
        from transformers import AutoTokenizer
        summary = score(a.root, AutoTokenizer.from_pretrained(a.model,local_files_only=True))
        if a.output.exists():
            raise FileExistsError(a.output)
        save(a.output,summary)
        return
    queue = ExtensionQueue(a)
    if socket.gethostname() != a.host or len(set(queue.devices)) != 2 or not queue.bounded(1320):
        raise ValueError("incorrect extension host, assigned devices or remaining budget")
    if (sha(a.root/"cohort.json") != a.cohort_sha256 or sha(a.root/"settings.json") != a.settings_sha256
            or any(sha(queue.worker.parent/n) != h for n,h in load(a.source).items())):
        raise ValueError("unexpected extension cohort, settings or sources")
    if a.action == "supervise":
        queue.supervise()
        return
    check_devices(queue.devices)
    queue.lane.mkdir()
    with (queue.lane/"supervisor.log").open("x") as log:
        job = subprocess.Popen([sys.executable,"-B",__file__,"supervise",*sys.argv[2:]],
                               env=queue.env,start_new_session=True,stdout=log,stderr=subprocess.STDOUT)
    print(json.dumps({"supervisor":job.pid,"root":str(a.root),"deadline":a.deadline,"devices":queue.devices,
                      "source_sha256":sha(a.source)}))


if __name__ == "__main__":
    main()
