"""Freeze an independent late-replay panel from verified long-horizon rollouts."""
import argparse
import json
from pathlib import Path

from repetition_baseline import sha
from repetition_cache_factorial import BOTH, CONV, NATIVE
from repetition_late_budget import DRAWS, DIRECT
from repetition_late_replay import CONFIRM, cycle_landmarks
from repetition_mediation_queue import load, save
from repetition_phase_trace import first_long_cycle
from repetition_state_data import keyed

CAPS = (512,1024,2048)
ARMS = (NATIVE,CONV,BOTH,DIRECT)
TARGET = "Qwen3.5-0.8B"


def freeze(source,output):
    summary,cohort = source/"summary.json",source/"cohort.json"
    status = load(source/"queue/status.json")
    if (output.exists() or status["stage"] not in ("completed","partial") or
            sha(summary)!=status["summary_sha256"]):
        raise ValueError("independent late replay needs a verified completed horizon shard")
    prior,plan = load(summary),load(cohort)
    labels = {x["row"]["id"]:x["row"] for x in plan["eval"]}
    if (prior["cohort_sha256"]!=sha(cohort) or len(labels)!=16 or
            len(plan["excluded_ids"])!=80 or set(labels).intersection(plan["excluded_ids"])):
        raise ValueError("independent horizon cohort is inconsistent")
    if TARGET not in prior["models"]:
        return {"stage":"skip","reason":"target_model_horizon_shard_unavailable","eligible":0}
    manifest = prior["models"][TARGET]["manifest"]
    [path,fingerprint], = [x for x in prior["verified_shards"]
                           if Path(x[0]).stem.endswith("-"+str(manifest["part"]))]
    path = Path(path)
    if sha(path)!=fingerprint:
        raise ValueError("target horizon raw shard checksum changed")
    with path.open() as stream:
        header,*body,end = map(json.loads,stream)
    if (header!=manifest or end["kind"]!="complete" or header["phase"]!="eval"
            or header["cohort_sha256"]!=sha(cohort) or header["native_cap"]!=prior["selected_cap"]):
        raise ValueError("target horizon raw receipt changed")
    marker=header["think_end_id"]
    chosen,seen = [],set()
    for r in body:
        if r["arm"]!="native":
            continue
        p = r["id"]
        if p not in labels or p in seen or r["draw"]!=0:
            raise ValueError("duplicate or invalid horizon prompt")
        seen.add(p)
        ids = r["tokens"]
        think = ids[:ids.index(marker)] if marker in ids else ids
        confirm = first_long_cycle(think)
        if confirm is None:
            continue
        chosen.append({"row":labels[p],"state":CONFIRM,"cycle":cycle_landmarks(think,confirm),
                       "cut":confirm,"completion_prefix":think[:confirm],"render_sha256":r["render_sha256"],
                       "source_draw":0,"source_shard_sha256":fingerprint})
    if seen!=labels.keys():
        raise ValueError("incomplete native horizon source for independent replication")
    if len(chosen)<4:
        return {"stage":"skip","reason":"fewer_than_four_independent_exact_cycles","eligible":len(chosen),
                "native_cap":prior["selected_cap"],"source_summary_sha256":sha(summary),
                "source_shard_sha256":fingerprint}
    fixed = sorted(chosen,key=lambda x:keyed(x["row"]["id"],"late-replication-exact-v1"))[:8]
    save(output,{"cases":fixed,"draws":DRAWS,"caps":CAPS,"arms":ARMS,"states":[CONFIRM],
                 "probe_motif":False,"replication":True,"native_horizon_cap":prior["selected_cap"],
                 "eligible_pool":len(chosen),"source_cohort_sha256":sha(cohort),"source_summary_sha256":sha(summary),
                 "source_shards":[[str(path),fingerprint]],"scope":"new exact cycles from the disjoint Qwen3.5 horizon panel",
                 "primary":{"budget":1024,"contrast":"conv_only-native","outcome":"correct_eos"},
                 "secondary_policy":"conv_only for 1024, then no_think for the remaining 1024 if unfinished"})
    return {"stage":"frozen","eligible":len(chosen),"cases":len(fixed),"cohort_sha256":sha(output),
            "source_summary_sha256":sha(summary),"source_shard_sha256":fingerprint}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(freeze(a.source,a.output)))
