"""Score answer availability and a matched restart policy from confirmed rollout artifacts."""
import argparse
from decimal import Decimal
import json
from pathlib import Path
import socket
import statistics
import time

import repetition_late_budget as replay
from repetition_baseline import sha
from repetition_budget_score import cut
from repetition_late_confirmation import ARMS, BASE_CUTS, FULL_DIRECT, INCREMENT, REPLAY_CUTS, TOTAL
from repetition_late_confirmation import baseline_read, paired, score_cut, summarize_rows
from repetition_mediation_queue import load, save
from repetition_native_horizon import first_correct_box


def details(row,raw,cap,tokenizer,marker,turn,spent=0):
    scored = score_cut(row,raw,cap,tokenizer,turn,spent)
    ids,_ = cut(raw,"total",cap)
    if raw["arm"]==FULL_DIRECT:
        start = 0
    else:
        start = ids.index(marker)+1 if marker in ids else len(ids)
    answer = ids[start:]
    if turn in answer:
        answer = answer[:answer.index(turn)]
    found = first_correct_box(answer,tokenizer,Decimal(row["gold"]))
    return scored|{"numeric_correct_without_eos":scored["numeric_correct"] and not scored["eos"],
                   "first_correct_box_token":spent+start+found if found is not None else None}


def aggregate(rows):
    boxed = sorted(r["first_correct_box_token"] for r in rows if r["first_correct_box_token"] is not None)
    return summarize_rows(rows)|{"numeric_correct_without_eos":sum(r["numeric_correct_without_eos"] for r in rows),
             "ever_correct_box":len(boxed),"first_correct_box_median_token":statistics.median(boxed) if boxed else None}


def audit(root,model,status):
    from transformers import AutoTokenizer
    summary_path = root/"summary.json"
    if sha(summary_path)!=status["summary_sha256"]:
        raise ValueError("confirmation summary checksum changed")
    summary = load(summary_path)
    panel_path = root/"panel.json"
    if sha(panel_path)!=status["panel_sha256"]:
        raise ValueError("confirmation panel checksum changed")
    panel = load(panel_path)
    rows = {x["row"]["id"]:x["row"] for x in panel["eval"]}
    baseline = {}
    for file,digest,part in summary["evidence"]["baseline"]:
        _,raw,_ = baseline_read(Path(file),digest,Path(part),TOTAL,panel["direct_cap"])
        if baseline.keys()&raw.keys():
            raise ValueError("duplicate baseline data")
        baseline.update(raw)
    if baseline.keys()!={(p,a) for p in rows for a in ("native",FULL_DIRECT)}:
        raise ValueError("baseline data no longer covers the panel")
    tok = AutoTokenizer.from_pretrained(model,local_files_only=True)
    [marker] = tok.encode("</think>",add_special_tokens=False)
    [turn] = tok.encode("<|im_end|>",add_special_tokens=False)
    base_table = {}
    for cap in BASE_CUTS:
        base_table[str(cap)] = {a:aggregate([details(row,baseline[p,a],cap,tok,marker,turn)
                                  for p,row in rows.items()]) for a in ("native",FULL_DIRECT)
                                  if a=="native" or cap<=panel["direct_cap"]}
    cohorts,replays = {},{}
    if summary["replayed"]:
        cohort_path = root/"cohort.json"
        if sha(cohort_path)!=status["replay_cohort_sha256"]:
            raise ValueError("confirmation replay cohort checksum changed")
        cohorts = {(x["row"]["id"],x["state"]):x for x in load(cohort_path)["cases"]}
        for file,digest in summary["evidence"]["replay"]:
            raw,_,_ = replay.read_shard(Path(file),digest,cohort_path,"eval",cohorts)
            for r in raw:
                key = r["id"],r["draw"],r["arm"]
                if key in replays:
                    raise ValueError("duplicate replay data")
                replays[key] = r
        if replays.keys()!={(p,d,a) for p,_ in cohorts for d in range(replay.DRAWS) for a in ARMS}:
            raise ValueError("replay data no longer covers all declared pairs")
    replay_table = {}
    for cap in REPLAY_CUTS:
        replay_table[str(cap)] = {a:aggregate([details(rows[p],replays[p,d,a],cap,tok,marker,turn)
                           for p,_ in cohorts for d in range(replay.DRAWS)]) for a in ARMS}
    itt = None
    if summary["itt_status"]=="complete_draw0" and len(cohorts)==summary["eligible"]:
        cuts = {p:v["cut"] for (p,_),v in cohorts.items()}
        base = {p:[details(row,baseline[p,"native"],TOTAL,tok,marker,turn)] for p,row in rows.items()}
        policies = {"native":base}
        for label,arm in (("trigger_then_keep",replay.NATIVE),("trigger_then_conv",replay.CONV),
                          ("trigger_then_fresh_direct",replay.DIRECT)):
            policies[label] = {p:([details(row,replays[p,0,arm],INCREMENT,tok,marker,turn,spent=cuts[p])]
                                   if p in cuts else base[p]) for p,row in rows.items()}
        if panel["direct_cap"]==TOTAL:
            policies["direct_from_start"] = {p:[details(row,baseline[p,FULL_DIRECT],TOTAL,tok,marker,turn)]
                                              for p,row in rows.items()}
        conv = policies["trigger_then_conv"]
        fresh = policies["trigger_then_fresh_direct"]
        itt = {"draw":0,"total_cap":TOTAL,"policy":{a:aggregate([v[0] for v in cells.values()])
               for a,cells in policies.items()},"conv_minus_triggered_fresh_direct":paired(conv,fresh),
               "triggered_fresh_direct_minus_native":paired(fresh,base),
               "triggered_fresh_direct_minus_direct_from_start":paired(fresh,policies["direct_from_start"])
               if "direct_from_start" in policies else None}
    return {"primary_summary_sha256":sha(summary_path),"n":len(rows),"eligible":summary["eligible"],
            "replayed":summary["replayed"],"baseline":base_table,"conditional":replay_table,
            "itt":itt,"scope":"supplemental offline replay, first box after the reasoning close"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--host",required=True)
    p.add_argument("--deadline",type=int,required=True)
    a = p.parse_args()
    if socket.gethostname()!=a.host:
        raise ValueError("wrong host for confirmation audit")
    out = a.root/"supplement.json"
    if out.exists():
        raise FileExistsError(out)
    status = a.root/"queue/status.json"
    while time.time()<a.deadline-30:
        if status.is_file():
            s = load(status)
            if s["deadline"]!=a.deadline:
                raise ValueError("confirmation audit deadline mismatch")
            if s["stage"]=="completed":
                save(out,audit(a.root,a.model,s))
                print(json.dumps({"stage":"completed","path":str(out),"sha256":sha(out)}))
                return
            if s["stage"] in {"failed","baseline_only","skipped_budget"}:
                print(json.dumps({"stage":"skipped","main_stage":s["stage"]}))
                return
        time.sleep(20)
    print(json.dumps({"stage":"skipped","reason":"insufficient_cpu_window"}))


if __name__=="__main__":
    main()
