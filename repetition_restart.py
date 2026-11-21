"""Matched-budget policies from the question with an auditable shared trunk."""
import argparse
import json
from pathlib import Path
import time

from repetition_baseline import math_answer, render, sha
from repetition_budget_run import packed_hash, tokens
from repetition_budget_score import cut
from repetition_exit_localize import capture
from repetition_mediation_queue import load
from repetition_sampler_reach import pack_at
from repetition_state_data import THINK_END
from repetition_state_value import Runtime, inputs_for

ARMS = ("native","no_think","question_e0","switch_keep","switch_restart")
BATCH = 8
BASE = "4f9403621008c0f8ffe56c3e5087b0ae95ede75a"


def reaches_switch(row, switch, marker=THINK_END):
    """Only interrupt a still-thinking trajectory after exactly switch tokens."""
    before = tokens(row)[:switch]
    return len(before)==switch and marker not in before and (
        row["eos_token"] is None or len(row["tokens"])>=switch)


def nested_restart(question, budget, switch):
    response, eos = cut(question,"total",budget-switch)
    return {"tokens":response,"eos_token":question["eos_token"] if eos else None}


def output_row(source, arm, budget, *, prefix=(), switched=False, source_arm=None, marker=THINK_END):
    result = {"kind":"rollout","id":source["id"],"draw":source["draw"],"seed":source["seed"],
              "post_seed":source["post_seed"],"arm":arm,"tokens":list(prefix)+source["tokens"],
              "eos_token":source["eos_token"],"switched":switched,"source_arm":source_arm,
              "spent_before_switch":len(prefix)}
    used = len(tokens(result))
    if used>budget or (result["eos_token"] is None and used!=budget):
        raise ValueError("continuation does not respect the fixed total budget")
    if switched and result["tokens"][len(prefix)]!=marker:
        raise ValueError("restart failed to generate the real closing marker")
    if arm=="question_e0" and result["tokens"][0]!=marker:
        raise ValueError("question E0 did not close at step zero")
    return result


def prepare_items(runtime,cases):
    items = inputs_for(runtime.tokenizer,[x | {"completion_prefix":[]} for x in cases])
    for item in items:
        prompt = render(runtime.tokenizer,item["row"],"no-think")
        item["no_think_ids"] = runtime.tokenizer.encode(prompt, add_special_tokens=False)
    return items


def batch_run(runtime, group, draw, budget, switch, diagnostics):
    [marker] = runtime.groups["think"]
    width = max(max(len(x["native_ids"])+switch,len(x["no_think_ids"])) for x in group)
    question_pack = pack_at(runtime,[x["native_ids"] for x in group],width)
    direct_pack = pack_at(runtime,[x["no_think_ids"] for x in group],width)
    seconds = {}

    def generate(purpose, arm, batch, cap, packed):
        start = time.monotonic()
        result = runtime.generate(batch,arm,draw,cap,packed=packed)
        runtime.torch.cuda.synchronize()
        seconds[purpose] = time.monotonic()-start
        return result

    native = generate("native","native",group,budget,question_pack)
    direct = generate("no_think","no_think",group,budget,direct_pack)
    question = generate("question_e0","e0",group,budget-1,question_pack)
    switches = [reaches_switch(r,switch,marker) for r in native]
    prior = [r["tokens"][:switch] if applies else [] for r,applies in zip(native,switches,strict=True)]
    retained = [x | {"native_ids":x["native_ids"]+s,"presence_seed_ids":[]}
                for x,s in zip(group,prior,strict=True)]
    keep_pack = pack_at(runtime,[x["native_ids"] for x in retained],width)
    keep = generate("switch_keep_suffix","e0",retained,budget-switch-1,keep_pack) if any(switches) else question
    # The pilot tests exact nested equivalence and measures real re-prefill.
    restart = generate("explicit_restart_suffix","e0",group,budget-switch-1,question_pack) if diagnostics else None
    probe = {}
    if diagnostics:
        short = generate("pilot_trunk","native",group,switch,question_pack)
        probe["short_trunk_exact"] = [tokens(a)==tokens(b)[:switch] for a,b in zip(short,native,strict=True)]
        probe["prefill_seconds"] = {}
        for name,pack in (("question",question_pack),("retained",keep_pack),("direct",direct_pack)):
            runtime.torch.cuda.synchronize()
            start = time.monotonic()
            capture(runtime.model,pack,{},use_cache=True)
            runtime.torch.cuda.synchronize()
            probe["prefill_seconds"][name] = time.monotonic()-start
    rows = []
    for i,item in enumerate(group):
        rows.extend(output_row(r,arm,budget,marker=marker) for r,arm in
                    ((native[i],"native"),(direct[i],"no_think"),(question[i],"question_e0")))
        if switches[i]:
            nested = question[i] | nested_restart(question[i],budget,switch)
            resumed = restart[i] if restart is not None else nested
            matches = tokens(resumed)==tokens(nested)
            keep_result = output_row(keep[i],"switch_keep",budget,prefix=prior[i],switched=True,marker=marker)
            reset_result = output_row(resumed,"switch_restart",budget,prefix=prior[i],switched=True,
                                      source_arm="explicit_restart" if restart is not None else "question_e0",marker=marker)
            reset_result["nested_parity"] = matches if restart is not None else None
            rows.extend((keep_result,reset_result))
        else:
            rows.extend(output_row(native[i],arm,budget,source_arm="native_early") for arm in ("switch_keep","switch_restart"))
    meta = {"kind":"batch","width":width,"ids":[x["row"]["id"] for x in group],"time_seconds":seconds,
            "question_sha256":[packed_hash(x["native_ids"]) for x in group],
            "direct_sha256":[packed_hash(x["no_think_ids"]) for x in group],
            "render_sha256":[x["render_sha256"] for x in group],"scratchpad_sha256":[packed_hash(s) for s in prior],
            **probe}
    return rows,meta


def run(args):
    cohort = args.root/"cohort.json"
    plan = load(cohort)
    budget,switch = plan["budget"],plan["switch"]
    if args.phase=="smoke":
        selected = plan["pilot"][:4]
    else:
        selected = plan[args.phase][args.part::args.partitions]
    r = Runtime(args)
    items = prepare_items(r,selected)
    start = time.monotonic()
    with args.output.open("x") as f,r.torch.inference_mode():
        def emit(obj):
            print(json.dumps(obj,allow_nan=False),file=f)
        emit({"kind":"manifest","part":args.part,"partitions":args.partitions,"phase":args.phase,"draw":args.draw,
              "arms":ARMS,"base_commit":BASE,"budget":budget,"switch":switch,"cohort_sha256":sha(cohort),
              "worker_sha256_policy":sha(Path(__file__)),"ids":[x["row"]["id"] for x in items],**r.manifest})
        for i in range(0,len(items),BATCH):
            if time.time()>=args.deadline-90:
                raise TimeoutError("restart hard deadline")
            rows,meta = batch_run(r,items[i:i+BATCH],args.draw,budget,switch,args.phase!="eval")
            emit(meta)
            for row in rows:
                emit(row)
            f.flush()
            print(f"{args.phase} draw={args.draw} prompts={i+len(meta['ids'])}/{len(items)}",flush=True)
        emit({"kind":"complete","seconds":time.monotonic()-start})


def score_policy(row, record, tokenizer, turn):
    decoded = tokenizer.decode(record["tokens"],skip_special_tokens=False)
    requires_close = record["arm"]!="no_think"
    a = math_answer(row,decoded,pending_thought=requires_close)
    eos = record["eos_token"] is not None
    visible = record["tokens"]
    if turn in visible:
        visible = visible[:visible.index(turn)]
    strict_visible = math_answer(row,tokenizer.decode(visible,skip_special_tokens=False),
                                 pending_thought=requires_close)["exact_format"]
    return {"correct_eos":eos and a["numeric_correct"],"eos":eos,"wrong_eos":eos and a["answer_found"] and not a["numeric_correct"],
            "missing_eos":eos and not a["answer_found"],"cap":not eos,"strict_boxed":a["exact_format"],
            "visible_boxed":strict_visible,"numeric_correct":a["numeric_correct"],"switched":record["switched"]}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase",choices=("smoke","pilot","eval"),required=True)
    for name in ("root","model","data","output"):
        p.add_argument("--"+name,type=Path,required=True)
    for name,default in (("part",0),("partitions",1),("draw",0),("deadline",None)):
        p.add_argument("--"+name,type=int,default=default)
    run(p.parse_args())
