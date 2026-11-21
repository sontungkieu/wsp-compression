"""E0 factorial over native scratchpad context and sampler history."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from repetition_baseline import sha
from repetition_budget_run import packed_hash
from repetition_budget_score import cut
from repetition_canonical import seed_for
from repetition_exit_localize import capture
from repetition_mediation import presence_ids, presence_scores
from repetition_mediation_queue import load
from repetition_sampler_reach import pack_at
from repetition_state_data import STRATA, THINK_END
from repetition_state_value import Runtime, inputs_for, post_seed

ARMS = ("keep_context_keep_history", "keep_context_reset_history",
        "drop_context_keep_history", "drop_context_reset_history")
CAP = 512
BATCH_SIZE = 18
BASE_COMMIT = "ec5e4446fe193873c3e6cfad22818b473e4c19d9"


def factor(item):
    scratch = item["completion_prefix"]
    source = item["native_ids"]
    prompt = source[:len(source)-len(scratch)]
    if not prompt or source != prompt + scratch or THINK_END in scratch:
        raise ValueError("scratchpad did not match the canonical input suffix")
    history = sorted(set(scratch))
    return {name: item | {"context_ids": source if name.startswith("keep_context") else prompt,
                          "presence_seed_ids": history if name.endswith("keep_history") else []}
            for name in ARMS}


def prepare_group(runtime, base):
    rows = [factor(item) for item in base]
    batches = {a:[r[a] for r in rows] for a in ARMS}
    width = max(len(x["context_ids"]) for x in batches[ARMS[0]])
    packed = {}
    for keep, drop in ((ARMS[0],ARMS[1]),(ARMS[2],ARMS[3])):
        packed[keep] = pack_at(runtime, [x["context_ids"] for x in batches[keep]], width)
        packed[drop] = packed[keep]
    records = []
    for i,x in enumerate(base):
        f = rows[i]
        r, s, h = f[ARMS[2]]["context_ids"], x["completion_prefix"], f[ARMS[0]]["presence_seed_ids"]
        ids = {a:packed_hash(f[a]["context_ids"]) for a in ARMS}
        masks = {a:packed_hash([0]*(width-len(f[a]["context_ids"]))+[1]*len(f[a]["context_ids"])) for a in ARMS}
        records.append({"id":x["row"]["id"],"stratum":x["stratum"],"origin":x.get("origin"),
                        "render_sha256":x["render_sha256"],"prompt_tokens_sha256":packed_hash(r),
                        "scratchpad_sha256":packed_hash(s),"history_sha256":packed_hash(h),
                        "length":len(s),"history_tokens":len(h),"width":width,
                        "arm_prompt_sha256":ids,"arm_mask_sha256":masks,
                        "arm_history_sha256":{a:packed_hash(f[a]["presence_seed_ids"]) for a in ARMS}})
    return batches, packed, records


def make_policy(runtime, batch, packed):
    ids = [x["row"]["id"] for x in batch]
    return runtime.policy(batch, packed["input_ids"].shape[1], [seed_for(p,0) for p in ids],
                          [post_seed(p,0) for p in ids],0,0.)


def tensor_sha(tensor):
    return hashlib.sha256(tensor.float().detach().cpu().numpy().tobytes()).hexdigest()


def first_step_audit(runtime, batches, packed):
    torch = runtime.torch
    raw, mask, adjusted, now = {}, {}, {}, {}
    for arm in ARMS:
        batch, inputs = batches[arm], packed[arm]
        logits,_ = capture(runtime.model, inputs, {}, use_cache=True)
        policy = make_policy(runtime, batch, inputs)
        prefix = inputs["input_ids"]
        empty = prefix[:, :0]
        tag = torch.full((len(batch),1),THINK_END,device=prefix.device,dtype=prefix.dtype)
        first = policy.seen_mask(empty, logits)
        after = policy.seen_mask(tag, logits)
        for i,item in enumerate(batch):
            expected = presence_ids(item)
            if (first[i].nonzero().flatten().tolist()!=expected
                    or after[i].nonzero().flatten().tolist()!=sorted(set(expected)|{THINK_END})):
                raise ValueError("presence mask disagrees with the declared history and new tag")
        raw[arm], mask[arm], now[arm] = logits, first, after
        adjusted[arm] = presence_scores(logits,first)
    for a,b in ((ARMS[0],ARMS[1]),(ARMS[2],ARMS[3])):
        if any(not torch.equal(packed[a][k],packed[b][k]) for k in ("input_ids","attention_mask")):
            raise ValueError("history toggle unexpectedly changed input or padding")
        if not torch.equal(raw[a],raw[b]):
            raise ValueError("history toggle unexpectedly changed raw first-step logits")
        if not torch.equal(adjusted[a]!=adjusted[b],mask[a]^mask[b]):
            raise ValueError("history toggle changed logits outside the intended token IDs")
    if not torch.equal(mask[ARMS[0]],mask[ARMS[2]]) or not torch.equal(mask[ARMS[1]],mask[ARMS[3]]):
        raise ValueError("context toggle unexpectedly changed the initial history")
    return {a:{"raw_logits_sha256":tensor_sha(raw[a]),"adjusted_logits_sha256":tensor_sha(adjusted[a]),
               "seen_counts":mask[a].sum(-1).tolist(),"seen_counts_after_tag":now[a].sum(-1).tolist()} for a in ARMS}


def verify_trace(record, cap, eos_ids):
    t = record["tokens"]
    if not t or t[0]!=THINK_END or record["think_step"]!=0 or record["forced_step"]!=0:
        raise ValueError("forced marker was not emitted at step zero")
    if (len(t)>cap+1 or record["eos_token"] not in (None,*eos_ids)
            or (record["eos_token"] is not None and len(t)>cap)
            or (record["eos_token"] is None and len(t)!=cap+1)):
        raise ValueError("output exceeds post-marker cap or uses an invalid EOS")
    a, eos = cut(record,"post",cap)
    if a!=t or eos!=record["eos"]:
        raise ValueError("full window lost a recorded EOS")
    b, short_eos = cut(record,"post",min(256,cap))
    if b!=t[:min(256,cap)+1] or short_eos!=(eos and len(t)<=min(256,cap)):
        raise ValueError("short window used padding or an out-of-window EOS")
    if record["post_seed"]!=post_seed(record["id"],record["draw"]):
        raise ValueError("post-marker seed mismatch")


def smoke(runtime, rows, out):
    base = inputs_for(runtime.tokenizer, rows)
    batches, packed, hashes = prepare_group(runtime, base)
    emit = lambda data: print(json.dumps(data,allow_nan=False),file=out,flush=True)
    evidence = first_step_audit(runtime, batches, packed)
    for arm in ARMS:
        for row in runtime.generate(batches[arm],"e0",0,CAP,packed=packed[arm]):
            verify_trace(row,CAP,runtime.eos)
            emit(row | {"arm":arm})
    empty = inputs_for(runtime.tokenizer,[row | {"completion_prefix":[]} for row in rows])
    b,p,_ = prepare_group(runtime,empty)
    # Once the source scratchpad is empty, all four arms have identical inputs
    # and history. Confirm both the instrumented logits and the actual tokens.
    empty_audit = first_step_audit(runtime,b,p)
    results = [runtime.generate(b[a],"e0",0,32,packed=p[a]) for a in ARMS]
    traces = [[(r["tokens"],r["eos_token"]) for r in batch] for batch in results]
    if any(batch!=traces[0] for batch in traces[1:]):
        raise ValueError("empty scratchpad failed exact paired rollout parity")
    emit({"kind":"audit","first":evidence,"empty":empty_audit,"input_hashes":hashes,
          "empty_pair_exact":True,"gpu_canary":True})


def run(args):
    input_file = args.canary if args.action=="smoke" else args.root/"cohort.json"
    manifest = load(input_file)
    rows = manifest["cases"]
    if args.action=="smoke":
        rows = [next(i for i in rows if i["stratum"]==g) for g in STRATA]
    else:
        rows = rows[args.part::args.partitions]
    runtime = Runtime(args)
    begin = time.monotonic()
    with args.output.open("x") as out, runtime.torch.inference_mode():
        def emit(data):
            print(json.dumps(data,allow_nan=False),file=out)
        emit({"kind":"manifest","partitions":args.partitions,"part":args.part,"draw":args.draw,"arms":ARMS,
              "cohort_sha256":sha(input_file),"base_commit":BASE_COMMIT,"factorial_sha256":sha(Path(__file__)),
              "cap":CAP,"batch_size":BATCH_SIZE,"prompt_ids":[x["row"]["id"] for x in rows],**runtime.manifest})
        if args.action=="smoke":
            smoke(runtime,rows,out)
        else:
            items = inputs_for(runtime.tokenizer,rows)
            for start in range(0,len(items),BATCH_SIZE):
                batch, packed, hashes = prepare_group(runtime,items[start:start+BATCH_SIZE])
                for h in hashes:
                    emit({"kind":"inputs",**h})
                for arm in ARMS:
                    if time.time()>=args.deadline-45:
                        raise TimeoutError("scratchpad hard deadline")
                    results = runtime.generate(batch[arm],"e0",args.draw,CAP,packed=packed[arm])
                    for row in results:
                        verify_trace(row,CAP,runtime.eos)
                        emit(row | {"arm":arm})
                    out.flush()
                    print(f"draw={args.draw} prompts={start+len(batch[arm])}/{len(items)} arm={arm}",flush=True)
        emit({"kind":"complete","seconds":time.monotonic()-begin})


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("smoke","run"))
    for name in ("root","model","data","canary","output"):
        p.add_argument("--"+name,type=Path)
    for name,default in (("part",0),("partitions",1),("draw",0),("deadline",None)):
        p.add_argument("--"+name,type=int,default=default)
    run(p.parse_args())
