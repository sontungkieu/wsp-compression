"""Verify paired factorial shards and report prompt-clustered effects."""
import argparse
import json
from pathlib import Path

from repetition_baseline import sha
from repetition_budget_run import packed_hash
from repetition_budget_score import assess_window
from repetition_canonical import paired_interval, seed_for
from repetition_mediation import SAMPLING
from repetition_mediation_queue import load, save
from repetition_scratchpad import ARMS, CAP, verify_trace
from repetition_state_value import post_seed

KK, KR, DK, DR = ARMS
EFFECTS = {
    "keep_context_when_reset": {KR:1,DR:-1},
    "keep_context_when_keep": {KK:1,DK:-1},
    "keep_history_when_drop": {DK:1,DR:-1},
    "keep_history_when_keep": {KK:1,KR:-1},
    "context_history_interaction": {KK:1,DK:-1,KR:-1,DR:1},
    "drop_context_and_reset_history": {DR:1,KK:-1},
}
METRICS = ("eos_correct","success","numeric_correct","correct_without_eos","eos",
           "wrong_parsed_eos","missing_answer_eos","unfinished","terminal_cycle",
           "exact_format","visible_exact_format","multiple_close")


def verify_inputs(row, expected):
    s = expected["completion_prefix"]
    h = sorted(set(s))
    if (row["length"]!=len(s) or row["history_tokens"]!=len(h)
            or row["render_sha256"]!=expected["render_sha256"]
            or row["scratchpad_sha256"]!=packed_hash(s) or row["history_sha256"]!=packed_hash(h)):
        raise ValueError("a recorded input does not have the selected scratchpad")
    hashes = row["arm_history_sha256"]
    if hashes!={KK:packed_hash(h),DK:packed_hash(h),KR:packed_hash([]),DR:packed_hash([])}:
        raise ValueError("an arm used the wrong scratchpad history set")
    for field in ("arm_prompt_sha256","arm_mask_sha256"):
        v = row[field]
        contexts_should_match = not s
        if v[KK]!=v[KR] or v[DK]!=v[DR] or (v[KK]==v[DK])!=contexts_should_match:
            raise ValueError("context/history input or mask parity failed")
    if row["arm_prompt_sha256"][DK]!=row["prompt_tokens_sha256"]:
        raise ValueError("dropped context did not match the canonical prompt-only tokens")


def read_matrix(root):
    manifest = root/"cohort.json"
    cases = {x["row"]["id"]:x for x in load(manifest)["cases"]}
    source = sha(manifest)
    ready = sorted((root/"queue").glob("draw*-done.json"))
    if not ready:
        raise ValueError("no complete factorial draw")
    observations, expected_layout, receipts = {}, {}, []
    for draw,receipt in enumerate(ready):
        if receipt.name!=f"draw{draw}-done.json":
            raise ValueError("draw receipts have gaps")
        lanes, seen = set(), set()
        for file,digest in load(receipt)["outputs"]:
            path = Path(file)
            if sha(path)!=digest:
                raise ValueError("factorial shard checksum mismatch")
            receipts.append([str(path),digest])
            with path.open() as f:
                header = json.loads(next(f))
                lane, prompts = header["part"], header["prompt_ids"]
                if (header["draw"]!=draw or header["cohort_sha256"]!=source or header["cap"]!=CAP
                        or tuple(header["arms"])!=ARMS or header["sampling"]!=SAMPLING or lane in lanes
                        or set(prompts)&seen or expected_layout.setdefault(lane,prompts)!=prompts):
                    raise ValueError("draw contract, shard order, or sampler changed")
                lanes.add(lane)
                seen.update(prompts)
                inputs = {}
                completed = False
                for line in f:
                    row = json.loads(line)
                    if row["kind"]=="complete":
                        completed = True
                        continue
                    ident = row["id"]
                    if ident not in prompts or completed:
                        raise ValueError("row outside its source partition")
                    if row["kind"]=="inputs":
                        verify_inputs(row,cases[ident])
                        if ident in inputs or expected_layout.setdefault(ident,row)!=row:
                            raise ValueError("duplicate or changing arm packing across draws")
                        inputs[ident] = row
                        continue
                    key = ident,draw,row["arm"]
                    if (row["kind"]!="rollout" or row["draw"]!=draw or row["seed"]!=seed_for(ident,draw)
                            or row["post_seed"]!=post_seed(ident,draw) or key in observations):
                        raise ValueError("invalid or duplicate factorial rollout")
                    verify_trace(row,CAP,header["eos_ids"])
                    observations[key] = row
                if set(inputs)!=set(prompts) or not completed:
                    raise ValueError("missing input metadata or completion footer")
        if lanes!={0,1} or seen!=set(cases):
            raise ValueError("incomplete draw panel")
    expected = {(p,d,a) for p in cases for d in range(len(ready)) for a in ARMS}
    if observations.keys()!=expected:
        raise ValueError("incomplete paired arm matrix")
    return cases,observations,receipts,len(ready)


def paired_effect(scored, prompt_ids, draws, weights, metric):
    return paired_interval({p:[sum(weight*int(scored[p,d,arm][metric])
                                   for arm,weight in weights.items()) for d in range(draws)] for p in prompt_ids})


def summarize(cases, observations, receipts, draws, tokenizer):
    [turn] = tokenizer.encode("<|im_end|>",add_special_tokens=False)
    groups = {"all":list(cases)}
    for p,case in cases.items():
        groups.setdefault("stratum:"+case["stratum"],[]).append(p)
        groups.setdefault("origin:"+case["origin"],[]).append(p)
        length = len(case["completion_prefix"])
        band = "under_1024" if length<1024 else "1024_to_3071" if length<3072 else "3072_or_more"
        groups.setdefault("length:"+band,[]).append(p)
    report = {"observations":len(observations),"draws":draws,"receipts":receipts,"windows":{}}
    for budget in (256,CAP):
        y = {k:assess_window(cases[k[0]]["row"],r,"post",budget,tokenizer,turn)
             for k,r in observations.items()}
        table = {}
        for group,ids in groups.items():
            counts = {}
            for arm in ARMS:
                results = [y[p,d,arm] for p in ids for d in range(draws)]
                counts[arm] = {"n":len(results),**{m:sum(bool(r[m]) for r in results) for m in METRICS}}
            effects = {name:{metric:paired_effect(y,ids,draws,weight,metric)
                             for metric in ("eos_correct","success")} for name,weight in EFFECTS.items()}
            table[group] = {"prompts":len(ids),"arms":counts,"effects":effects}
        report["windows"][str(budget)] = table
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root","model","output"):
        p.add_argument("--"+name,type=Path,required=True)
    a = p.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    out = summarize(*read_matrix(a.root),tokenizer)
    if a.output.exists():
        raise FileExistsError(a.output)
    save(a.output,out)
    print(json.dumps({"report_sha256":sha(a.output),"n":out["observations"],"draws":out["draws"]}))
