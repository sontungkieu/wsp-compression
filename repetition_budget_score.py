"""Score nested cuts of recorded reasoning trajectories without regenerating."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from repetition_baseline import math_answer, sha
from repetition_budget_run import ARMS, QUESTION
from repetition_canonical import score, seed_for
from repetition_canonical_queue import paired_interval
from repetition_mediation_queue import load, save
from repetition_state_value import THINK_END, post_seed, read_matrix

POST = (256, 512, 1024, 1536)
TOTAL = (512, 1024, 1536)
METRICS = ("success", "closed", "eos", "eos_correct", "wrong_parsed_eos", "missing_answer_eos",
           "unfinished", "correct_without_eos", "numeric_correct", "exact_format", "visible_exact_format",
           "terminal_cycle", "eos_before_close", "multiple_close")


def cut(record, basis, budget):
    """An EOS token at continuation index j counts only if j < the window length."""
    ids = record["tokens"]
    at = ids.index(THINK_END) if THINK_END in ids else None
    limit = budget if basis == "total" else at+1+budget if at is not None else len(ids)+1
    inside = ids[:limit]
    eos = record["eos_token"] is not None and len(ids) < limit
    return inside, eos


def assess_window(row, record, basis, budget, tokenizer, chat_end):
    inside, eos = cut(record, basis, budget)
    at = inside.index(THINK_END) if THINK_END in inside else None
    decoded = tokenizer.decode(inside, skip_special_tokens=False)
    counted = score(row, decoded, inside, eos, len(inside), at)
    # Keep the archived strict boxed metric. Also score exact visible formatting
    # at the explicit Qwen assistant turn boundary using the same numeric parser.
    shown = inside[:inside.index(chat_end)] if chat_end in inside else inside
    visible = (counted["exact_format"] if len(shown)==len(inside) else
               math_answer(row, tokenizer.decode(shown, skip_special_tokens=False))["exact_format"])
    counted.update(closed=at is not None, eos_correct=eos and counted["numeric_correct"],
                   unfinished=not eos, correct_without_eos=counted["numeric_correct"] and not eos,
                   eos_before_close=eos and at is None, multiple_close=inside.count(THINK_END)>1,
                   visible_exact_format=visible)
    return counted


def load_budget(root):
    cohort_path = root/"cohort.json"
    manifest = load(cohort_path)
    cases = {x["row"]["id"]: x for x in manifest["cases"]}
    source = sha(cohort_path)
    expected_question = set(manifest["question_only"])
    ready = sorted((root/"queue").glob("draw*-done.json"))
    if not ready:
        raise ValueError("no complete paired wave")
    raw, trunks, receipts = {}, {}, []
    for wave, receipt in enumerate(ready):
        if receipt.name != f"draw{wave}-done.json":
            raise ValueError("noncontiguous draw receipts")
        parts, wave_cases, question = set(), set(), set()
        for file, digest in load(receipt)["outputs"]:
            p = Path(file)
            if sha(p)!=digest:
                raise ValueError("worker output digest mismatch")
            receipts.append([str(p),digest])
            with p.open() as f:
                h = json.loads(next(f))
                if h["cohort_sha256"]!=source or h["wave"]!=wave or h["part"] in parts or h["draw"]!=wave:
                    raise ValueError("wrong cohort or draw")
                parts.add(h["part"])
                part_ids = {x["id"] for x in h["cases"]}
                thin = {x["id"] for x in h["question_only"]}
                if wave_cases & part_ids or not thin<=part_ids:
                    raise ValueError("wrong shard partition")
                wave_cases.update(part_ids)
                question.update(thin)
                footer = False
                for line in f:
                    r = json.loads(line)
                    if r["kind"]=="complete":
                        footer = True
                        continue
                    if r["id"] not in part_ids or r["draw"]!=wave or footer:
                        raise ValueError("bad rollout identity")
                    k = r["id"],wave
                    if r["kind"]=="trunk":
                        if k in trunks:
                            raise ValueError("duplicate trunk check")
                        trunks[k] = r["checks"]
                        continue
                    if (r["kind"]!="rollout" or r["seed"]!=seed_for(*k) or r["post_seed"]!=post_seed(*k)
                            or (r["eos_token"] is not None and r["eos_token"] not in h["eos_ids"])):
                        raise ValueError("wrong timing seed or EOS ID")
                    k = (*k,r["arm"])
                    if k in raw:
                        raise ValueError("duplicate timing record")
                    raw[k] = r
                if not footer:
                    raise ValueError("missing timing completion footer")
        if parts!={0,1} or wave_cases!=set(cases) or expected_question!=question:
            raise ValueError("incomplete paired prompt coverage")
    expected = {(p,d,a) for p in cases for d in range(len(ready)) for a in (*ARMS,QUESTION)
                if a!=QUESTION or p in expected_question}
    if raw.keys()!=expected or set(trunks)!={(p,d) for p,d,_ in expected}:
        raise ValueError("incomplete rollout/trunk matrix")
    return cases, raw, trunks, receipts


def load_archive(root, phase):
    _, _, raw = read_matrix(root, phase)
    cases = {x["row"]["id"]: x for x in load(root/"cohort.json")["eval"]}
    return cases, raw, {}, load(root/"queue"/f"{phase}-done.json")["outputs"]


def summarize_budget(cases, raw, trunks, receipts, tokenizer, post_horizons, totals):
    [chat_end] = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    groups = {"all": set(cases)}
    strata = sorted({v["stratum"] for v in cases.values()})
    for name in strata:
        groups[name] = {p for p,v in cases.items() if v["stratum"]==name}
    for origin in sorted({v.get("eval_origin") for v in cases.values()} - {None}):
        groups[origin] = {p for p,v in cases.items() if v.get("eval_origin")==origin}
        for name in strata:
            selected = groups[name] & groups[origin]
            if selected:
                groups[f"{name}:{origin}"] = selected
    arms = sorted({a for _,_,a in raw})
    comparisons = (("e0","e128"),("e0","e32"),("e0",QUESTION),("e0","native"),
                   ("e128","e32"),("e32","native"),("e128","native"))
    output = {"observations": len(raw), "source_receipts": receipts, "windows": {}}
    for basis, horizons in (("post",post_horizons),("total",totals)):
        for budget in horizons:
            scored = {k: assess_window(cases[k[0]]["row"],r,basis,budget,tokenizer,chat_end)
                      for k,r in raw.items() if basis=="total" or k[2]!="native"}
            window = {}
            for group, ids in groups.items():
                gs = {}
                for a in arms:
                    obs = [v for (p,_,k),v in scored.items() if p in ids and a==k]
                    if not obs:
                        continue
                    gs[a] = {m:sum(bool(r[m]) for r in obs) for m in METRICS}
                    gs[a]["n"] = len(obs)
                    gs[a]["median_post_tokens"] = median([r["after_think_tokens"] for r in obs if r["closed"]])
                    gs[a]["median_close_step"] = median([r["think_step"] for r in obs if r["closed"]])
                contrasts = {}
                for a,b in comparisons:
                    diffs = {}
                    if a not in gs or b not in gs:
                        continue
                    for p,d,k in scored:
                        if p in ids and k==a and (p,d,b) in scored:
                            diffs.setdefault(p,[]).append(int(scored[p,d,a]["success"])-int(scored[p,d,b]["success"]))
                    contrasts[f"{a}_minus_{b}"] = paired_interval(diffs)
                window[group] = {"prompts":len(ids),"arms":gs,"paired_success":contrasts}
            output["windows"][f"{basis}:{budget}"] = window
    trunk = Counter()
    for checks in trunks.values():
        for pair,m in checks.items():
            for label,v in m.items():
                trunk[pair,label,str(v)] += 1
    output["common_trunk"] = {":".join(key):n for key,n in sorted(trunk.items())}
    output["actual_closure"] = {a: {"n":len(r),"forced":sum(v["forced_step"] is not None for v in r),
                    "natural_close":sum(v["think_step"] is not None and v["forced_step"] is None for v in r),
                    "eos_before_close":sum(v["eos_before_think"] for v in r)}
                    for a in arms if (r:=[v for (_,_,k),v in raw.items() if a==k])}
    return output


def median(numbers):
    return statistics.median(numbers) if numbers else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("budget","archive1536","archive512"))
    for name in ("root","model","output"):
        p.add_argument("--"+name,type=Path,required=True)
    args = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    if args.mode == "budget":
        data = load_budget(args.root)
        horizons, totals = POST,TOTAL
    else:
        data = load_archive(args.root,"labels-eval" if args.mode=="archive1536" else "replay")
        horizons, totals = (POST,(256,*TOTAL)) if args.mode=="archive1536" else ((256,512),(256,512))
    result = summarize_budget(*data, tok, horizons, totals)
    if args.output.exists():
        raise FileExistsError(args.output)
    save(args.output,result)
    print(json.dumps({"mode":args.mode,"observations":result["observations"],
                      "scorer_sha256":sha(Path(__file__)),"windows":list(result["windows"])}))


if __name__ == "__main__":
    main()
