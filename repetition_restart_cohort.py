"""Freeze disjoint numeric math panels for policies starting from the question."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from repetition_baseline import DECIMAL, astra_instruction, sha, validate_astra_prompt
from repetition_mediation_queue import load, save
from repetition_state_data import keyed


def read_source(path):
    if path.suffix==".parquet":
        from pyarrow.parquet import read_table
        return read_table(path).to_pylist()
    if path.suffix==".jsonl":
        with path.open() as f:
            return [json.loads(line) for line in f if line.strip()]
    items = load(path)
    if not isinstance(items,list):
        raise ValueError("expected a JSON array of math examples")
    return items


def canonical_problem(problem):
    instruction = astra_instruction("canonical")
    problem = problem.rstrip()
    prompt = problem if problem.endswith(instruction) else problem+"\n"+instruction
    validate_astra_prompt(prompt, "canonical")
    return prompt


def freeze(source, dataset, output, exposure=None, scratch=None):
    rows = read_source(source)
    rejected = set()
    if dataset=="gsm8k":
        if exposure is None or scratch is None:
            raise ValueError("GSM8K panel requires prior exposure receipts")
        rejected.update(load(exposure)["excluded_discovery_ids"])
        for path,_ in load(scratch/"cohort.json")["discovery_sources"]:
            with Path(path).open() as f:
                rejected.update(x["id"] for x in json.loads(next(f))["panel"])
    pool = []
    for i,r in enumerate(rows):
        if dataset=="math500":
            problem = r["problem"]
            # The staged public mirror retained problem and answer only.
            ident = "math500:"+hashlib.sha256(problem.encode()).hexdigest()
            gold = str(r["answer"]).strip()
        else:
            ident, problem = f"gsm8k:{i}",r["question"]
            gold = r["answer"].rsplit("####",1)[-1].strip()
        if ident in rejected or not DECIMAL.fullmatch(gold):
            continue
        case = {"id":ident,"task":"math","prompt":canonical_problem(problem),
                "gold":gold.replace(",",""),"boxed_answer_required":True,"astra_prompt_arm":"canonical"}
        pool.append({"row":case,"subject":r.get("subject",dataset),"level":r.get("level")})
    if len(pool)<28 or len({r["row"]["id"] for r in pool})!=len(pool) or output.exists():
        raise ValueError("insufficient distinct numeric problems or existing output")
    pool.sort(key=lambda r:keyed(r["row"]["id"],"from-question-restart-v1"))
    pilot, test = pool[:16],pool[16:80]
    report = {"dataset":dataset,"scope":"finite decimal answers compatible with the pinned canonical scorer",
              "source_path":str(source),"source_sha256":sha(source),"pool":len(pool),
              "pilot":pilot,"eval":test,"budget":2048,"switch":512,
              "eval_subjects":dict(Counter(x["subject"] for x in test)),
              "prior_exposure_sha256":sha(exposure) if exposure is not None else None}
    save(output,report)
    print(json.dumps({"cohort_sha256":sha(output),"dataset":dataset,"pool":len(pool),"pilot":len(pilot),
                      "eval":len(test),"subjects":report["eval_subjects"],"budget":2048,"switch":512}))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",choices=("math500","gsm8k"),required=True)
    for name in ("source","output","exposure","scratch"):
        p.add_argument("--"+name,type=Path,required=name in ("source","output"))
    a=p.parse_args()
    freeze(a.source,a.dataset,a.output,a.exposure,a.scratch)
