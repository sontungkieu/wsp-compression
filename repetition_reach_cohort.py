"""Freeze prompt-disjoint panels for native-context reachability."""
import argparse
from pathlib import Path

from repetition_baseline import sha
from repetition_mediation_queue import load, save
from repetition_state_data import STRATA, candidate_states, keyed, length_summary, load_traces


def prepare(canonical, prior_state, output):
    traces, sources = load_traces(canonical)
    state_path = prior_state / "cohort.json"
    state = load(state_path)
    previous_ids = {x["row"]["id"] for x in load(canonical / "cohort.json")["cases"]}
    import json
    with (canonical / "canary.jsonl").open() as stream:
        previous_ids.update(x["id"] for x in json.loads(next(stream))["panel"])
    dev = [x for x in state["dev"] if x["stratum"] in STRATA]
    dev_ids = {x["row"]["id"] for x in state["dev"]}
    old_eval_ids = {x["row"]["id"] for x in state["eval"]}
    forbidden = previous_ids | dev_ids | old_eval_ids
    pool = candidate_states(traces)
    eval_rows, used = [], set()
    # Reserve pre-onset before loop because they share the same source pool.
    for group in ("pre_onset", "loop", "progress"):
        fresh = sorted(pool[group].keys() - forbidden - used, key=lambda k: keyed(k, "reach:" + group))[:10]
        selected = [pool[group][k] | {"eval_origin": "fresh"} for k in fresh]
        available = [r for r in state["eval"] if r["stratum"] == group and r["row"]["id"] not in used]
        selected += [r | {"eval_origin": "reproduction"} for r in available[:10 - len(selected)]]
        eval_rows.extend(selected)
        used.update(r["row"]["id"] for r in selected)
    if len(dev) != 24 or len(eval_rows) != 30 or len(used) != 30 or used & (previous_ids | dev_ids):
        raise ValueError("could not construct the preregistered prompt-disjoint panels")
    for panel in (dev, eval_rows):
        panel.sort(key=lambda r: keyed(r["row"]["id"], "reach:order"))
    result = {"version": "sampler-reach-v1", "dev": dev, "eval": eval_rows, "source_shards": sources,
              "prior_cohort_sha256": sha(state_path), "length_summary": {"dev": length_summary(dev),
              "eval": length_summary(eval_rows)},
              "reused_eval": [r["row"]["id"] for r in eval_rows if r["eval_origin"] == "reproduction"]}
    if output.exists():
        raise FileExistsError(output)
    save(output, result)
    print({"cohort_sha256": sha(output), "dev": len(dev), "eval": len(eval_rows),
           "reused_eval": result["reused_eval"], "length_summary": result["length_summary"]})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("canonical", "prior-state", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    prepare(a.canonical, a.prior_state, a.output)


if __name__ == "__main__":
    main()
