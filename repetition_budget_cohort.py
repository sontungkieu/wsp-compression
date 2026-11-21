"""Freeze a balanced timing panel before measuring any intervention."""
import argparse
import json
from pathlib import Path

from repetition_baseline import sha
from repetition_mediation_queue import load, save
from repetition_state_data import STRATA, candidate_states, keyed, length_summary, load_traces


def prepare(baseline, state, reach, output):
    traces, shards = load_traces(baseline)
    prior_files = [p / "cohort.json" for p in (baseline, state, reach)]
    old = {r["row"]["id"] for r in load(prior_files[0])["cases"]}
    for path in prior_files[1:]:
        c = load(path)
        old.update(r["row"]["id"] for split in ("dev", "eval") for r in c[split])
    with (baseline / "canary.jsonl").open() as f:
        old.update(r["id"] for r in json.loads(next(f))["panel"])
    pool = candidate_states(traces)
    rows, used, question_only = [], set(), []
    for group in ("pre_onset", "loop", "progress"):
        def order(p):
            return keyed(p, "timing-budget-v1:" + group)
        ids = pool[group].keys() - used
        fresh = sorted(ids - old, key=order)[:20]
        repeated = sorted(ids & old, key=order)[:20-len(fresh)]
        selected = fresh + repeated
        if len(selected) != 20:
            raise ValueError("canonical pool does not cover 20 independent prompts per stratum")
        rows.extend(pool[group][p] | {"eval_origin": "previously_scored" if p in old else "fresh"} for p in selected)
        question_only.extend(sorted(selected, key=lambda p: keyed(p, "timing-budget-v1:question-only"))[:4])
        used.update(selected)
    rows.sort(key=lambda r: keyed(r["row"]["id"], "timing-budget-v1:order"))
    manifest = {"version": "timing-budget-v1", "cases": rows, "question_only": question_only,
                "baseline_shards": shards, "prior_cohorts": {str(p): sha(p) for p in prior_files},
                "length_summary": length_summary(rows)}
    if output.exists():
        raise FileExistsError(output)
    save(output, manifest)
    print(json.dumps({"cohort_sha256": sha(output), "n": len(rows), "question_only": len(question_only),
                      "fresh": {g: sum(r["stratum"]==g and r["eval_origin"]=="fresh" for r in rows) for g in STRATA},
                      "length_summary": manifest["length_summary"]}))


def prepare_extension(source, output):
    base = load(source)
    chosen = []
    for group in STRATA:
        candidates = sorted((r for r in base["cases"] if r["stratum"] == group),
                            key=lambda r: keyed(r["row"]["id"], "timing-budget-v1:3072"))
        chosen.extend(candidates[:8])
    if len(chosen) != 24 or output.exists():
        raise ValueError("expected 24 frozen extension cases and a fresh output path")
    chosen.sort(key=lambda r: keyed(r["row"]["id"], "timing-budget-v1:3072-order"))
    save(output, {**base, "version": "timing-budget-extension-v1", "cases": chosen, "question_only": [],
                  "parent_cohort_sha256": sha(source), "length_summary": length_summary(chosen)})
    print(json.dumps({"cohort_sha256": sha(output), "n": len(chosen)}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "state", "reach", "output"):
        p.add_argument("--" + name, type=Path, required=(name=="output"))
    p.add_argument("--extend-from", type=Path)
    a = p.parse_args()
    if a.extend_from is not None:
        prepare_extension(a.extend_from, a.output)
    elif all((a.baseline, a.state, a.reach)):
        prepare(a.baseline, a.state, a.reach, a.output)
    else:
        p.error("provide either the three archive roots or an existing cohort to extend")
