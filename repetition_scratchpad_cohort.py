"""Audit WSP exposure and freeze new native scratchpad states."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

from repetition_baseline import panel, sha
from repetition_mediation_queue import load, save
from repetition_state_data import STRATA, candidate_states, keyed, length_summary, load_traces, take

GSM8K = re.compile(rb"gsm8k:[0-9]+")
SHARDS = range(41)
TARGET = 24


def id_ledger(runs, baseline, output, data):
    baseline_summary = baseline / "baseline-summary.json"
    archived_traces = load(baseline_summary)["shards"]
    untouched_sources = {Path(path).resolve() for path, _ in archived_traces}
    touched, by_run, files = set(), {}, 0
    for folder in sorted(x for x in runs.iterdir() if x.is_dir() and x != output.parent):
        hits = set()
        for path in folder.rglob("*"):
            if (path.suffix not in {".json", ".jsonl"} or not path.is_file() or path.is_symlink()
                    or path.resolve() in untouched_sources or ".bridge-backups" in path.parts):
                continue
            files += 1
            with path.open("rb") as stream:
                for line in stream:
                    hits.update(x.decode() for x in GSM8K.findall(line))
        if hits:
            by_run[folder.name] = len(hits)
            touched.update(hits)
    forbid = set(touched)
    traces, _ = load_traces(baseline)
    forbid.update(traces)
    # The panel shuffler consumes the same pinned dataset rows as older runs.
    available = [shard for shard in SHARDS if
                 {r["id"] for r in panel(data, 32, shard, astra_math=True)}.isdisjoint(forbid)]
    ordered = sorted(available, key=lambda a: keyed(str(a), "scratchpad-discovery-shard"))
    ledger = {"excluded_intervention_ids": sorted(touched), "excluded_discovery_ids": sorted(forbid),
              "audited_roots": str(runs), "files_audited": files, "exposures_by_run": by_run,
              "source_baseline_summary_sha256": sha(baseline_summary),
              "old_retained_sources": archived_traces, "eligible_shards": ordered,
              "dataset_sha256": sha(data)}
    if output.exists():
        raise FileExistsError(output)
    save(output, ledger)
    pool = candidate_states(traces)
    print(json.dumps({"ledger_sha256": sha(output), "files": files, "excluded": len(touched),
                      "fresh_existing": {g: len(pool[g].keys()-touched) for g in STRATA},
                      "disjoint_discovery_shards": ordered, "prior_runs_with_ids": len(by_run)}))


def plan_discovery(root, dataset):
    from pyarrow.parquet import ParquetFile

    ledger_file = root/"exposure.json"
    ledger = load(ledger_file)
    if sha(dataset) != ledger["dataset_sha256"]:
        raise ValueError("cannot allocate shards from a different dataset")
    n = ParquetFile(dataset).metadata.num_rows
    permitted = {f"gsm8k:{i}" for i in range(n)} - set(ledger["excluded_discovery_ids"])
    ordered = sorted(permitted,key=lambda p:keyed(p,"scratchpad-new-id"))
    panels = []
    for i in range(0,len(ordered),32):
        path = root/f"discovery-panel-{i//32:02d}.json"
        if path.exists():
            raise FileExistsError(path)
        ids = ordered[i:i+32]
        save(path,{"ids":ids,"exposure_sha256":sha(ledger_file),"dataset_sha256":ledger["dataset_sha256"]})
        panels.append({"path":str(path),"sha256":sha(path),"n":len(ids)})
    output = root/"discovery-plan.json"
    if output.exists():
        raise FileExistsError(output)
    save(output,{"panels":panels,"fresh_ids":len(ordered),"exposure_sha256":sha(ledger_file)})
    print(json.dumps({"discovery_plan_sha256":sha(output),"fresh_ids":len(ordered),
                      "panels":[p["n"] for p in panels]}))


def checked_discovery(sources, ledger, discovery):
    previous_ids = set(ledger["excluded_discovery_ids"])
    seen = set()
    for path, checksum in sources:
        with Path(path).open() as stream:
            header = json.loads(next(stream))
        planned = discovery["panels"][header["shard"]]
        expected = load(Path(planned["path"]))["ids"]
        ids = [row["id"] for row in header["panel"]]
        if (sha(Path(path)) != checksum or header["data_sha"] != ledger["dataset_sha256"]
                or sha(Path(planned["path"])) != planned["sha256"] or header["panel_manifest_sha256"] != planned["sha256"]
                or ids != expected or set(ids) & (previous_ids | seen) or header["prompt_arms"] != ["canonical"]
                or header["cap"] != 8192 or header["decoders"] != ["astra"]):
            raise ValueError("discovery shard violates the pinned disjoint panel")
        seen.update(ids)
    return seen


def choose_panel(baseline, root, dataset, sources, quota=TARGET, allow_short=False):
    ledger_file = root / "exposure.json"
    ledger = load(ledger_file)
    old, old_sources = load_traces(baseline)
    old = {k:v for k,v in old.items() if k not in set(ledger["excluded_intervention_ids"])}
    fresh, seen = {}, set()
    if sources:
        discovery = load(root/"discovery-plan.json")
        if discovery["exposure_sha256"]!=sha(ledger_file) or sha(dataset)!=ledger["dataset_sha256"]:
            raise ValueError("discovery plan lineage mismatch")
        seen = checked_discovery(sources, ledger, discovery)
        save(root/"baseline-summary.json", {"prompts":len(seen),"shards":sources})
        fresh, _ = load_traces(root)
        if set(fresh)!=seen or old.keys() & fresh.keys():
            raise ValueError("incomplete or overlapping discovery rollout panel")
    pool = candidate_states(old | fresh)
    available = set(old) | set(fresh)
    def select(q):
        return take(pool, "scratchpad-v1", set(), available, q)
    selected = select(quota)
    if allow_short and len(selected) != 3*quota:
        for smaller in range(quota-1, 19, -1):
            trial = select(smaller)
            if len(trial) == 3*smaller:
                selected = trial
                break
        used = {x["row"]["id"] for x in selected}
        for group in ("progress","pre_onset","loop"):
            extra = sorted(pool[group].keys()-used,key=lambda p:keyed(p,"scratchpad-fallback:"+group))
            new = extra[:3*quota-len(selected)]
            selected.extend(pool[group][p] for p in new)
            used.update(new)
        selected.sort(key=lambda x:keyed(x["row"]["id"],"scratchpad-fallback-order"))
    counts = dict(Counter(row["stratum"] for row in selected))
    print(json.dumps({"pool_fresh":{k:len(v) for k,v in pool.items()},"selectable":counts,
                      "discovery_ids":len(fresh),"retained_previous_baseline":len(old)}),flush=True)
    if len(selected) != 3*quota and not allow_short:
        return None
    if not selected:
        raise ValueError("no independent native scratchpad states")
    for row in selected:
        row["origin"] = "new_discovery" if row["row"]["id"] in fresh else "unused_canonical_trace"
    report = {"version":"scratchpad-presence-v1","cases":selected,"counts":counts,
              "length_summary":length_summary(selected),"exposure_sha256":sha(ledger_file),
              "discovery_sources":sources,"canonical_sources":old_sources,
              "selection":"balanced strata using only source canonical traces and fixed ID order",
              "dataset_sha256":ledger["dataset_sha256"]}
    path = root/"cohort.json"
    if path.exists():
        raise FileExistsError(path)
    save(path,report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("runs","baseline","output","data","plan-root"):
        p.add_argument("--"+name,type=Path)
    a = p.parse_args()
    if a.plan_root is not None:
        plan_discovery(a.plan_root,a.data)
    else:
        id_ledger(a.runs,a.baseline,a.output,a.data)
