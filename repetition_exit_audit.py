"""Read-only reanalysis of the completed exit rollout traces."""
import argparse
from collections import defaultdict
from pathlib import Path
import json

from repetition_baseline import assess, sha
from repetition_canonical_queue import paired_interval
from repetition_mediation_queue import load, save

ARMS = ("native", "blank", "left", "fixed_pulse", "periodic_pulse")


def read(old, tokenizer):
    cohort = load(old / "cohort.json")
    goals = {r["row"]["id"]: r["row"] for r in cohort["cases"]}
    rows = defaultdict(dict)
    sources = []
    for lane in ("collection", "mechanism"):
        receipt = load(old / lane / "heldout-done.json")
        for name, digest in receipt["outputs"]:
            path = Path(name)
            if sha(path) != digest:
                raise ValueError("historical shard checksum mismatch")
            sources.append([name, digest])
            with path.open() as f:
                manifest = json.loads(next(f))
                meta = {c["id"]: c for c in manifest["cases"]}
                complete = False
                for line in f:
                    r = json.loads(line)
                    if r["kind"] == "complete":
                        complete = True
                        continue
                    arm = r["arm"]
                    if r["id"] not in goals or arm not in ARMS:
                        continue
                    arm_rows = rows[meta[r["id"]]["stratum"], arm]
                    key = r["id"], r["draw"]
                    if key in arm_rows:
                        raise ValueError("duplicate historical result")
                    body = tokenizer.decode(r["tokens"], skip_special_tokens=False)
                    strict = assess(goals[r["id"]], body, [], r["eos"], manifest["cap"], True)
                    if strict["correct"] != r["correct"]:
                        raise ValueError("historical scorer parity failed")
                    repaired = body.replace(chr(8) + "oxed{", chr(92) + "boxed{")
                    after = assess(goals[r["id"]], repaired, [], r["eos"], manifest["cap"], True)
                    arm_rows[key] = r | {"rescored": after["correct"] and r["eos"] and not r["terminal_cycle"]}
                if not complete:
                    raise ValueError("historical shard is incomplete")
    return rows, sources


def compare(a, b, included):
    grouped = defaultdict(list)
    for key in sorted(included):
        grouped[key[0]].append(int(a[key]["rescored"]) - int(b[key]["rescored"]))
    return {"rollouts": len(included), "a": sum(bool(a[k]["rescored"]) for k in included),
            "b": sum(bool(b[k]["rescored"]) for k in included), "paired": paired_interval(grouped)}


def analyze(old, tokenizer):
    rows, sources = read(old, tokenizer)
    outputs = {}
    for group, total in (("loop", 768), ("clean", 384)):
        cells = {arm: rows[group, arm] for arm in ARMS}
        keys = cells["native"].keys()
        if len(keys) != total or any(c.keys() != keys for c in cells.values()):
            raise ValueError("incomplete historical pair matrix")
        periodic = cells["periodic_pulse"]
        trigger_groups = {
            "zero": {k for k in keys if periodic[k]["trigger_step"] == 0},
            "late": {k for k in keys if periodic[k]["trigger_step"] > 0},
            "never": {k for k in keys if periodic[k]["trigger_step"] < 0},
        }
        subsets = {k: compare(cells["fixed_pulse"], periodic, matched)
                   for k, matched in trigger_groups.items()}
        subsets["zero"]["token_and_eos_exact"] = sum(
            cells["fixed_pulse"][k]["tokens"] == periodic[k]["tokens"]
            and cells["fixed_pulse"][k]["eos"] == periodic[k]["eos"] for k in trigger_groups["zero"])
        outputs[group] = {
            "native_minus_blank": compare(cells["native"], cells["blank"], keys),
            "left_minus_native": compare(cells["left"], cells["native"], keys),
            "fixed_minus_periodic_by_observed_trigger": subsets,
        }
    return {"historical_shards": sources, "normalization": "only U+0008 + oxed marker",
            "trigger_analysis": "post hoc, strata are defined by the periodic arm's sampled path",
            "strata": outputs}


def main():
    from tokenizers import Tokenizer

    p = argparse.ArgumentParser(description=__doc__)
    for k in ("old", "tokenizer", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    args = p.parse_args()
    result = analyze(args.old, Tokenizer.from_file(str(args.tokenizer)))
    if args.output.exists():
        raise FileExistsError(args.output)
    save(args.output, result | {"tokenizer_sha256": sha(args.tokenizer), "worker_sha256": sha(Path(__file__))})
    print(json.dumps(result["strata"]))


if __name__ == "__main__":
    main()
