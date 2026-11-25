"""Describe READ termination outcomes by record family and queried slot."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


METRICS = ("correct_eos", "correct_without_eos", "cap", "opposite_eos",
           "other_numeric_eos", "missing_numeric_eos")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(path.read_text())


def results(orig, tail, before, after):
    runs = (orig, tail, before, after)
    status, panels = {}, {}
    for run in runs:
        source = run / "status.json"
        if load(run / "release.json")["status_sha256"] != sha(source):
            raise ValueError(f"Release receipt differs for {run.name}")
        status[run] = load(source)["jobs"]
        panels[run] = sha(run / "panel.json")
    panel = load(orig / "panel.json")
    cases = {row["id"]: (row["family"], row["slot"]) for row in panel["panels"]["eval"]}
    common = [(start, 8) for start in range(0, 64, 8)]
    placements = {"12+0": [(before, s, n) for s, n in common],
                  "6+6": [(orig, s, n) for s, n in common[:-1]] + [(tail, 56, 4), (tail, 60, 4)],
                  "0+12": [(after, s, n) for s, n in common]}
    result = {}
    for name, plan in placements.items():
        cells = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
        seen = set()
        for root, start, count in plan:
            job = f"DQL-eval-{start:03d}"
            raw_mark, mark = status[root][job], status[root][f"{job}-score"]
            raw = root / "jobs" / f"{job}.jsonl"
            score = root / "jobs" / f"{job}-score.json"
            if any(entry["stage"] != "completed" or sha(path) != entry["sha256"]
                   for entry, path in ((raw_mark, raw), (mark, score))):
                raise ValueError(f"Corrupt raw/score receipt for {root.name}/{job}")
            data = load(score)
            h = data["header"]
            if (h["start"] != start or h["count"] != count or h["panel_sha256"] != panels[root]
                    or data["raw_sha256"] != raw_mark["sha256"] or data["answer_boundary"] != "assistant-turn-v1"):
                raise ValueError(f"Unexpected score protocol for {root.name}/{job}")
            for cap, rows in data["windows"].items():
                for row in rows:
                    if row["mode"] != "read":
                        continue
                    ident, arm = row["id"], row["arm"]
                    key = cap, ident, row["draw"], row["variant"], arm
                    if key in seen:
                        raise ValueError("Duplicate READ score row")
                    seen.add(key)
                    case = panel["panels"]["eval"][start:start+count]
                    if ident not in {item["id"] for item in case}:
                        raise ValueError("Score is for a different question shard")
                    family, slot = cases[ident]
                    for group in ("all", family, f"slot{slot}"):
                        cells[cap][group][arm].update(
                            n=1, **{metric: int(row[metric]) for metric in METRICS})
        if len(seen) != 2 * 64 * 2 * 2 * 2:
            raise ValueError(f"Incomplete READ score matrix for {name}")
        result[name] = cells
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("orig", "tail", "before", "after"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(results(a.orig, a.tail, a.before, a.after)))


if __name__ == "__main__":
    main()
