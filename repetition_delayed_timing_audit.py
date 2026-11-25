"""Audit reset-offset replay parity against archived READ traces and describe subgroups."""
import argparse
from collections import Counter, defaultdict
import json
from itertools import zip_longest
from pathlib import Path

from repetition_delayed_length_audit import IDENTITY, UNCHANGED, load, outcomes, sha


ARMS = ("keep", "reset_0", "reset_7", "reset_17")
METRICS = ("correct_eos", "correct_without_eos", "cap", "opposite_eos",
           "other_numeric_eos", "missing_numeric_eos")
BANDS = ((1, "token1"), (4, "token2-4"), (16, "token5-16"), (64, "token17-64"), (512, "token65-512"))


def answer_path(row, query_length):
    visible = row["tokens"][query_length + 1:]
    return visible if row["eos_token"] is None else visible + [row["eos_token"]]


def first_divergence(before, after, cap):
    for index, (left, right) in enumerate(zip_longest(before[:cap], after[:cap]), start=1):
        if left != right:
            return next(label for bound, label in BANDS if index <= bound)
    return "identical"


def state(run, require_release=False):
    path = run / "status.json"
    release = run / "release.json"
    if require_release or release.is_file():
        if load(release)["status_sha256"] != sha(path):
            raise ValueError(f"Release receipt does not match status for {run.name}")
    return load(path)["jobs"]


def verified_timing_shard(root, status, name, panel_sha):
    raw_path, score_path = (root / "jobs" / f"{name}{suffix}"
                            for suffix in (".jsonl", "-score.json"))
    gpu, score = status[name], status[name + "-score"]
    for jobname, mark, path in ((name, gpu, raw_path), (name + "-score", score, score_path)):
        if (mark["stage"] != "completed" or sha(path) != mark["sha256"]
                or Path(mark["output"]) != path or load(root / "jobs" / f"{jobname}.receipt.json") != mark):
            raise ValueError(f"Unverified timing artifact {jobname}")
    data = load(score_path)
    with raw_path.open() as stream:
        header, *raw, end = map(json.loads, stream)
    if (data["header"] != header or header["panel_sha256"] != panel_sha or tuple(header["arms"]) != ARMS
            or data["raw_sha256"] != gpu["sha256"] or end["kind"] != "complete"
            or set(data["windows"]) != {"256", "512"}):
        raise ValueError(f"Timing provenance differs for {name}")
    rows = {(r["id"], r["draw"], r["variant"], r["arm"]): r for r in raw if r["kind"] == "rollout"}
    if len(rows) != header["count"] * 2 * 2 * len(ARMS):
        raise ValueError(f"Incomplete timing matrix for {name}")
    return header, rows, data


def audit(root, original, tail):
    current = state(root)
    sources = {p: state(p, require_release=True) for p in (original, tail)}
    panel_path = original / "panel.json"
    pin = sha(panel_path)
    panel = load(panel_path)["panels"]["eval"]
    cases = {c["id"]: (c["family"], c["slot"]) for c in panel}
    widths = {c["id"]: len(c["query"]["read"]) for c in panel}
    matches = {arm: Counter() for arm in ("keep", "reset_0")}
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
    divergences = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
    shards = {}
    for start in range(0, 64, 8):
        name = f"DQT-eval-{start:03d}"
        if current.get(name + "-score", {}).get("stage") != "completed":
            continue
        header, rows, data = verified_timing_shard(root, current, name, pin)
        reference = {}
        requests = [(original, start)] if start < 56 else [(tail, 56), (tail, 60)]
        for path, offset in requests:
            old_name = f"DQL-eval-{offset:03d}"
            old_header, old = outcomes(path, sources[path], old_name, mode="read")
            if old_header["panel_sha256"] != pin or any(old_header[k] != header[k] for k in IDENTITY):
                raise ValueError(f"Archived runtime, panel or checkpoint mismatch for {old_name}")
            if reference.keys() & old.keys():
                raise ValueError("Archived comparison rows overlap")
            reference.update(old)
        seen = set()
        for (ident, draw, variant, arm), row in rows.items():
            if arm not in matches:
                continue
            old = reference[ident, draw, variant, {"keep": "keep", "reset_0": "reset_conv"}[arm]]
            if any(row[k] != old[k] for k in UNCHANGED):
                raise ValueError("Archived timing prefix or seed changed")
            equal = row["tokens"] == old["tokens"] and row["eos_token"] == old["eos_token"]
            matches[arm].update(n=1, exact_continuation=int(equal))
            seen.add(ident)
        if seen != {c["id"] for c in panel[start:start + 8]}:
            raise ValueError(f"Timing shard {name} differs from its locked question slice")
        shards[name] = {"raw_sha256": current[name]["sha256"],
                        "score_sha256": current[name + "-score"]["sha256"], "count": len(seen)}
        for cap, scores in data["windows"].items():
            correct = {(r["id"], r["draw"], r["variant"], r["arm"]): bool(r["correct_eos"]) for r in scores}
            for row in scores:
                family, slot = cases[row["id"]]
                for group in ("all", family, f"slot{slot}"):
                    groups[cap][group][row["arm"]].update(
                        n=1, **{key: int(row[key]) for key in METRICS})
                arm = row["arm"]
                if arm == "keep":
                    continue
                key = row["id"], row["draw"], row["variant"]
                width = widths[row["id"]]
                keep_key, reset_key = (*key, "keep"), (*key, arm)
                lag = first_divergence(answer_path(rows[keep_key], width), answer_path(rows[reset_key], width),
                                       int(cap))
                label = {
                    (False, False): "both_fail", (True, True): "both_correct",
                    (False, True): "gain", (True, False): "loss",
                }[correct[keep_key], correct[reset_key]]
                divergences[cap][arm]["all"][lag] += 1
                divergences[cap][arm][label][lag] += 1
    return {"complete": len(shards) == 8, "prompts": sum(x["count"] for x in shards.values()),
            "panel_sha256": pin, "shards": shards, "archived_read_parity": matches, "subgroups": groups,
            "first_answer_divergence": divergences}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "original", "tail"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(audit(a.root, a.original, a.tail)))


if __name__ == "__main__":
    main()
