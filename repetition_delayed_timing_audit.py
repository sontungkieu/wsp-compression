"""Audit reset-offset replay parity against archived READ traces and describe subgroups."""
import argparse
from collections import Counter, defaultdict
import json
from itertools import zip_longest
from pathlib import Path
import random
from statistics import fmean

from repetition_delayed_length_audit import IDENTITY, UNCHANGED, load, outcomes, sha
from repetition_replay import terminal_cycle


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


def end_state(row):
    return next(key for key in ("correct_eos", "cap", "opposite_eos", "other_numeric_eos", "missing_numeric_eos")
                if row[key])


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
    end_transitions = defaultdict(lambda: defaultdict(Counter))
    terminal_loops_at_cap = defaultdict(Counter)
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
            score_by_key = {(r["id"], r["draw"], r["variant"], r["arm"]): r for r in scores}
            for row in scores:
                family, slot = cases[row["id"]]
                for group in ("all", family, f"slot{slot}"):
                    groups[cap][group][row["arm"]].update(
                        n=1, **{key: int(row[key]) for key in METRICS})
                arm = row["arm"]
                key = row["id"], row["draw"], row["variant"]
                width = widths[row["id"]]
                keep_key, reset_key = (*key, "keep"), (*key, arm)
                path = answer_path(rows[reset_key], width)
                if row["cap"]:
                    terminal_loops_at_cap[cap][arm] += int(terminal_cycle(path[:int(cap)]) is not None)
                if arm == "keep":
                    continue
                before = score_by_key[keep_key]
                end_transitions[cap][arm][f"{end_state(before)}_to_{end_state(row)}"] += 1
                lag = first_divergence(answer_path(rows[keep_key], width), path, int(cap))
                label = {
                    (False, False): "both_fail", (True, True): "both_correct",
                    (False, True): "gain", (True, False): "loss",
                }[before["correct_eos"], row["correct_eos"]]
                divergences[cap][arm]["all"][lag] += 1
                divergences[cap][arm][label][lag] += 1
    return {"complete": len(shards) == 8, "prompts": sum(x["count"] for x in shards.values()),
            "panel_sha256": pin, "shards": shards, "archived_read_parity": matches, "subgroups": groups,
            "first_answer_divergence": divergences, "end_transitions": end_transitions,
            "terminal_loops_at_cap": terminal_loops_at_cap}


def timing_scores(run):
    marks = state(run, require_release=True)
    panel = run / "panel.json"
    pin = sha(panel)
    cases = load(panel)["panels"]["eval"]
    identity = None
    scores = defaultdict(dict)
    for start in range(0, 64, 8):
        header, _, report = verified_timing_shard(run, marks, f"DQT-eval-{start:03d}", pin)
        current = {k: header[k] for k in (*IDENTITY, "worker_sha256_timing", "protocol", "draws", "cap")}
        if identity is not None and current != identity:
            raise ValueError("mixed fixed-query worker contracts")
        identity = current
        for cap, rows in report["windows"].items():
            if report["header"]["start"] != start or {r["id"] for r in rows} != {c["id"] for c in cases[start:start+8]}:
                raise ValueError("timing score does not cover the expected frozen range")
            for row in rows:
                key = row["id"], row["draw"], row["variant"], row["arm"]
                if key in scores[cap]:
                    raise ValueError("duplicate fixed-query score")
                scores[cap][key] = int(row["correct_eos"])
    return identity, cases, scores


def compare_replication(original, replication):
    specification, left_cases, left = timing_scores(original)
    other, right_cases, right = timing_scores(replication)
    if specification != other or {c["id"] for c in left_cases} & {c["id"] for c in right_cases}:
        raise ValueError("replication needs different cases under the same worker contract")
    def token_pairs(cases):
        return {tuple(sorted(tuple(ids) for ids in c["retained"])) for c in cases}

    if token_pairs(left_cases) & token_pairs(right_cases):
        raise ValueError("replication repeats archived records")
    results = {}
    for cap in ("256", "512"):
        results[cap] = {}
        for a, b in (("reset_7", "keep"), ("reset_7", "reset_0"), ("reset_17", "keep")):
            def by_question(cases, cells):
                return [fmean(cells[c["id"], draw, variant, a] - cells[c["id"], draw, variant, b]
                              for draw in (0, 1) for variant in (0, 1)) for c in cases]

            x, y = by_question(left_cases, left[cap]), by_question(right_cases, right[cap])
            rng = random.Random(87131)
            pair_draws = [(fmean(rng.choices(x, k=len(x))), fmean(rng.choices(y, k=len(y)))) for _ in range(3000)]

            def ci(point, values):
                ordered = sorted(values)
                return {"delta_pp": round(100 * point, 2),
                        "ci95_pp": [round(100 * ordered[i], 2) for i in (75, 2924)]}

            results[cap][f"{a}_minus_{b}"] = {
                "original": ci(fmean(x), [u for u, _ in pair_draws]),
                "replication": ci(fmean(y), [v for _, v in pair_draws]),
                "pooled_equal_panel_weight": ci(fmean(x + y), [fmean((u, v)) for u, v in pair_draws]),
                "replication_minus_original": ci(fmean(y) - fmean(x), [v - u for u, v in pair_draws]),
            }
    return {"original_prompts": len(left_cases), "replication_prompts": len(right_cases), "windows": results}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "original"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--tail", type=Path)
    p.add_argument("--replication", action="store_true")
    a = p.parse_args()
    if a.replication:
        print(json.dumps(compare_replication(a.original, a.root)))
    else:
        if a.tail is None:
            p.error("archived-trace audit needs --tail")
        print(json.dumps(audit(a.root, a.original, a.tail)))


if __name__ == "__main__":
    main()
