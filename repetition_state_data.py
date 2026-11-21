"""Freeze prompt-disjoint native states from checksum-verified canonical traces."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import statistics

from repetition_mediation_queue import digest, load, save
from repetition_replay import ASTRA_REVISION, terminal_cycle

THINK_END = 248069
STRATA = ("loop", "pre_onset", "progress")


def keyed(prompt_id, purpose):
    return hashlib.sha256(f"state-value-v1:{purpose}:{prompt_id}".encode()).digest()


def load_traces(parent):
    summary = load(parent / "baseline-summary.json")
    output = {}
    for file, actual in summary["shards"]:
        path = Path(file)
        if digest(path) != actual:
            raise ValueError(f"canonical trace checksum failed: {path}")
        with path.open() as s:
            head = json.loads(next(s))
            if head["model_revision"] != ASTRA_REVISION or head["cap"] != 8192:
                raise ValueError("unexpected canonical trace protocol")
            panel = {r["id"]: r for r in head["panel"]}
            done = False
            for line in s:
                r = json.loads(line)
                if r["record"] == "completed":
                    done = True
                    continue
                if r["prompt_arm"] != "canonical":
                    continue
                key = r["id"]
                row = panel[key]
                if key in output or row["astra_prompt_arm"] != "canonical" or chr(8) in row["prompt"]:
                    raise ValueError("invalid or reused canonical prompt")
                eos = r["outcomes"]["8192"]["eos"]
                ids = r["token_ids"][:-1] if eos else r["token_ids"]
                output[key] = (row, r["render_sha256"], ids, eos)
            if not done:
                raise ValueError("incomplete baseline shard")
    if len(output) != summary["prompts"]:
        raise ValueError("canonical panel coverage mismatch")
    return output, summary["shards"]


def candidate_states(traces):
    output = {name: {} for name in STRATA}
    for key, (row, rendered_sha, ids, eos) in traces.items():
        bound = ids.index(THINK_END) if THINK_END in ids else len(ids)
        common = {"row": row, "render_sha256": rendered_sha, "source_eos": eos}
        cyc = terminal_cycle(ids)
        if cyc is not None:
            period, end = cyc
            onset = end - 5 * period
            if 0 < end <= bound:
                output["loop"][key] = {**common, "stratum": "loop", "completion_prefix": ids[:end],
                                       "onset": onset, "period": period}
                cut = onset - 48
                if cut >= 128 and not terminal_cycle(ids[:cut]):
                    output["pre_onset"][key] = {**common, "stratum": "pre_onset", "completion_prefix": ids[:cut],
                                                "onset": onset, "period": period}
        else:
            levels = [cut for cut in (256, 768, 1536, 3072, 4608) if cut + 128 <= bound and not terminal_cycle(ids[:cut])]
            if levels:
                cut = levels[keyed(key, "early")[0] % len(levels)]
                output["progress"][key] = {**common, "stratum": "progress", "completion_prefix": ids[:cut],
                                           "onset": None, "period": None}
    return output


def take(pool, split, used, allowed, quota):
    # Pre-onset is a subset of loop IDs. Reserve it first.
    result = []
    for name in ("pre_onset", "loop", "progress"):
        choices = sorted((k for k in pool[name] if k in allowed and k not in used),
                         key=lambda k: keyed(k, f"{split}:{name}"))
        selected = choices[:quota]
        result.extend(pool[name][key] for key in selected)
        used.update(selected)
    return sorted(result, key=lambda r: keyed(r["row"]["id"], split + ":order"))


def length_summary(rows):
    by_group = {}
    for name in sorted({x["stratum"] for x in rows}):
        values = sorted(len(x["completion_prefix"]) for x in rows if x["stratum"] == name)
        bins = Counter("short" if n < 1024 else "middle" if n < 3072 else "long" for n in values)
        by_group[name] = {"n": len(values), "range": [values[0], values[-1]],
                          "median": statistics.median(values), "bins": dict(bins)}
    return by_group


def prepare(parent, output):
    traces, sources = load_traces(parent)
    old = load(parent / "cohort.json")["cases"]
    old_ids = {x["row"]["id"] for x in old}
    with (parent / "canary.jsonl").open() as s:
        old_ids.update(x["id"] for x in json.loads(next(s))["panel"])
    pool = candidate_states(traces)
    used = set()
    test = take(pool, "eval", used, traces.keys() - old_ids, 10)
    dev = take(pool, "dev", used, traces.keys(), 8)
    clean = sorted((x for x in old if x["stratum"] == "clean" and x["row"]["id"] not in used),
                   key=lambda x: keyed(x["row"]["id"], "old-eos"))
    dev.extend(x | {"stratum": "old_clean_eos"} for x in clean[:6])
    if not dev or not test:
        raise ValueError("no eligible prompt-disjoint split")
    result = {"version": "state-value-v1", "dev": dev, "eval": test, "source_shards": sources,
              "old_exclusion": len(old_ids), "pool_counts": {k: len(v) for k, v in pool.items()},
              "length_summary": {"dev": length_summary(dev), "eval": length_summary(test)}}
    if output.exists():
        raise FileExistsError(output)
    save(output, result)
    print(json.dumps({k: result[k] for k in ("version", "old_exclusion", "pool_counts", "length_summary")}))


def copies(split, repetitions=None):
    words = [a + b + c for a in "mnpbrst" for b in "aeiou" for c in "lnmrkst"]
    random.Random(314597).shuffle(words)
    words = words[:24] if split == "dev" else words[24:40]
    items = []
    for i, word in enumerate(words):
        amount = i % 4 + 1 if split == "dev" else repetitions
        row = {"id": f"state-copy:{split}:{i}", "task": "copy",
               "prompt": f"Your answer must be exactly the word {word} written {amount} time(s), "
                         "separated by one space. Output only the requested text.",
               "gold": " ".join([word] * amount)}
        items.append({"row": row, "completion_prefix": [], "stratum": "copy", "repetitions": amount})
    return items


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    prepare(args.parent, args.output)


if __name__ == "__main__":
    main()
