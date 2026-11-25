"""Check token-level parity of the unchanged APPLY control across delayed-query runs."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys


ARMS = ("keep", "reset_conv")
IDENTITY = ("tokenizer_sha256", "checkpoint", "runtime", "sampling")
UNCHANGED = ("seed", "post_seed", "prefix_sha256")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(path.read_text())


def outcomes(root, state, name, mode="apply"):
    mark = state[name]
    path = root / "jobs" / f"{name}.jsonl"
    if mark["stage"] != "completed" or Path(mark["output"]) != path or sha(path) != mark["sha256"]:
        raise ValueError(f"Invalid raw artifact receipt for {name}")
    with path.open() as stream:
        head, *entries, end = map(json.loads, stream)
    if head["split"] != "eval" or head["cap"] != 512 or head["draws"] != 2 or end["kind"] != "complete":
        raise ValueError(f"Wrong replay protocol in {name}")
    pairs = {}
    for row in entries:
        if row["kind"] != "rollout" or row["mode"] != mode or row["arm"] not in ARMS:
            continue
        key = tuple(row[k] for k in ("id", "draw", "variant", "arm"))
        if key in pairs:
            raise ValueError(f"Duplicate apply trace for {name}")
        pairs[key] = row
    if len(pairs) != head["count"] * head["draws"] * 2 * len(ARMS):
        raise ValueError(f"Incomplete {mode} traces for {name}")
    return head, pairs


def audit(root, archive, archive_prefix="DQ"):
    new = load(root / "status.json")["jobs"]
    old = load(archive / "status.json")["jobs"]
    by_arm = {arm: Counter() for arm in ARMS}
    compared = []
    for start in range(0, 64, 8):
        reference, candidate = f"{archive_prefix}-eval-{start:03d}", f"DQL-eval-{start:03d}"
        if new[candidate]["stage"] != "completed":
            continue
        h_old, old_rows = outcomes(archive, old, reference)
        h_new, new_rows = outcomes(root, new, candidate)
        if old_rows.keys() != new_rows.keys() or any(h_old[k] != h_new[k] for k in IDENTITY):
            raise ValueError(f"Comparison used different cases or runtime in {candidate}")
        for key, r_old in old_rows.items():
            r_new = new_rows[key]
            if any(r_old[k] != r_new[k] for k in UNCHANGED):
                raise ValueError(f"Changed APPLY prefix or RNG seed in {candidate}")
            identical_tokens = r_old["tokens"] == r_new["tokens"]
            identical_eos = r_old["eos_token"] == r_new["eos_token"]
            by_arm[key[-1]].update(
                n=1, exact_continuation=int(identical_tokens and identical_eos),
                exact_token_ids=int(identical_tokens), exact_eos_token=int(identical_eos),
            )
        compared.append({"start": start, "original_raw_sha256": old[reference]["sha256"],
                         "matched_raw_sha256": new[candidate]["sha256"]})
    return {"complete": len(compared) == 8, "shards": compared, "apply": by_arm}


def read_missing(root, model):
    config = load(root / "config.json")
    sys.path[:0] = config["runtime"].split(":")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    state = load(root / "status.json")["jobs"]
    panel = {item["id"]: item for item in load(root / "panel.json")["panels"]["eval"]}
    counts = {arm: Counter() for arm in ARMS}
    common = {arm: Counter() for arm in ARMS}
    examples = {arm: {} for arm in ARMS}
    for start in range(0, 64, 8):
        name = f"DQL-eval-{start:03d}"
        path = root / "jobs" / f"{name}-score.json"
        mark = state[name + "-score"]
        if mark["stage"] != "completed" or sha(path) != mark["sha256"]:
            raise ValueError(f"Invalid READ score receipt for {name}")
        score = load(path)
        if score["raw_sha256"] != state[name]["sha256"]:
            raise ValueError(f"READ score/raw receipt mismatch for {name}")
        _, raw = outcomes(root, state, name, mode="read")
        for scored in score["windows"]["512"]:
            if scored["mode"] != "read" or not scored["missing_numeric_eos"]:
                continue
            key = tuple(scored[k] for k in ("id", "draw", "variant", "arm"))
            offset = len(panel[key[0]]["query"]["read"]) + 1
            tokens = raw[key]["tokens"][offset:]
            answer = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            kind = ("empty" if not answer else "unknown" if re.fullmatch(
                r"""[\s"'*]*UNKNOWN[\s."'*!]*""", answer, re.I) else
                "contains_digit" if re.search(r"[0-9]", answer) else "other_text")
            counts[key[-1]][kind] += 1
            common[key[-1]][answer[:120]] += 1
            raw_text = tokenizer.decode(tokens, skip_special_tokens=False).strip()
            shape = re.sub(r"[0-9]+", "#", raw_text[:120])
            if len(examples[key[-1]]) < 8 and shape not in examples[key[-1]]:
                examples[key[-1]][shape] = {
                    "raw": raw_text[:120], "clean": answer[:120],
                    "expected": panel[key[0]]["targets"]["read"][key[2]],
                }
    return {arm: {"counts": counts[arm], "common_outputs": common[arm].most_common(5),
                  "examples": list(examples[arm].values())} for arm in ARMS}


def position_effects(root, before):
    sys.path[:0] = load(root / "config.json")["runtime"].split(":")
    from repetition_canonical import paired_interval

    results = {}
    for phase, run in (("before", before), ("after", root)):
        state = load(run / "status.json")["jobs"]
        panel_sha = sha(run / "panel.json")
        windows = defaultdict(dict)
        for start in range(0, 64, 8):
            name = f"DQL-eval-{start:03d}"
            path = run / "jobs" / f"{name}-score.json"
            mark = state[name + "-score"]
            if mark["stage"] != "completed" or sha(path) != mark["sha256"]:
                raise ValueError(f"Unverified position score for {phase} {name}")
            score = load(path)
            if (score["header"]["panel_sha256"] != panel_sha or score["raw_sha256"] != state[name]["sha256"]
                    or score["answer_boundary"] != "assistant-turn-v1"):
                raise ValueError("Before/after score provenance differs")
            for cap, entries in score["windows"].items():
                for row in entries:
                    if row["mode"] != "read":
                        continue
                    key = tuple(row[k] for k in ("id", "draw", "variant", "arm"))
                    if key in windows[cap]:
                        raise ValueError("Duplicate position score cell")
                    windows[cap][key] = int(row["correct_eos"])
        results[phase] = windows
    contrasts = {}
    for cap, after_rows in results["after"].items():
        before_rows = results["before"][cap]
        if len(after_rows) != 512 or after_rows.keys() != before_rows.keys():
            raise ValueError("Expected identical sets of 512 before/after READ cells")
        by_arm = {arm: defaultdict(list) for arm in ARMS}
        for key in sorted(after_rows):
            by_arm[key[-1]][key[0]].append(after_rows[key] - before_rows[key])
        contrasts[cap] = {arm: paired_interval(dict(values)) for arm, values in by_arm.items()}
    return contrasts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--archive-prefix", choices=("DQ", "DQL"), default="DQ")
    p.add_argument("--model", type=Path, help="Decode missing-numeric READ outputs after all scores finish")
    p.add_argument("--position-effects", action="store_true")
    args = p.parse_args()
    result = audit(args.root, args.archive, args.archive_prefix)
    if args.model is not None:
        result["read_missing_numeric_eos"] = read_missing(args.root, args.model)
    if args.position_effects:
        result["after_minus_before_read_by_arm_pp"] = position_effects(args.root, args.archive)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
