"""Rescore archived pairs at the first assistant turn delimiter and retain EOS accounting."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

from repetition_delayed_length_audit import ARMS, load, outcomes, sha


MODES = ("read", "apply")
WINDOWS = (256, 512)


def rescore(root, archive, model):
    from decimal import Decimal
    from repetition_baseline import math_answer
    from repetition_delayed_query import correct_eos_contrasts, interval_by_id
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    marker = tok.encode("<|im_end|>", add_special_tokens=False)
    if len(marker) != 1 or marker[0] not in tok.all_special_ids:
        raise ValueError("Expected a single assistant turn delimiter")
    [marker] = marker
    [think_end] = tok.encode("</think>", add_special_tokens=False)
    tokenizer_sha = sha(model / "tokenizer.json")
    scores = {}
    pins = {}
    for label, run, prefix in (("original", archive, "DQ"), ("matched", root, "DQL")):
        state = load(run / "status.json")["jobs"]
        panel_path = run / "panel.json"
        panel_sha = sha(panel_path)
        cases = {item["id"]: item for item in load(panel_path)["panels"]["eval"]}
        rows = {cap: [] for cap in WINDOWS}
        provenance = []
        for start in range(0, 64, 8):
            name = f"{prefix}-eval-{start:03d}"
            score = run / "jobs" / f"{name}-score.json"
            stat = state[name + "-score"]
            if stat["stage"] != "completed" or sha(score) != stat["sha256"]:
                raise ValueError(f"Unverified score receipt {name}")
            old = load(score)
            raw_mark = state[name]
            if old["raw_sha256"] != raw_mark["sha256"] or old["header"]["panel_sha256"] != panel_sha:
                raise ValueError(f"Score and raw provenance differ for {name}")
            provenance.append([start, raw_mark["sha256"], stat["sha256"]])
            for mode in MODES:
                header, decoded = outcomes(run, state, name, mode)
                if header["panel_sha256"] != panel_sha or header["tokenizer_sha256"] != tokenizer_sha:
                    raise ValueError(f"Wrong panel or tokenizer for {name}")
                for (ident, draw, variant, arm), trajectory in decoded.items():
                    item = cases[ident]
                    forced = item["query"][mode] + [think_end]
                    tokens = trajectory["tokens"]
                    stop = trajectory["eos_token"]
                    if tokens[:len(forced)] != forced or stop is not None and stop not in header["eos_ids"]:
                        raise ValueError(f"Readout boundary differs for {name}")
                    targets = item["targets"][mode]
                    gold = Decimal(targets[variant])
                    post = tokens[len(forced):]
                    for cap in WINDOWS:
                        eos = stop is not None and len(post) < cap
                        limited = post[:cap]
                        turn_seen = marker in limited
                        answer = limited[:limited.index(marker)] if turn_seen else limited
                        decoded_answer = tok.decode(answer, skip_special_tokens=False)
                        parsed = math_answer({"gold": str(gold)}, decoded_answer, pending_thought=False)["parsed_answer"]
                        value = Decimal(parsed) if parsed is not None else None
                        correct, opposite = value == gold, value == Decimal(targets[1-variant])
                        rows[cap].append({
                            "id": ident, "draw": draw, "variant": variant, "mode": mode, "arm": arm,
                            "correct_eos": eos and correct, "opposite_eos": eos and opposite,
                            "other_numeric_eos": eos and value is not None and not (correct or opposite),
                            "missing_numeric_eos": eos and value is None, "cap": not eos,
                            "correct_without_eos": not eos and correct, "turn_seen": turn_seen})
        scores[label] = rows
        pins[label] = {"panel_sha256": panel_sha, "shards": provenance}

    metrics = ("correct_eos", "opposite_eos", "other_numeric_eos", "missing_numeric_eos",
               "cap", "correct_without_eos", "turn_seen")
    windows = {}
    for cap in WINDOWS:
        results = {}
        for label, rows in scores.items():
            outcomes_at_cap = rows[cap]
            keyed, effects = correct_eos_contrasts(outcomes_at_cap)
            if len(keyed) != 64 * 2 * 2 * 2 * 2:
                raise ValueError("Expected the complete paired holdout")
            counts = {}
            for mode in MODES:
                counts[mode] = {}
                for arm in ARMS:
                    entries = [row for row in outcomes_at_cap if row["arm"] == arm and row["mode"] == mode]
                    counts[mode][arm] = {"n": len(entries)} | Counter(
                        {name: sum(row[name] for row in entries) for name in metrics})
            results[label] = {"counts": counts, "effects": effects, "keys": keyed}
        matched, original = results["matched"], results["original"]
        m, o = matched["effects"]["read_minus_apply"], original["effects"]["read_minus_apply"]
        if m.keys() != o.keys():
            raise ValueError("Matched vs original did not use the same cases")
        windows[cap] = {
            label: {"counts": block["counts"], "contrasts_pp":
                    {name: interval_by_id(effect) for name, effect in block["effects"].items()}}
            for label, block in results.items()}
        windows[cap]["matched_minus_original_interaction"] = interval_by_id({k: m[k] - o[k] for k in m})
        windows[cap]["unchanged_apply_correct_eos_agreement"] = sum(
            matched["keys"][k] == original["keys"][k] for k in matched["keys"] if k[3] == "apply")
    return {"protocol": "first-assistant-turn-delimiter-v1", "turn_token": marker,
            "worker_sha256": sha(Path(__file__)), "sources": pins, "windows": windows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("root", "archive", "model", "output"):
        p.add_argument("--" + arg, type=Path, required=True)
    args = p.parse_args()
    if args.output.parent != args.root or args.output.exists():
        raise ValueError("Expected a fresh report under the matched run root")
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    sys.path[:0] = load(args.root / "config.json")["runtime"].split(":")
    report = rescore(args.root, args.archive, args.model)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
    print(json.dumps({"report_sha256": sha(args.output), "turn_token": report["turn_token"],
                      "windows": report["windows"]}))


if __name__ == "__main__":
    main()
