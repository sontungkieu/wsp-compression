"""Pair first-answer cache recovery across the two pulse offsets."""

import argparse
import json
from pathlib import Path
from statistics import mean

from repetition_canonical import paired_interval
from repetition_delayed_value_audit import checked_rows, load, sha


def compare(panel_path, early, late):
    panel = load(panel_path)["panels"]["eval"]
    summary = late / "jobs" / "DQP7-eval-summary.json"
    receipt = load(late / "status.json")["jobs"]["DQP7-eval-summary"]
    if receipt["stage"] != "completed" or sha(summary) != receipt["sha256"]:
        raise ValueError("Late-pulse summary differs from its completed receipt")
    report = load(summary)
    if report["pulse_offset"] != 7 or report["anchors"] != [17, 23]:
        raise ValueError("Expected the READ-boundary pulse and its two restoration offsets")
    panel = panel[:report["prompts"]]
    before, before_pins, before_identity = checked_rows(early, "DQR", panel, "arms")
    after, after_pins, after_identity = checked_rows(late, "DQP7", panel, "arms")
    if before_identity != after_identity:
        raise ValueError("Pulse comparison requires the same checkpoint and runtime")

    def metric(group, arm):
        return {pid: row[arm]["tv"] for pid, row in group.items()}

    def difference(left, right):
        return {pid: [x - y for x, y in zip(values, right[pid], strict=True)] for pid, values in left.items()}

    pulses = {"t0": metric(before, "pulse_0"), "t7": metric(after, "pulse_7")}
    recoveries = {}
    for label, cells in (("t0", before), ("t7", after)):
        conv = metric(cells, "17_conv")
        recoveries[label] = {cache: difference(conv, metric(cells, f"17_conv_{cache}"))
                             for cache in ("matrix", "kv")}
    return {
        "prompts": len(panel), "panel_sha256": sha(panel_path), "early_shards": before_pins,
        "late_shards": after_pins,
        "mean_pulse_tv": {label: mean(mean(pair) for pair in cases.values()) for label, cases in pulses.items()},
        "pulse_t7_minus_t0_tv_pp": paired_interval(difference(pulses["t7"], pulses["t0"])),
        "restoration_at_t17_reduction_tv_pp": {
            label: {cache: paired_interval(groups) for cache, groups in by_cache.items()}
            for label, by_cache in recoveries.items()},
        "t7_minus_t0_restoration_at_t17_reduction_tv_pp": {
            cache: paired_interval(difference(recoveries["t7"][cache], recoveries["t0"][cache]))
            for cache in recoveries["t7"]},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("panel", "early", "late"):
        p.add_argument("--" + field, required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(compare(a.panel, a.early, a.late)))


if __name__ == "__main__":
    main()
