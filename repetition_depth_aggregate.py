"""Report the follow-on campaign from independently verified completed shards."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import statistics
import time

from repetition_baseline import sha
from repetition_followup_aggregate import contrast, replay_results, submitted_scores, summarize, tally
from repetition_mediation_queue import load, save


def borrowed_reports(root, status):
    motif, depth = [], []
    for name, mark in status["jobs"].items():
        if mark["stage"] != "completed" or "canary" in name:
            continue
        if name.startswith("MOTIF-") and name.endswith("-review"):
            parent = name.removesuffix("-review")
            group = motif
        elif name.startswith("ND-") and name.endswith("-depthreview"):
            parent = name.removesuffix("-depthreview")
            group = depth
        else:
            continue
        path = root / "jobs" / (name + ".json")
        if str(path) != mark["output"] or sha(path) != mark["sha256"]:
            raise ValueError(f"report receipt differs from bytes for {name}")
        record = load(path)
        if record["raw_sha256"] != status["jobs"][parent]["sha256"]:
            raise ValueError(f"secondary scorer receipt does not match {parent}")
        group.append(record)
    return motif, depth


def motif_summary(shards):
    staged = defaultdict(list)
    for piece in shards:
        h = piece["header"]
        staged[(h["dataset"], h["split"], h["cap"], h["source_draw"])].append(piece)
    report = {}
    for (dataset, split, cap, draw), pieces in sorted(staged.items()):
        events = [r for piece in pieces for r in piece["events"]]
        windows = {}
        for budget in pieces[0]["windows"]:
            arms = defaultdict(dict)
            for piece in pieces:
                for arm, cells in piece["windows"][budget]["cells"].items():
                    if arms[arm].keys() & cells.keys():
                        raise ValueError("duplicate motif prompts within a registered panel")
                    arms[arm].update(cells)
            counts = {}
            for arm, cells in arms.items():
                rows = [entry for group in cells.values() for entry in group]
                counts[arm] = tally(rows) | {k: sum(r[k] for r in rows)
                                             for k in ("original_motif_tail", "correct_without_eos")}
            windows[budget] = {"arms": counts, "break_minus_close": contrast(arms["motif_break"], arms["close"]),
                               "break_minus_native": contrast(arms["motif_break"], arms["native"]),
                               "break_minus_fresh": contrast(arms["motif_break"], arms["no_think"])}
        report[f"{dataset}/{split}/{cap}/source-draw-{draw}"] = {
            "eligible": sum(x["header"]["eligible"] for x in pieces),
            "total": sum(x["header"]["panel_size"] for x in pieces),
            "mean_motif_mass": statistics.fmean(x["original_motif_probability"] for x in events) if events else None,
            "mean_log_alternative_mass": statistics.fmean(x["allowed_log_mass_before_truncation"] for x in events) if events else None,
            "windows": windows,
        }
    return report


def information_summary(scores):
    groups = defaultdict(list)
    for record in scores:
        h = record["header"]
        if "worker_sha256_information" in h and "budgets" in record:
            group = h.get("checkpoint_revision", "base"), h["panel_sha256"], h["split"]
            groups[group].append(record)
    result = {}
    for g, shards in sorted(groups.items()):
        budgets = {}
        for cap in shards[0]["budgets"]:
            totals, paired = defaultdict(Counter), defaultdict(dict)
            for shard in shards:
                window = shard["budgets"][cap]
                for arm, data in window["arms"].items():
                    totals[arm].update(data)
                    values = window["pairs"][arm]
                    if paired[arm].keys() & values.keys():
                        raise ValueError("duplicate information depth prompts")
                    paired[arm].update(values)
            outcomes = {a: {p: [{"ok": y} for y in ys] for p, ys in by_id.items()}
                        for a, by_id in paired.items()}
            contrasts = (("conv_only", "native"), ("zero_conv", "native"), ("zero_conv", "conv_only"),
                         ("both", "native"), ("verified_record", "fresh_evidence"))
            budgets[cap] = {"arms": totals, "eos_paired_both_values": {
                f"{a}-{b}": contrast(outcomes[a], outcomes[b], "ok") for a, b in contrasts}}
        result["/".join(g)] = budgets
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--watch", action="store_true")
    a = p.parse_args()
    previous = None
    while True:
        status = load(a.root / "status.json")
        terminal = (a.root / "release.json").is_file()
        signature = (tuple((k, v["stage"]) for k, v in status["jobs"].items()), terminal)
        if signature != previous:
            result = summarize(a.root, status)
            motif, depth = borrowed_reports(a.root, status)
            scores = [x for n, x in submitted_scores(a.root, status).items() if "canary" not in n]
            result.update(motif=motif_summary(motif), replay_nested=replay_results(depth),
                          information_nested=information_summary(scores))
            path = a.output if terminal or not a.watch else a.output.with_suffix(".partial.json")
            save(path, result)
            previous = signature
        if terminal or not a.watch or time.time() > status["deadline"] + 240:
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
