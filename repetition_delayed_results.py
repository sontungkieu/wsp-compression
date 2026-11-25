"""Audit settled delayed-query scores by mode, template, and token budget."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(path.read_text())


def results(root, split):
    from repetition_delayed_query import ARMS, FRESH, KEEP, MODES, paired_interval, tally

    status = load(root / "status.json")
    panel_path = root / "panel.json"
    panel_sha = sha(panel_path)
    selected = {p["id"] for p in load(panel_path)["panels"][split]}
    announced = load(root / "sealed.json")
    if panel_sha != announced["panel_sha256"]:
        raise ValueError("Panel differs from the locked cohort")
    by_window, sources = defaultdict(list), {}
    for name, job in status["jobs"].items():
        if job["stage"] != "completed" or not name.startswith(f"DQ-{split}-") or not name.endswith("-score"):
            continue
        parent = status["jobs"][name.removesuffix("-score")]
        raw_path, score_path = Path(parent["output"]), Path(job["output"])
        if (raw_path.parent != root / "jobs" or score_path.parent != root / "jobs"
                or sha(raw_path) != parent["sha256"] or sha(score_path) != job["sha256"]):
            raise ValueError(f"Receipt or bytes differ for {name}")
        data = load(score_path)
        h = data["header"]
        if (data["raw_sha256"] != parent["sha256"] or h["panel_sha256"] != panel_sha or h["split"] != split
                or h["worker_sha256_delayed"] != job["worker_sha256"] or h["arms"] != list(ARMS[split])):
            raise ValueError(f"Shard provenance differs for {name}")
        for budget, rows in data["windows"].items():
            by_window[budget].extend(rows)
        sources[name] = {"raw_sha256": parent["sha256"], "score_sha256": job["sha256"],
                         "seconds": parent["wall_seconds"]}

    windows = {}
    for budget, rows in sorted(by_window.items(), key=lambda item: int(item[0])):
        present = {row["id"] for row in rows}
        draws = {row["draw"] for row in rows}
        actual = {(r["id"], r["draw"], r["mode"], r["arm"], r["variant"]) for r in rows}
        expected = {(p, d, m, a, v) for p in present for d in draws
                    for m in MODES for a in ARMS[split] for v in (0, 1)}
        if not present <= selected or actual != expected or len(actual) != len(rows):
            raise ValueError("Score matrix contains incomplete or duplicated cells")
        groups = {"all": rows}
        for family in {row["family"] for row in rows}:
            groups[family] = [row for row in rows if row["family"] == family]
        contrasts = [(arm, KEEP) for arm in ARMS[split] if arm != KEEP]
        if "reset_conv" in ARMS[split]:
            contrasts.append(("reset_conv", FRESH))
        tables, interactions = {}, {}
        for family, subset in sorted(groups.items()):
            modes = {}
            reset_effects = {}
            for mode in MODES:
                arms = {a: [r for r in subset if r["mode"] == mode and r["arm"] == a] for a in ARMS[split]}
                signal = {a: {(r["id"], r["draw"], r["variant"]): int(r["correct_eos"]) - int(r["opposite_eos"])
                              for r in cells} for a, cells in arms.items()}
                comparisons = {}
                for arm, control in contrasts:
                    diff = defaultdict(list)
                    for key, effect in signal[arm].items():
                        diff[key[0]].append(effect - signal[control][key])
                    comparisons[f"{arm}-{control}"] = paired_interval(dict(diff))
                if "reset_conv" in signal:
                    reset_effects[mode] = {key: value - signal[KEEP][key]
                                           for key, value in signal["reset_conv"].items()}
                modes[mode] = {"outcomes": {a: tally(cells) for a, cells in arms.items()},
                               "contrasts_pp": comparisons}
            tables[family] = modes
            if reset_effects:
                diff = defaultdict(list)
                for key, read in reset_effects["read"].items():
                    diff[key[0]].append(read - reset_effects["apply"][key])
                interactions[family] = paired_interval(dict(diff))
        windows[budget] = {"prompts": len(present), "draws": len(draws), "complete": selected == present,
                           "tables": tables, "reset_read_minus_apply_pp": interactions}
    return {"split": split, "panel_sha256": panel_sha, "registered_prompts": len(selected),
            "status": status["state"], "sources": sources, "windows": windows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--split", default="eval", choices=("pilot", "eval"))
    args = parser.parse_args()
    print(json.dumps(results(args.root, args.split)))


if __name__ == "__main__":
    main()
