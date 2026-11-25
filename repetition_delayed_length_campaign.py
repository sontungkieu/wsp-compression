"""Run the token-matched delayed-query follow-up on GPU indexes 1 and 3."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import time

from repetition_delayed_campaign import DEVICES, HOST, env_vars, load, paths, preflight, save, sha


DEADLINE = 1795631904
SELECTED = (1, 3)
ARCHIVED_PANEL = "317618796f245a9caacc0a444c3777eba6b241a2690e364a8dcb2b86403547fe"
ARCHIVED_SUMMARY = "e548dc6ce259a22ac0dd4a1f0835e9e319fe54573efc051f8f4227a19329a9ad"
CAMPAIGN_HELPER = "d87750404144dea62a981ebf4e350ab2a7ebd1ef59a4d851b1bd444aa5ede598"
PRESERVED = ("id", "family", "slot", "public", "retained", "evidence",
             "targets", "record_changed_positions")


def panel_proof(source_panel, new_panel):
    if sha(source_panel) != ARCHIVED_PANEL:
        raise ValueError("Archived panel bytes differ from the locked holdout")
    old, new = load(source_panel), load(new_panel)
    if old["tokenizer_sha256"] != new["tokenizer_sha256"]:
        raise ValueError("Matched queries used a different tokenizer")
    counts = {}
    for split, count in (("canary", 2), ("eval", 64)):
        before, after = old["panels"][split], new["panels"][split]
        if len(before) != count or len(after) != count:
            raise ValueError(f"Unexpected cohort size for {split}")
        lengths = Counter()
        for existing, derived in zip(before, after, strict=True):
            old_query, query = existing["query"], derived["query"]
            if (any(existing[key] != derived[key] for key in PRESERVED)
                    or query["apply"] != old_query["apply"] or query["read"] == old_query["read"]
                    or len(query["read"]) != len(query["apply"])):
                raise ValueError(f"Unexpected change besides the matched READ query for {derived['id']}")
            lengths[(len(old_query["read"]), len(query["read"]), len(query["apply"]))] += 1
        counts[split] = [{"old_read": a, "new_read": b, "apply": c, "ids": n}
                         for (a, b, c), n in sorted(lengths.items())]
    return counts


def reference_args(archive):
    summary = archive / "jobs" / "DQ-eval-summary.json"
    if sha(summary) != ARCHIVED_SUMMARY:
        raise ValueError("Archived comparison summary digest differs")
    ref = load(summary)
    status = load(archive / "status.json")["jobs"]
    reports, hashes = [], []
    for start in range(0, 64, 8):
        name = f"DQ-eval-{start:03d}-score"
        report = archive / "jobs" / f"{name}.json"
        mark = status[name]
        receipt = load(archive / "jobs" / f"{name}.receipt.json")
        if receipt != mark or mark["stage"] != "completed" or sha(report) != mark["sha256"]:
            raise ValueError(f"Archived score receipt differs for {name}")
        reports.append(str(report))
        hashes.append(mark["sha256"])
    if hashes != ref["score_shas"]:
        raise ValueError("Archived scores do not match the released summary")
    return ["--reference-reports", *reports, "--reference-fingerprints", *hashes]


def jobs_for(root, worker, panel, model, data, references=()):
    out = root / "jobs"
    jobs = []

    def add(name, lane, kind, target, argv, deps, extension, preferred=0, eta=120):
        jobs.append({"id": name, "lane": lane, "kind": kind, "worker": str(target),
                     "extension": extension, "deps": deps, "preferred": preferred,
                     "priority": 0 if lane == "canary" else 10, "eta": eta,
                     "argv": [*argv, "--output", str(out / (name + extension))]})

    common = ["--panel", str(panel), "--model", str(model)]

    def shard(split, start, count, draws, preferred, deps):
        canary = split == "canary"
        name = f"DQL-{split}-{start:03d}"
        add(name, split, "gpu", worker, ["run", *common, "--data", str(data), "--split", split,
            "--start", str(start), "--count", str(count), "--draws", str(draws),
            "--cap", "128" if canary else "512", "--deadline", str(DEADLINE)],
            deps, ".jsonl", preferred=preferred, eta=240 if canary else 700)
        score = name + "-score"
        add(score, split, "cpu", worker, ["score", *common, "--raw", str(out / (name + ".jsonl")),
            "--fingerprint", f"{{sha:{name}}}"], [name], ".json")
        return score

    def score_args(names):
        return ["--panel", str(panel), "--reports", *(str(out / (n + ".json")) for n in names),
                "--fingerprints", *(f"{{sha:{n}}}" for n in names)]

    technical = [shard("canary", j, 1, 1, j, []) for j in range(2)]
    gate = "DQL-canary-gate"
    add(gate, "canary", "cpu", worker, ["summarize", "--split", "canary", *score_args(technical)],
        technical, ".json")
    heldout = [shard("eval", j * 8, 8, 2, j % 2, [gate]) for j in range(8)]
    add("DQL-eval-summary", "eval", "cpu", worker,
        ["summarize", "--split", "eval", *score_args(heldout), *references],
        heldout, ".json")
    return jobs


def launch(args):
    live = preflight()
    if not set(SELECTED) <= set(live["idle"]):
        raise ValueError("Expected GPU indexes 1 and 3 to still be idle")
    if time.time() >= DEADLINE - 1800:
        raise TimeoutError("Insufficient allocation time to begin the gated wave")
    source, root, base = args.source, args.root, args.base
    if root.parent != base / "runs" or source.parent != base / "tmp" or (root / "launch.json").exists():
        raise ValueError("Use the fresh WSP run and source roots")
    worker = source / "repetition_delayed_query.py"
    queue = source / "repetition_followup_queue.py"
    helper = source / "repetition_delayed_campaign.py"
    if sha(worker) != args.worker_sha256 or sha(queue) != args.queue_sha256 or sha(helper) != CAMPAIGN_HELPER:
        raise ValueError("Staged worker checksum differs")
    verified = load(root / "preflight.json")
    _, model, data = paths(base)
    if verified["abi"]["checkpoint"] != str(model) or verified["inventory"]["host"] != HOST:
        raise ValueError("Preflight used a different host or checkpoint")
    for pinned in verified["abi"]["harness"].values():
        if sha(Path(pinned["path"])) != pinned["sha256"]:
            raise ValueError("Pinned harness bytes changed after preflight")
    env = env_vars(source, base)
    panel = root / "panel.json"
    if sha(args.source_panel) != ARCHIVED_PANEL:
        raise ValueError("Archived holdout digest differs")
    subprocess.run([sys.executable, "-B", str(worker), "freeze-match", "--source-panel", str(args.source_panel),
                    "--fingerprint", ARCHIVED_PANEL, "--model", str(model), "--panel", str(panel)],
                   check=True, env=env)
    shapes = panel_proof(args.source_panel, panel)
    controller = Path(__file__).resolve()
    config = {"host": HOST, "host_devices": DEVICES, "devices": [DEVICES[i] for i in SELECTED],
              "deadline": DEADLINE, "access": DEADLINE - 900, "source_root": str(source),
              "runtime": env["PYTHONPATH"], "queue_sha256": args.queue_sha256}
    jobs = jobs_for(root, worker, panel, model, data, reference_args(args.source_panel.parent))
    (root / "plans").mkdir()
    plan = root / "plans" / "plan-00-delayed-query-length.json"
    save(plan, {"jobs": jobs, "pins": {str(worker): args.worker_sha256, str(controller): sha(controller),
                                      str(helper): CAMPAIGN_HELPER}})
    save(root / "config.json", config)
    seal = {"deadline": DEADLINE, "device_indexes": SELECTED, "panel_sha256": sha(panel),
            "source_panel_sha256": ARCHIVED_PANEL, "panel_query_lengths": shapes,
            "preflight_sha256": sha(root / "preflight.json"), "manifest_shas": {plan.name: sha(plan)},
            "jobs": dict(Counter(j["lane"] + ":" + j["kind"] for j in jobs))}
    save(root / "sealed.json", seal)
    subprocess.run([sys.executable, "-B", str(queue), "launch", "--root", str(root),
                    "--config-sha256", sha(root / "config.json")], check=True, env=env)
    print(json.dumps(seal | {"seal_sha256": sha(root / "sealed.json")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("base", "root", "source", "source-panel"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--worker-sha256", required=True)
    parser.add_argument("--queue-sha256", required=True)
    launch(parser.parse_args())


if __name__ == "__main__":
    main()
