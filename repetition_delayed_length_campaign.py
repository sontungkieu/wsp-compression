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
ARCHIVED_WORKER = "db7ddc8598b6b78c3ac9e1653303cbada5b8285a8596e17fe4b9248b5e05417c"
CAMPAIGN_HELPER = "d87750404144dea62a981ebf4e350ab2a7ebd1ef59a4d851b1bd444aa5ede598"
BEFORE_PANEL = "979a96af65143cda4908f4b1b27adff4b84be033b1a1b95ec9e2ba9e698ffad6"
BEFORE_SUMMARY = "9fe701e3c7ab664327995f4a37915ce3696227dc3efd789aeb25bba091493627"
BEFORE_RELEASE = "115c4c77f017622e489a39f191e98fa108c48a9280b8d28d56e78e93c94ab1e0"
PRESERVED = ("id", "family", "slot", "public", "retained", "evidence",
             "targets", "record_changed_positions")


def panel_proof(source_panel, new_panel, whitespace=False, before_panel=None, front_padding=0):
    if sha(source_panel) != ARCHIVED_PANEL:
        raise ValueError("Archived panel bytes differ from the locked holdout")
    old, new = load(source_panel), load(new_panel)
    if old["tokenizer_sha256"] != new["tokenizer_sha256"]:
        raise ValueError("Matched queries used a different tokenizer")
    if before_panel is not None and sha(before_panel) != BEFORE_PANEL:
        raise ValueError("Before-padding panel does not match the frozen reference")
    before_pad = load(before_panel)["panels"] if before_panel is not None else None
    counts = {}
    splits = {"canary": 2, "pilot": 16, "eval": 64} if whitespace else {"canary": 2, "eval": 64}
    for split, count in splits.items():
        before, after = old["panels"][split], new["panels"][split]
        if len(before) != count or len(after) != count:
            raise ValueError(f"Unexpected cohort size for {split}")
        lengths = Counter()
        for index, (existing, derived) in enumerate(zip(before, after, strict=True)):
            old_query, query = existing["query"], derived["query"]
            if (any(existing[key] != derived[key] for key in PRESERVED)
                    or query["apply"] != old_query["apply"] or query["read"] == old_query["read"]
                    or len(query["read"]) != len(query["apply"])):
                raise ValueError(f"Unexpected change besides the matched READ query for {derived['id']}")
            if whitespace and (query["read"][0] != old_query["read"][0]
                               or query["read"][-(len(old_query["read"]) - 1):] != old_query["read"][1:]):
                raise ValueError(f"Whitespace experiment changed a READ clause token for {derived['id']}")
            if before_pad is not None:
                original_read = old_query["read"]
                reference = before_pad[split][index]
                reference_read = reference["query"]["read"]
                pad = reference_read[1:-len(original_read) + 1]
                if (reference["id"] != existing["id"] or len(pad) != 12
                        or reference_read != original_read[:1] + pad + original_read[1:]
                        or query["read"] != original_read[:1] + pad[:front_padding]
                        + original_read[1:] + pad[front_padding:]):
                    raise ValueError(f"READ is not an exact move of the padding for {derived['id']}")
            lengths[(len(old_query["read"]), len(query["read"]), len(query["apply"]))] += 1
        counts[split] = [{"old_read": a, "new_read": b, "apply": c, "ids": n}
                         for (a, b, c), n in sorted(lengths.items())]
    return counts


def reference_args(archive, prefix="DQ", summary_pin=ARCHIVED_SUMMARY):
    summary = archive / "jobs" / f"{prefix}-eval-summary.json"
    if sha(summary) != summary_pin:
        raise ValueError("Archived comparison summary digest differs")
    ref = load(summary)
    status = load(archive / "status.json")["jobs"]
    reports, hashes = [], []
    for start in range(0, 64, 8):
        name = f"{prefix}-eval-{start:03d}-score"
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


def original_raws(archive):
    provenance = reference_args(archive)
    reports = provenance[1:9]
    status = load(archive / "status.json")["jobs"]
    raws = []
    for report in reports:
        name = Path(report).stem.removesuffix("-score")
        mark = status[name]
        raw = archive / "jobs" / f"{name}.jsonl"
        if (mark["stage"] != "completed" or Path(mark["output"]) != raw
                or load(Path(report))["raw_sha256"] != mark["sha256"] or sha(raw) != mark["sha256"]):
            raise ValueError(f"Original raw receipt differs for {name}")
        raws.append((raw, mark["sha256"]))
    return raws


def jobs_for(root, worker, panel, model, data, references=(), archived=None, whitespace=False,
             technical_only=False, split_smoke=False):
    out = root / "jobs"
    jobs = []

    def add(name, lane, kind, target, argv, deps, extension, preferred=0, eta=120):
        jobs.append({"id": name, "lane": lane, "kind": kind, "worker": str(target),
                     "extension": extension, "deps": deps, "preferred": preferred,
                     "priority": {"canary": 0, "pilot": 5, "eval": 10, "archive": 20}[lane], "eta": eta,
                     "argv": [*argv, "--output", str(out / (name + extension))]})

    common = ["--panel", str(panel), "--model", str(model)]

    def shard(split, start, count, draws, preferred, deps):
        canary = split == "canary"
        name = f"DQL-{split}-{start:03d}"
        add(name, split, "gpu", worker, ["run", *common, "--data", str(data), "--split", split,
            "--start", str(start), "--count", str(count), "--draws", str(draws),
            "--cap", "128" if canary else "512", "--deadline", str(DEADLINE)],
            deps, ".jsonl", preferred=preferred, eta=330 if split_smoke else 240 if canary else 700)
        score = name + "-score"
        add(score, split, "cpu", worker, ["score", *common, "--raw", str(out / (name + ".jsonl")),
            "--fingerprint", f"{{sha:{name}}}"], [name], ".json", eta=30 if split_smoke else 120)
        return score

    def score_args(names):
        return ["--panel", str(panel), "--reports", *(str(out / (n + ".json")) for n in names),
                "--fingerprints", *(f"{{sha:{n}}}" for n in names)]

    technical = [] if split_smoke else [shard("canary", j, 1, 1, j, []) for j in range(2)]
    gates = technical
    if not (technical_only or split_smoke):
        gate = "DQL-canary-gate"
        add(gate, "canary", "cpu", worker, ["summarize", "--split", "canary", *score_args(technical)],
            technical, ".json")
        gates = [gate]
    if whitespace:
        pilot = [shard("pilot", j * 4, 4, 1, j % 2, gates) for j in range(4)]
        gate = "DQL-pilot-gate"
        add(gate, "pilot", "cpu", worker, ["summarize", "--split", "pilot", *score_args(pilot)],
            pilot, ".json")
        gates = [gate]
    heldout = [shard("eval", j * 8, 8, 2, j % 2, gates) for j in range(8)]
    originals = []
    if archived is not None:
        old_panel, old_worker, raw_inputs = archived
        for j, (raw, pin) in enumerate(raw_inputs):
            name = f"DQL-old-rescore-{j*8:03d}"
            add(name, "archive", "cpu", worker, ["rescore", "--panel", str(old_panel), "--model", str(model),
                "--raw", str(raw), "--fingerprint", pin, "--archive-worker", str(old_worker),
                "--archive-worker-sha256", ARCHIVED_WORKER], [], ".json")
            originals.append(name)
        references = ["--reference-reports", *(str(out / f"{name}.json") for name in originals),
                      "--reference-fingerprints", *(f"{{sha:{name}}}" for name in originals)]
    add("DQL-eval-summary", "eval", "cpu", worker,
        ["summarize", "--split", "eval", *score_args(heldout), *references],
            heldout + originals, ".json", eta=20 if split_smoke else 120)
    return jobs


def launch(args):
    live = preflight()
    if not set(SELECTED) <= set(live["idle"]):
        raise ValueError("Expected GPU indexes 1 and 3 to still be idle")
    split = args.padding == "split"
    if time.time() >= DEADLINE - (450 if split else 1800):
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
    whitespace = args.padding == "whitespace"
    after = args.padding in ("after", "split")
    front_padding = 6 if split else 0
    if after:
        before = args.before_panel
        if before is None or sha(before) != BEFORE_PANEL:
            raise ValueError("Expected the pinned before-padding panel")
        release = before.parent / "release.json"
        if sha(release) != BEFORE_RELEASE or load(release)["status_sha256"] != sha(before.parent / "status.json"):
            raise ValueError("Before-padding release proof differs")
        freeze = ["freeze-after", "--before-panel", str(before), "--before-fingerprint", BEFORE_PANEL]
        if split:
            freeze += ["--front-padding-tokens", str(front_padding)]
    else:
        before = None
        freeze = ["freeze-match", "--padding", args.padding]
    subprocess.run([sys.executable, "-B", str(worker), *freeze, "--source-panel", str(args.source_panel),
                    "--fingerprint", ARCHIVED_PANEL, "--model", str(model), "--panel", str(panel)],
                   check=True, env=env)
    shapes = panel_proof(args.source_panel, panel, whitespace, before, front_padding)
    controller = Path(__file__).resolve()
    config = {"host": HOST, "host_devices": DEVICES, "devices": [DEVICES[i] for i in SELECTED],
              "deadline": DEADLINE, "access": DEADLINE - 900, "source_root": str(source),
              "runtime": env["PYTHONPATH"], "queue_sha256": args.queue_sha256}
    if split:
        config.update(access=DEADLINE - 120, admission_floor_seconds=45, admission_buffer_seconds=30)
    archive = args.source_panel.parent
    if whitespace:
        prior_worker = base / "tmp" / archive.name / worker.name
        if sha(prior_worker) != ARCHIVED_WORKER:
            raise ValueError("Original source worker checksum differs")
        archived = (args.source_panel, prior_worker, original_raws(archive))
        jobs = jobs_for(root, worker, panel, model, data, archived=archived, whitespace=True)
    elif after:
        references = reference_args(before.parent, prefix="DQL", summary_pin=BEFORE_SUMMARY)
        jobs = jobs_for(root, worker, panel, model, data, references, technical_only=True, split_smoke=split)
    else:
        jobs = jobs_for(root, worker, panel, model, data, reference_args(archive))
    (root / "plans").mkdir()
    plan = root / "plans" / "plan-00-delayed-query-length.json"
    save(plan, {"jobs": jobs, "pins": {str(worker): args.worker_sha256, str(controller): sha(controller),
                                      str(helper): CAMPAIGN_HELPER}})
    save(root / "config.json", config)
    seal = {"deadline": DEADLINE, "device_indexes": SELECTED, "panel_sha256": sha(panel), "padding": args.padding,
            "source_panel_sha256": ARCHIVED_PANEL, "panel_query_lengths": shapes,
            "preflight_sha256": sha(root / "preflight.json"), "manifest_shas": {plan.name: sha(plan)},
            "jobs": dict(Counter(j["lane"] + ":" + j["kind"] for j in jobs))}
    if after:
        seal["before_panel_sha256"] = BEFORE_PANEL
        seal["before_release_sha256"] = BEFORE_RELEASE
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
    parser.add_argument("--padding", choices=("wording", "whitespace", "after", "split"), required=True)
    parser.add_argument("--before-panel", type=Path)
    launch(parser.parse_args())


if __name__ == "__main__":
    main()
