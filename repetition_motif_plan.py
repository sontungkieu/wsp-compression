"""Admit motif replays only from receipt-verified 16K archival baselines."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics


def load(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, data):
    if path.exists():
        raise FileExistsError(path)
    temp = path.with_suffix(".incoming")
    with temp.open("x") as stream:
        json.dump(data, stream)
    temp.replace(path)


def option(job, name):
    args = job["argv"]
    return args[args.index(name) + 1]


def append(root, jobs, name, lane, worker, argv, *, deps=(), priority, eta, preferred):
    jobs.append(dict(id=name, lane=lane, kind="gpu", worker=str(worker), extension=".jsonl", argv=argv,
                     deps=list(deps), priority=priority, eta=eta, preferred=preferred))
    report = name + ("-review" if lane == "MOTIF" else "-score")
    args = ["score", "--model", option(jobs[-1], "--model"), "--panel", option(jobs[-1], "--panel"),
            "--raw", option(jobs[-1], "--output"), "--fingerprint", "{sha:" + name + "}",
            "--output", str(root / "jobs" / (report + ".json"))]
    jobs.append(dict(id=report, lane=lane, kind="cpu", worker=str(worker), extension=".json", argv=args,
                     deps=[name], priority=priority + 1, eta=150, preferred=0))


def file_source(archive, status, job):
    name = job["id"]
    mark = status["jobs"].get(name, {})
    file = archive / "jobs" / (name + ".jsonl")
    if mark.get("stage") != "completed" or not file.is_file() or sha(file) != mark["sha256"]:
        return None
    return str(file), mark["sha256"]


def best_dev(archive, status, candidates, anchor_prefix):
    selected = []
    for job in candidates:
        score = anchor_prefix + job["id"].split("-", 1)[1] + "-score"
        mark = status["jobs"].get(score, {})
        file = archive / "jobs" / (score + ".json")
        if mark.get("stage") != "completed" or sha(file) != mark["sha256"]:
            continue
        detail = load(file)
        if detail["eligible"]:
            selected.append((detail["eligible"], job["id"], job))
    if not selected:
        raise ValueError("canary source must have a previously confirmed exact cycle")
    return max(selected, key=lambda row: row[:2])[-1]


def motif(root, archive, status, plan, worker, nemo, deadline):
    source = [j for j in plan if j["kind"] == "gpu"]
    if nemo:
        primary = [j for j in source if j["id"].startswith("N-qwen35-08b-amc_val-")
                   and j["id"].endswith("-d0-b16384")]
        evaluate = [j for j in primary if "-eval-" in j["id"]]
        dev = best_dev(archive, status, [j for j in primary if "-dev-" in j["id"]], "NTR-exact-")
    else:
        primary = [j for j in source if j["id"].startswith("A-math500-")]
        evaluate = [j for j in primary if "-eval-" in j["id"]]
        evaluate += [j for j in source if j["id"].startswith("H-08b-amc_val-eval-")
                     and j["id"].endswith("-d0-b16384")]
        dev = best_dev(archive, status, [j for j in primary if "-dev-" in j["id"]], "B-exact-")
    jobs = []
    for base in [dev, *evaluate]:
        canary = base is dev
        result = file_source(archive, status, base)
        if result is None:
            if canary:
                raise ValueError("canary parent raw receipt is missing")
            continue
        raw, digest = result
        name = "MOTIF-canary" if canary else "MOTIF-" + base["id"]
        cap = 256 if canary else 4096
        args = ["run", "--panel", option(base, "--panel"), "--model", option(base, "--model"),
                "--data", option(base, "--data"), "--raw", raw, "--fingerprint", digest,
                "--deadline", str(deadline), "--cap", str(cap), "--draws", "1" if canary else "2",
                "--output", str(root / "jobs" / (name + ".jsonl"))]
        if canary:
            args += ["--canary", "--limit", "4"]
        append(root, jobs, name, "MOTIF", worker, args, deps=() if canary else ("MOTIF-canary-review",),
               priority=1 if canary else 18, eta=600 if canary else 1600, preferred=(len(jobs) // 2) % (8 if nemo else 4))
    if len(jobs) <= 2:
        raise ValueError("no verified evaluation archive for the motif branch")
    return jobs


def huy_reserve(root, archive_state, old_plan, worker, deadline):
    times = defaultdict(list)
    for name, state in archive_state["jobs"].items():
        m = re.match(r"(H-[^-]+)-(math500|amc_val)-eval-.*-b16384$", name)
        if m and state["stage"] == "completed":
            times[m[1], m[2]].append(state["wall_seconds"])
    jobs = []
    predicted = 0.0
    for job in old_plan:
        name = job["id"]
        m = re.match(r"(H-[^-]+)-(math500|amc_val)-eval-(d+)-d(2|14)-b16384$", name)
        if not m or job["kind"] != "gpu":
            continue
        model, dataset, offset, source_draw = m.groups()
        if not times[model, dataset]:
            continue
        elapsed = statistics.median(times[model, dataset])
        for draw in (40, 42, 44, 46, 48, 50):
            ident = "RES-" + name.replace(f"-d{source_draw}-", f"-d{draw}-")
            argv = job["argv"].copy()
            argv[argv.index("--draw") + 1] = str(draw)
            argv[argv.index("--deadline") + 1] = str(deadline)
            argv[argv.index("--output") + 1] = str(root / "jobs" / (ident + ".jsonl"))
            append(root, jobs, ident, "RES-" + dataset, worker, argv, priority=40 + draw,
                   eta=max(300, int(1.25 * elapsed)), preferred=(draw // 2 + int(offset) // 8) % 4)
            predicted += elapsed
    return jobs, predicted


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "root", "source"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--node", choices=("nemo", "huy"), required=True)
    a = p.parse_args()
    old, root, src = a.archive, a.root, a.source
    release = load(old / "release.json")
    status = load(old / "status.json")
    deadline = load(root / "config.json")["deadline"]
    if release["status_sha256"] != sha(old / "status.json") or release["cutoff_epoch"] != deadline:
        raise ValueError("archive completion or allocation checksum mismatch")
    nemo = a.node == "nemo"
    base = load(old / "plans" / ("plan-01-transfer.json" if nemo else "plan-01-A.json"))["jobs"]
    reserve = [] if nemo else load(old / "plans" / "plan-08-fla-transfer.json")["jobs"]
    worker = src / "repetition_motif_intervention.py"
    jobs = motif(root, old, status, base + reserve, worker, nemo, deadline)
    dest = root / "plans" / ("plan-03-motif.json" if nemo else "plan-00-motif.json")
    save(dest, {"pins": {str(worker): sha(worker)}, "jobs": jobs})
    result = {"motif_jobs": len(jobs), "motif_sha256": sha(dest)}
    if not nemo:
        worker = src / "repetition_followup_baseline_extended.py"
        supplement, seconds = huy_reserve(root, status, reserve, worker, deadline)
        if supplement:
            dest = root / "plans" / "plan-01-archive-power-reserve.json"
            save(dest, {"pins": {str(worker): sha(worker)}, "jobs": supplement})
            result.update(reserve_jobs=len(supplement), reserve_gpu_hours_archive_median=round(seconds / 3600, 2),
                          reserve_sha256=sha(dest))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
