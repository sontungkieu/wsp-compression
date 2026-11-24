"""Track the original motif's return and phase closure on sealed raw rollouts."""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from repetition_baseline import sha
from repetition_late_replay import ends_in_motif
from repetition_motif_intervention import BREAK, CLOSE, NATIVE, RESTART, outcome, source_cases
from repetition_motif_plan import load, save
from repetition_replay import terminal_cycle


def first_return(ids, motif):
    """Return start and confirmation as 1-based token positions."""
    m = len(motif)
    rotations = {tuple(motif[i:] + motif[:i]) for i in range(m)}
    for start in range(len(ids) - 2 * m + 1):
        first = tuple(ids[start:start + m])
        if first in rotations and first == tuple(ids[start + m:start + 2 * m]):
            return start + 1, start + 2 * m
    return None, None


def inspect(root, name, status, tokenizers, manifest):
    mark, review = status["jobs"][name], status["jobs"][name + "-review"]
    raw, score = Path(mark["output"]), Path(review["output"])
    if sha(raw) != mark["sha256"] or sha(score) != review["sha256"] or load(score)["raw_sha256"] != mark["sha256"]:
        raise ValueError(f"motif audit receipt differs for {name}")
    with raw.open() as stream:
        h, *body, end = map(json.loads, stream)
    plan = next(j for j in manifest if j["id"] == name)
    args = plan["argv"]
    panel = Path(args[args.index("--panel") + 1])
    model = args[args.index("--model") + 1]
    if h["panel_sha256"] != sha(panel) or end["kind"] != "complete":
        raise ValueError("motif source panel or completion differs")
    if model not in tokenizers:
        from transformers import AutoTokenizer
        tokenizers[model] = AutoTokenizer.from_pretrained(model, local_files_only=True)
    _, cases = source_cases(Path(h["source"]), h["source_sha256"], panel, h["cap"], h["limit"],
                            h["source_draw"], h["canary"], h.get("source_worker", "baseline"))
    by_id = {x["row"]["id"]: x for x in cases}
    report = []
    for row in body:
        if row["kind"] == "event":
            continue
        case = by_id[row["id"]]
        arm, motif = row["arm"], case["motif"]
        m = len(motif)
        for budget_name, budget in (("2m", min(2*m, h["cap"])), ("8m", min(8*m, h["cap"])),
                                    ("full", h["cap"])):
            used = row["tokens"][:budget]
            closure = used.index(h["marker"]) if h["marker"] in used else None
            thought = used if closure is None else used[:closure]
            first, confirmed = first_return(thought, motif) if arm != RESTART else (None, None)
            scored = outcome(case, row, tokenizers[model], h, budget)
            report.append({"id": row["id"], "draw": row["draw"], "arm": arm, "budget": budget_name,
                "limit": budget, "period": m, "first_original_start": first, "first_original_confirmed": confirmed,
                "first_natural_close": closure + 1 if closure is not None and arm in (BREAK, NATIVE) else None,
                "new_terminal_motif": arm != RESTART and terminal_cycle(thought) is not None
                                      and not ends_in_motif(thought, motif),
                **{k: scored[k] for k in ("correct_eos", "correct_without_eos", "wrong_eos", "missing_eos", "cap")}})
    return h["dataset"], {"raw_sha256": mark["sha256"], "review_sha256": review["sha256"],
                         "rows": report, "events": [x for x in body if x["kind"] == "event"]}


def finish(root, output, deadline):
    partial = output.with_suffix(".partial.json")
    cache, tokenizers = {}, {}
    old_signature = None
    while time.time() < deadline + 240:
        f = root / "status.json"
        if not f.is_file():
            if (root / "gate-result.json").is_file() and load(root / "gate-result.json")["state"] != "launched":
                return
            time.sleep(30)
            continue
        status = load(f)
        terminal = (root / "release.json").is_file()
        ids = [n for n, r in status["jobs"].items() if n.startswith("MOTIF-") and n.endswith("-review")
               and "canary" not in n and r["stage"] == "completed"]
        signature = tuple(ids), terminal
        if signature != old_signature:
            plans = [j for plan in (root / "plans").glob("plan-*-motif.json") for j in load(plan)["jobs"]]
            for n in ids:
                if n not in cache:
                    cache[n] = inspect(root, n.removesuffix("-review"), status, tokenizers, plans)
            counts, starts, closures = defaultdict(Counter), defaultdict(list), defaultdict(list)
            for dataset, shard in cache.values():
                for r in shard["rows"]:
                    k = "/".join((dataset, r["budget"], r["arm"]))
                    counts[k].update({metric: int(r[metric]) for metric in (
                        "correct_eos", "correct_without_eos", "wrong_eos", "missing_eos", "cap", "new_terminal_motif")})
                    counts[k].update(n=1, original_return=r["first_original_confirmed"] is not None)
                    if r["first_original_start"] is not None:
                        starts[k].append(r["first_original_start"])
                    if r["first_natural_close"] is not None:
                        closures[k].append(r["first_natural_close"])
            for k, counts_by_arm in counts.items():
                counts_by_arm["natural_close"] = len(closures[k])
                counts_by_arm["median_original_return_start"] = statistics.median(starts[k]) if starts[k] else None
                counts_by_arm["median_natural_close"] = statistics.median(closures[k]) if closures[k] else None
            target = output if terminal else partial
            # Snapshot replacement is deliberate. The frozen inputs remain byte-identical.
            temp = target.with_suffix(".incoming")
            with temp.open("x") as stream:
                json.dump({"state": status["state"], "n_scored_shards": len(cache), "counts": counts,
                           "scope": "mechanistic common-prefix replay; first return confirms two original periods before closure",
                           "shards": cache}, stream)
            temp.replace(target)
            old_signature = signature
        if terminal:
            return
        time.sleep(45)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("launch", "watch"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--deadline", type=int, required=True)
    a = p.parse_args()
    cfg = load(a.root / "config.json")
    if cfg["deadline"] != a.deadline:
        raise ValueError("audit deadline differs from the sealed campaign")
    output = a.root / "motif-path-audit.json"
    if a.action == "watch":
        finish(a.root, output, a.deadline)
        return
    receipt = a.root / "motif-path-launch.json"
    if receipt.exists() or time.time() >= a.deadline:
        raise FileExistsError(receipt)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": cfg["runtime"],
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "2"}
    with (a.root / "motif-path-audit.log").open("x") as log:
        proc = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()), "watch",
                                 "--root", str(a.root), "--deadline", str(a.deadline)], env=env,
                                start_new_session=True, stdout=log, stderr=subprocess.STDOUT)
    result = {"pid": proc.pid, "source_sha256": sha(Path(__file__)), "output": str(output), "deadline": a.deadline}
    save(receipt, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
