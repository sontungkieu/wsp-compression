"""Freeze a numeric transfer cohort for the existing follow-up workers."""

import argparse
from decimal import Decimal
import hashlib
import json
import re
from pathlib import Path

from repetition_baseline import DECIMAL, astra_instruction, sha
from repetition_mediation_queue import load, save
from repetition_restart_cohort import canonical_problem, read_source
from repetition_state_data import keyed

ARCHIVE_MANIFESTS = {"cohort.json", "panel.json", "selection.json"}


def problem_key(problem):
    statement = problem.rstrip().removesuffix(astra_instruction("canonical"))
    return hashlib.sha256(" ".join(statement.split()).encode()).hexdigest()


def source_problem(item):
    if "Body" in item:
        return item["Body"].strip() + " " + item["Question"].strip()
    return item["problem"] if "problem" in item else item["question"]


def manifest_problems(node):
    if isinstance(node, dict):
        for key, value in node.items():
            if key in {"problem", "prompt", "question"} and isinstance(value, str):
                yield value
            elif isinstance(value, (dict, list)):
                yield from manifest_problems(value)
    elif isinstance(node, list):
        for value in node:
            yield from manifest_problems(value)


def prior_problems(exclude_sources, archives, output):
    excluded = set()
    references = {}
    for path in exclude_sources:
        references[str(path)] = sha(path)
        excluded.update(problem_key(source_problem(item)) for item in read_source(path))

    manifests = {}
    unreadable = []
    for archive in archives:
        for path in sorted(archive.rglob("*.json")):
            if path.name not in ARCHIVE_MANIFESTS or path.is_symlink() or path == output:
                continue
            try:
                digest = sha(path)
                problems = {problem_key(problem) for problem in manifest_problems(load(path))}
            except (FileNotFoundError, json.JSONDecodeError):
                unreadable.append(str(path))
                continue
            manifests[str(path)] = {"sha256": digest, "problem_count": len(problems)}
            excluded.update(problems)
    return excluded, references, manifests, unreadable


def freeze(source, expected_sha, dataset, repo, revision, split, dev_count,
           output, exclude_sources, archives=()):
    source_sha = sha(source)
    if source_sha != expected_sha or output.exists():
        raise ValueError("source digest differs or cohort output already exists")
    if not re.fullmatch(r"[a-z][a-z0-9_]*", dataset):
        raise ValueError("use a stable lowercase dataset label")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("pin the dataset to a full repository revision")
    if dev_count < 0:
        raise ValueError("development count must be nonnegative")

    excluded, references, manifests, unreadable = prior_problems(exclude_sources, archives, output)
    cases = {}
    skipped = {"nonnumeric": 0, "published_overlap": 0, "duplicate": 0}
    for item in read_source(source):
        problem = source_problem(item)
        key = problem_key(problem)
        answer = str(item["Answer"] if "Answer" in item else item["answer"]).strip()
        if not DECIMAL.fullmatch(answer):
            skipped["nonnumeric"] += 1
            continue
        if key in excluded:
            skipped["published_overlap"] += 1
            continue
        gold = answer.replace(",", "")
        if key in cases:
            if Decimal(cases[key]["row"]["gold"]) != Decimal(gold):
                raise ValueError("duplicate normalized problem has conflicting answers")
            skipped["duplicate"] += 1
            continue
        cases[key] = {"row": {"id": f"{dataset}:{key}", "task": "math",
                             "prompt": canonical_problem(problem), "gold": gold,
                             "boxed_answer_required": True, "astra_prompt_arm": "canonical"}}

    if len(cases) <= dev_count:
        raise ValueError("no unseen numeric cases remain for evaluation")
    ranked = sorted(cases.values(),
                    key=lambda case: keyed(case["row"]["id"], f"external-transfer:{revision}"))
    save(output, {
        "panels": {dataset: {"dev": ranked[:dev_count], "eval": ranked[dev_count:],
                             "clean": len(ranked)}},
        "datasets": {dataset: {"repo": repo, "revision": revision, "split": split,
                                "source": str(source), "sha256": source_sha,
                                "excluded_source_sha256": references,
                                "archive_manifests": manifests, "unreadable_manifests": unreadable,
                                "excluded_problem_count": len(excluded), "skipped": skipped}},
        "selection": "numeric answers, normalized exact deduplication, stable hash order",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("source-sha256", "dataset", "repo", "revision", "split"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--exclude-source", type=Path, action="append", default=[])
    parser.add_argument("--archive", type=Path, action="append", default=[])
    parser.add_argument("--dev", type=int, default=8)
    args = parser.parse_args()
    freeze(args.source, args.source_sha256, args.dataset, args.repo, args.revision,
           args.split, args.dev, args.output, args.exclude_source, args.archive)


if __name__ == "__main__":
    main()
