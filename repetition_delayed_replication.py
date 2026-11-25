"""Freeze an independent, balanced set of delayed-READ record pairs."""

import argparse
from collections import Counter
from pathlib import Path
import random

from repetition_baseline import sha
from repetition_budget_run import packed_hash
from repetition_delayed_query import FAMILIES, PAD_SPLIT, make_case
from repetition_delayed_timing import frozen_panel
from repetition_mediation_queue import save


def paired_record_key(case):
    return tuple(sorted(packed_hash(ids) for ids in case["retained"]))


def freeze(args, tokenizer):
    tokenizer_pin = sha(args.model / "tokenizer.json")
    old = frozen_panel(args.source, args.source_sha256, tokenizer, tokenizer_pin)
    excluded = [old]
    if args.exclude_panel is not None:
        excluded.append(frozen_panel(args.exclude_panel, args.exclude_sha256, tokenizer, tokenizer_pin))
    occupied = {paired_record_key(case) for plan in excluded for block in plan["panels"].values() for case in block}
    keys = {case["id"] for plan in excluded for block in plan["panels"].values() for case in block}
    pads = {tuple(case["query"]["read"][1:7] + case["query"]["read"][17:])
            for block in old["panels"].values() for case in block}
    if len(pads) != 1:
        raise ValueError("replication requires one pinned archived whitespace block")
    pad = list(pads.pop())
    panels = {}
    for split, target in (("canary", 2), ("eval", 64)):
        cases = []
        start = args.block * 32
        for number in range(start, start + target // 2):
            for family in FAMILIES:
                rng = random.Random(f"wsp-independent-delayed-read-v1:{split}:{family}:{number}")
                slot_order = list(range(4))
                random.Random(f"wsp-independent-delayed-read-slots:{split}:{family}:{number // 4}").shuffle(slot_order)
                for _ in range(200):
                    case = make_case(tokenizer, split, family, number, slot_order[number % 4], rng)
                    if case is not None and paired_record_key(case) not in occupied:
                        break
                else:
                    raise ValueError("insufficient disjoint new record pairs")
                read = case["query"]["read"]
                if (case["id"] in keys or len(read) != 11 or len(case["query"]["apply"]) != 23
                        or not tokenizer.decode(read[:1]).isspace()):
                    raise ValueError("new READ does not satisfy the locked token contract")
                keys.add(case["id"])
                occupied.add(paired_record_key(case))
                query = case["query"] | {"read": read[:1] + pad[:6] + read[1:] + pad[6:]}
                cases.append(case | {"query": query, "original_read_sha256": packed_hash(read)})
        if split == "eval" and Counter((c["family"], c["slot"]) for c in cases) != {
            (f, s): 8 for f in FAMILIES for s in range(4)
        }:
            raise ValueError("unbalanced replication by family and slot")
        panels[split] = cases
    save(args.output, {"format": 1, "protocol": PAD_SPLIT, "tokenizer_sha256": tokenizer_pin,
                       "source_panel_sha256": args.source_sha256, "panels": panels, "families": FAMILIES,
                       "replication_block": args.block, "excluded_panel_sha256": args.exclude_sha256,
                       "scope": "new IDs and record values with the pinned 6+6 READ token layout"})
    frozen_panel(args.output, sha(args.output), tokenizer, tokenizer_pin)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "model", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--source-sha256", required=True)
    p.add_argument("--block", type=int, choices=(1, 2), default=1)
    p.add_argument("--exclude-panel", type=Path)
    p.add_argument("--exclude-sha256")
    args = p.parse_args()
    if (args.exclude_panel is None) != (args.exclude_sha256 is None):
        p.error("--exclude-panel and --exclude-sha256 must be used together")
    if args.output.exists():
        raise FileExistsError(args.output)
    from transformers import AutoTokenizer

    freeze(args, AutoTokenizer.from_pretrained(args.model, local_files_only=True))


if __name__ == "__main__":
    main()
