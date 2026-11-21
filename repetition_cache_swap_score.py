"""Score paired cache-swap rollouts after verifying cohort and raw shard receipts."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from repetition_baseline import BOXED, DECIMAL, sha
from repetition_budget_score import assess_window
from repetition_cache_swap import ARMS
from repetition_canonical import paired_interval, seed_for
from repetition_mediation_queue import load, save
from repetition_state_data import THINK_END
from repetition_state_value import post_seed


def first_box(ids, tokenizer):
    def has_box(tokens):
        return any(DECIMAL.fullmatch(m.group(1).strip())
                   for m in BOXED.finditer(tokenizer.decode(tokens, skip_special_tokens=False)))

    if not has_box(ids):
        return None
    lo, hi = 1, len(ids)
    while lo < hi:
        mid = (lo+hi)//2
        if has_box(ids[:mid]):
            hi = mid
        else:
            lo = mid+1
    return lo


def summarize(root, tokenizer):
    [turn] = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    cohort = root/"cohort.json"
    cases = {c["row"]["id"]: c for c in load(cohort)["cases"]}
    y, source = {}, {}
    events = {}
    manifests, logit_changes = [], []
    for receipt in sorted((root/"queue").glob("draw*-done.json")):
        for name, fingerprint in load(receipt)["outputs"]:
            path = Path(name)
            if sha(path) != fingerprint:
                raise ValueError("cache-swap output hash mismatch")
            with path.open() as stream:
                header, *rows, complete = map(json.loads, stream)
            if (header["cohort_sha256"] != sha(cohort) or header["phase"] != "dev"
                    or complete["kind"] != "complete" or tuple(header["arms"]) != ARMS):
                raise ValueError("incomplete or unexpected cache-swap shard")
            manifests.append([name, fingerprint])
            draw, cap = header["draw"], header["cap"]
            for row in rows:
                if row["kind"] == "audit":
                    logit_changes.extend(row["max_logit_change"])
                    continue
                p, arm = row["id"], row["arm"]
                key = p, draw, arm
                if (key in y or p not in header["ids"] or row["draw"] != draw or row["tokens"][0] != THINK_END
                        or row["seed"] != seed_for(p, draw) or row["post_seed"] != post_seed(p, draw)
                        or (row["eos_token"] is not None and row["eos_token"] not in header["eos_ids"])):
                    raise ValueError("unexpected cache rollout, seed, EOS or intervention")
                y[key] = assess_window(cases[p]["row"], row, "post", cap, tokenizer, turn)
                source[key] = tuple(row["tokens"]), row["eos_token"]
                answer = row["tokens"][1:]
                if turn in answer:
                    answer = answer[:answer.index(turn)]
                events[key] = {"first_numeric_box_post_token":first_box(answer,tokenizer),
                               "second_close_post_token":answer.index(THINK_END)+1 if THINK_END in answer else None}
    draws = {d for _, d, _ in y}
    if not draws or set(y) != {(p, d, a) for p in cases for d in draws for a in ARMS}:
        raise ValueError("incomplete paired cache-swap cohort")
    m = ("eos_correct", "eos", "wrong_parsed_eos", "missing_answer_eos", "unfinished",
         "visible_exact_format", "terminal_cycle", "success", "correct_without_eos")
    groups = defaultdict(list)
    for p, c in cases.items():
        groups["all"].append(p)
        groups[c["stratum"]].append(p)
        if any(source[p,d,ARMS[0]] != source[p,d,ARMS[1]] for d in draws):
            raise ValueError("self-copy differs despite using the same seed")
    table = {}
    for name, prompts in groups.items():
        cells = {}
        for a in ARMS:
            subset = [y[p,d,a] for p in prompts for d in draws]
            cells[a] = {"n":len(subset), **{k:sum(bool(row[k]) for row in subset) for k in m}}
            for event in ("first_numeric_box_post_token", "second_close_post_token"):
                observed = [events[p,d,a][event] for p in prompts for d in draws if events[p,d,a][event] is not None]
                cells[a][event] = {"n":len(observed), "median":statistics.median(observed) if observed else None,
                                   "range":[min(observed),max(observed)] if observed else None}
        swap_deltas = {
            ref: paired_interval({
                p: [int(y[p,d,ARMS[2]]["eos_correct"])-int(y[p,d,ref]["eos_correct"]) for d in draws]
                for p in prompts
            })
            for ref in (ARMS[0], ARMS[3])
        }
        table[name] = {"arms":cells, "swap_minus":swap_deltas}
    return {"draws":len(draws), "prompts":len(cases), "manifests":manifests, "groups":table,
            "marker_logit_max_change_range":[min(logit_changes),max(logit_changes)]}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "model", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    a = p.parse_args()
    from transformers import AutoTokenizer
    report = summarize(a.root, AutoTokenizer.from_pretrained(a.model, local_files_only=True))
    if a.output.exists():
        raise FileExistsError(a.output)
    save(a.output, report)
    print(json.dumps({"prompts":report["prompts"], "draws":report["draws"], "sha256":sha(a.output)}))
