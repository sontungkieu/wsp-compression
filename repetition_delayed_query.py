"""Read or apply a scratchpad record after receiving a prefill-blind query."""

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from decimal import Decimal
import json
from pathlib import Path
import random
import time

from repetition_baseline import astra_instruction, math_answer, render, sha
from repetition_budget_run import packed_hash, tokens
from repetition_cache_swap import replace_states, state_digest
from repetition_canonical import paired_interval, seed_for
from repetition_followup_baseline import runtime_for
from repetition_mediation_queue import load, save
from repetition_sampler_reach import pack_at
from repetition_state_value import THINK_END, post_seed


KEEP, RESET, NONE, FRESH, SELF = "keep", "reset_conv", "no_record", "fresh_record", "self_copy"
MODES = ("read", "apply")
FAMILIES = ("register", "counter")
SIZES = {"canary": 2, "pilot": 16, "eval": 64}
ARMS = {"canary": (KEEP, RESET, NONE, FRESH, SELF), "pilot": (KEEP, NONE, FRESH),
        "eval": (KEEP, RESET, NONE, FRESH)}
FIELDS = ("conv_states",)
KV = ("key_cache", "value_cache")
MATCHED = "query-length-matched-v1"
PADDED = "query-length-whitespace-v1"
MATCHED_ARMS = {"canary": (KEEP, RESET, NONE, SELF), "eval": (KEEP, RESET)}
PADDED_ARMS = MATCHED_ARMS | {"pilot": (KEEP, RESET)}
READ_ENDINGS = (
    "Give the exact recorded number without adding an explanation.",
    "Give only the exact recorded number without adding any explanation.",
    "Write the exact number as it appears in the record.",
    "Answer with the number exactly as it appears in the record.",
    "Return the recorded number exactly as written in the draft record.",
    "Keep the recorded value unchanged and use it as your answer.",
    "Answer with the recorded value itself, exactly as it was saved.",
    "Your answer should be exactly the number shown in the draft record.",
    "Use the recorded number itself, without changing or explaining it.",
    "Copy the recorded number exactly without changing the value.",
)


def encode(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def make_case(tok, split, family, number, slot, rng):
    ident = f"delayed-query-v1:{split}:{family}:{number:03d}"
    names = ["A", "B", "C", "D"]
    v0 = rng.sample(range(21, 95), 4)
    other = rng.choice(sorted(set(range(21, 95)) - set(v0)))
    v1 = v0.copy()
    v1[slot] = other
    if rng.randrange(2):
        v0, v1 = v1, v0
    rows = [[f"Saved {family} {name}: {value}." for name, value in zip(names, v, strict=True)]
            for v in (v0, v1)]
    prompt = (f"A draft record may list four {family}s. A later query specifies which one to use and what to do. "
              "Use the recorded value. If it is unavailable, answer UNKNOWN. Give only the requested value.\n")
    instruction = astra_instruction("canonical")
    base = {"id": ident, "task": "math", "prompt": prompt + instruction, "gold": "0",
            "astra_prompt_arm": "canonical", "boxed_answer_required": True}
    text = render(tok, base, "think")
    public = encode(tok, text)
    retained = [encode(tok, text + "Draft record:\n" + "\n".join(r) + "\n") for r in rows]
    evidence = [encode(tok, render(tok, base | {"prompt": prompt + "Verified record:\n" + "\n".join(r)
                                              + "\n" + instruction}, "think")) for r in rows]
    if (any(x[:len(public)] != public or len(x) == len(public) for x in retained)
            or len(retained[0]) != len(retained[1]) or len(evidence[0]) != len(evidence[1])):
        return None
    changed = [i for i, (a, b) in enumerate(zip(*retained, strict=True)) if a != b]
    if not changed or changed != list(range(changed[0], changed[-1] + 1)):
        return None
    name = names[slot]
    a, b = (2, 13) if rng.randrange(2) else (3, 11)
    queries = {"read": f"\nLater query: Return the saved {family} {name}.\n",
               "apply": f"\nLater query: Multiply saved {family} {name} by {a}, then add {b}. Return the result.\n"}
    targets = {"read": [v0[slot], v1[slot]], "apply": [a*v0[slot]+b, a*v1[slot]+b]}
    if any(len({len(encode(tok, str(t))) for t in pair}) != 1 for pair in targets.values()):
        return None
    qids = {k: encode(tok, v) for k, v in queries.items()}
    if any(not q or THINK_END in q or set(q) & set(tok.all_special_ids) for q in qids.values()):
        return None
    if any(THINK_END in ids for ids in (public, *retained, *evidence)):
        raise ValueError("unexpected earlier phase close")
    return {"id": ident, "family": family, "slot": slot, "public": public, "retained": retained,
            "evidence": evidence, "query": qids, "targets": targets, "record_changed_positions": changed}


def freeze(tok, path, tokenizer_sha):
    if path.exists():
        raise FileExistsError(path)
    plans = {}
    for split, total in SIZES.items():
        group = []
        for index in range(total//2):
            for family in FAMILIES:
                rng = random.Random(f"wsp-delayed-query-v1:{split}:{family}:{index}")
                order = list(range(4))
                random.Random(f"wsp-delayed-query-slots:{split}:{family}:{index//4}").shuffle(order)
                slot = order[index % 4]
                for _ in range(200):
                    item = make_case(tok, split, family, index, slot, rng)
                    if item is not None:
                        group.append(item)
                        break
                else:
                    raise ValueError("insufficient equal-token-length register pairs")
        plans[split] = group
    save(path, {"format": 1, "panels": plans, "families": FAMILIES, "tokenizer_sha256": tokenizer_sha,
                "scope": "synthetic hidden record, new query forced only after cached prefill"})


def matching_read(tok, original, length):
    read = tok.decode(original, skip_special_tokens=False)
    if encode(tok, read) != original or not read.endswith("\n") or len(original) >= length:
        raise ValueError("archived read query cannot be extended without changing its tokens")
    for ending in READ_ENDINGS:
        candidate = encode(tok, read + ending + "\n")
        if (len(candidate) == length and candidate[:len(original)] == original
                and set(candidate).isdisjoint(set(tok.all_special_ids) | {THINK_END})):
            return candidate
    raise ValueError(f"no registered read wording matches {length} tokens")


def whitespace_read(tok, original, length):
    read = tok.decode(original, skip_special_tokens=False)
    first = tok.decode(original[:1], skip_special_tokens=False)
    if (encode(tok, read) != original or not read.startswith(first) or not first.isspace()
            or len(original) >= length):
        raise ValueError("expected a shorter archived read query beginning with whitespace")
    excluded = set(tok.all_special_ids) | {THINK_END}
    choices = []
    for text in (" ", "\n", "\t", "  ", "\n\n"):
        ids = encode(tok, text)
        if len(ids) == 1 and ids[0] not in excluded and ids[0] not in choices:
            if tok.decode(ids, skip_special_tokens=False).isspace():
                choices.append(ids[0])
    if len(choices) < 2:
        raise ValueError("tokenizer did not expose distinct ordinary whitespace tokens")
    order = random.Random("wsp-delayed-query-neutral-padding-v1")
    pad = []
    for _ in range(length - len(original)):
        pad.append(order.choice([token for token in choices if not pad or token != pad[-1]]))
    padded = original[:1] + pad + original[1:]
    if tok.decode(padded, skip_special_tokens=False) != first + tok.decode(pad, skip_special_tokens=False) + read[len(first):]:
        raise ValueError("whitespace padding changed the executable read clause")
    return padded


def freeze_match(tok, source, fingerprint, path, tokenizer_sha, padding="wording"):
    if path.exists():
        raise FileExistsError(path)
    if sha(source) != fingerprint:
        raise ValueError("original delayed query panel checksum changed")
    plan = load(source)
    protocol, splits, extend = ((MATCHED, MATCHED_ARMS, matching_read) if padding == "wording"
                                else (PADDED, PADDED_ARMS, whitespace_read))
    if (plan["format"] != 1 or plan.get("protocol") is not None or plan["tokenizer_sha256"] != tokenizer_sha
            or any(len(plan["panels"][split]) != SIZES[split] for split in splits)):
        raise ValueError("expected the original complete delayed query panel and tokenizer")
    panels = {}
    for split in splits:
        panels[split] = []
        for case in plan["panels"][split]:
            query = case["query"]
            matched = extend(tok, query["read"], len(query["apply"]))
            panels[split].append(case | {"query": query | {"read": matched},
                                        "original_read_sha256": packed_hash(query["read"])})
    save(path, {"format": 1, "protocol": protocol, "source_panel_sha256": fingerprint,
                "panels": panels, "families": plan["families"], "tokenizer_sha256": tokenizer_sha,
                "scope": "original records and apply query; original read clause with matched-length " + padding})


def protocol_arms(plan, split):
    protocol = plan.get("protocol")
    if protocol == MATCHED:
        return MATCHED_ARMS[split]
    if protocol == PADDED:
        return PADDED_ARMS[split]
    if protocol is not None:
        raise ValueError("unrecognized delayed query protocol")
    return ARMS[split]


def prefill(runtime, ids, width):
    packed = pack_at(runtime, ids, width)
    pos = runtime.model._prepare_position_ids_for_generation(packed["input_ids"], dict(packed))
    cache = runtime.model(**packed, position_ids=pos, use_cache=True, logits_to_keep=1).past_key_values
    if cache.get_seq_length() != width or not cache.has_previous_state:
        raise ValueError("public donor prefill has no usable state")
    return cache


def query_policy(runtime, query):
    torch = runtime.torch

    class QueryPolicy(runtime.policy):
        def __call__(self, ids, scores):
            step = ids.shape[1] - self.width
            if step <= len(query):
                token = query[step] if step < len(query) else THINK_END
                return torch.full_like(scores, -torch.inf).scatter_(1, ids.new_full((self.n, 1), token), 0.)
            return super().__call__(ids, scores)

    return QueryPolicy


@contextmanager
def first_query(runtime, width, first_id, arm, donor, proof):
    torch = runtime.torch

    def pre(_module, _args, kwargs):
        cache = kwargs.get("past_key_values")
        if cache is None or cache.get_seq_length() != width:
            return
        ids = kwargs["input_ids"]
        if (proof or not cache.has_previous_state or ids.shape[1] != 1 or not bool((ids == first_id).all())
                or kwargs["cache_position"].tolist() != [width]
                or kwargs["attention_mask"].shape[1] != width+1):
            raise ValueError("cache intervention is not at the first real query token")
        proof.update(kv=state_digest(cache, KV, torch), mask_sha256=packed_hash(kwargs["attention_mask"].tolist()),
                     width=width, first_query_token=first_id)
        if arm in (RESET, SELF):
            proof["layers_changed"] = replace_states(cache, donor if arm == RESET else cache, torch, FIELDS)
            if state_digest(cache, KV, torch) != proof["kv"]:
                raise ValueError("convolution reset changed attention cache bytes")

    handle = runtime.model.register_forward_pre_hook(pre, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()
    if not proof:
        raise ValueError("did not observe a populated cache before query consumption")


def prefixes(case, arm):
    if arm == FRESH:
        return case["evidence"]
    if arm == NONE:
        return [case["public"]]*2
    return case["retained"]


def batch(runtime, case, mode, draw, post_cap, arms):
    query = case["query"][mode]
    width = max(len(ids) for arm in (KEEP, NONE, FRESH) for ids in prefixes(case, arm))
    donor = prefill(runtime, prefixes(case, NONE), width) if RESET in arms else None
    before = state_digest(donor, FIELDS, runtime.torch) if donor is not None else None
    policy = query_policy(runtime, query)
    result, audit = {}, {}
    for arm in arms:
        ids = prefixes(case, arm)
        rows = [{"row": {"id": case["id"], "task": "math", "gold": str(gold)}, "pending": True,
                 "presence_seed_ids": []} for gold in case["targets"][mode]]
        proof = {}
        with first_query(runtime, width, query[0], arm, donor, proof):
            traces = runtime.generate(rows, "native", draw, len(query)+1+post_cap,
                                      packed=pack_at(runtime, ids, width), policy=policy)
        result[arm] = [trace | {"arm": arm, "variant": v, "mode": mode, "prefix_sha256": packed_hash(ids[v])}
                       for v, trace in enumerate(traces)]
        audit[arm] = proof
    if NONE in result and tokens(result[NONE][0]) != tokens(result[NONE][1]):
        raise ValueError("no-record pair leaked value into generation")
    for arm in (RESET, SELF):
        if arm in audit and any(audit[arm][k] != audit[KEEP][k] for k in ("kv", "mask_sha256", "width")):
            raise ValueError("control and recurrence intervention used different pre-query receivers")
    if SELF in result and any(tokens(a) != tokens(b) for a, b in zip(result[SELF], result[KEEP], strict=True)):
        raise ValueError("self-copy changed the exact trajectory")
    if donor is not None and (state_digest(donor, FIELDS, runtime.torch) != before
                              or audit[RESET]["layers_changed"][FIELDS[0]] <= 0):
        raise ValueError("convolution treatment did not perform an isolated donor swap")
    return result, audit


def run(args):
    plan = load(args.panel)
    cases = plan["panels"][args.split][args.start:args.start + args.count]
    if len(cases) != args.count or not cases or args.output.exists():
        raise ValueError("require a complete frozen shard and a fresh output")
    arms = protocol_arms(plan, args.split)
    if plan.get("protocol") in (MATCHED, PADDED) and (any(len(c["query"]["read"]) != len(c["query"]["apply"]) for c in cases)
            or (args.split == "eval" and (args.cap != 512 or args.draws != 2))):
        raise ValueError("matched eval requires equal query length, two draws and a 512-token trace")
    r = runtime_for(args)
    if r.manifest["tokenizer_sha256"] != plan["tokenizer_sha256"]:
        raise ValueError("frozen IDs came from a different tokenizer")
    partial = args.output.with_suffix(".partial.jsonl")
    started = time.monotonic()
    with partial.open("x") as out, r.torch.inference_mode():
        def emit(row):
            print(json.dumps(row, allow_nan=False), file=out, flush=True)

        emit({"kind": "manifest", "worker_sha256_delayed": sha(Path(__file__)), "panel_sha256": sha(args.panel),
              "split": args.split, "start": args.start, "count": args.count, "draws": args.draws,
              "cap": args.cap, "arms": arms, "protocol": plan.get("protocol"), **r.manifest})
        for item in cases:
            for draw in range(args.draws):
                for mode in MODES:
                    if time.time() >= args.deadline - 120:
                        raise TimeoutError("delayed query shard nearing its deadline")
                    raw, audit = batch(r, item, mode, draw, args.cap, arms)
                    emit({"kind": "audit", "id": item["id"], "mode": mode, "draw": draw, "arms": audit})
                    for pair in raw.values():
                        for trace in pair:
                            emit(trace)
        emit({"kind": "complete", "seconds": time.monotonic()-started})
    partial.replace(args.output)


def assess(tok, case, row, cap, turn=None):
    mode, variant = row["mode"], row["variant"]
    lead = len(case["query"][mode]) + 1
    limited = row["tokens"][lead:lead+cap]
    eos = row["eos_token"] is not None and len(row["tokens"]) - lead < cap
    targets = case["targets"][mode]
    visible = limited[:limited.index(turn)] if turn in limited else limited
    parsed = math_answer({"gold": str(targets[variant])}, tok.decode(visible, skip_special_tokens=False),
                         pending_thought=False)["parsed_answer"]
    selected = targets.index(Decimal(parsed)) if parsed is not None and Decimal(parsed) in targets else None
    return {"correct_eos": eos and selected == variant, "opposite_eos": eos and selected == 1-variant,
            "other_numeric_eos": eos and selected is None and parsed is not None,
            "missing_numeric_eos": eos and parsed is None, "cap": not eos,
            "correct_without_eos": not eos and selected == variant,
            "post_tokens": len(limited) + int(eos)}


def score(args, tok, source_worker_sha=None):
    if sha(args.raw) != args.fingerprint:
        raise ValueError("delayed query raw checksum changed")
    with args.raw.open() as f:
        h, *body, end = map(json.loads, f)
    plan = load(args.panel)
    if (h["worker_sha256_delayed"] != (source_worker_sha or sha(Path(__file__))) or h["panel_sha256"] != sha(args.panel)
            or h["tokenizer_sha256"] != sha(args.model/"tokenizer.json") or end["kind"] != "complete"
            or tuple(h["arms"]) != protocol_arms(plan, h["split"]) or h.get("protocol") != plan.get("protocol")):
        raise ValueError("delayed query provenance changed")
    cohort = plan["panels"][h["split"]][h["start"]:h["start"]+h["count"]]
    items = {c["id"]: c for c in cohort}
    cells, audits = {}, set()
    for row in body:
        p, d, m = row["id"], row["draw"], row["mode"]
        item = items[p]
        if row["kind"] == "audit":
            if (p,d,m) in audits or set(row["arms"]) != set(h["arms"]):
                raise ValueError("duplicate or incomplete delayed visibility audit")
            audits.add((p,d,m))
            continue
        a, v = row["arm"], row["variant"]
        lead = [*item["query"][m], THINK_END]
        if ((p,d,m,a,v) in cells or row["kind"] != "rollout" or row["tokens"][:len(lead)] != lead
                or row["seed"] != seed_for(p,d) or row["post_seed"] != post_seed(p,d)
                or row["prefix_sha256"] != packed_hash(prefixes(item,a)[v])
                or len(tokens(row)) > len(lead)+h["cap"]
                or (row["eos_token"] is None and len(tokens(row)) != len(lead)+h["cap"])
                or (row["eos_token"] is not None and row["eos_token"] not in h["eos_ids"])):
            raise ValueError("invalid delayed query trace")
        cells[p,d,m,a,v] = row
    expected = {(p,d,m) for p in items for d in range(h["draws"]) for m in MODES}
    if audits != expected or cells.keys() != {(*key,a,v) for key in expected for a in h["arms"] for v in (0,1)}:
        raise ValueError("incomplete delayed query matrix")
    [turn] = encode(tok, "<|im_end|>")
    if tok.decode([turn], skip_special_tokens=False) != "<|im_end|>":
        raise ValueError("assistant turn token differs from the scorer policy")
    windows = {str(cap): [{"id": p, "family": items[p]["family"], "draw": d, "mode": m, "arm": a, "variant": v,
                          **assess(tok, items[p], row, cap, turn)} for (p,d,m,a,v),row in cells.items()]
               for cap in sorted({h["cap"], min(h["cap"],256)})}
    return {"header": h, "raw_sha256": args.fingerprint, "seconds": end["seconds"], "windows": windows,
            "answer_boundary": "assistant-turn-v1", "scorer_sha256": sha(Path(__file__))}


def tally(rows):
    counts = Counter()
    for row in rows:
        counts.update(k for k in ("correct_eos", "opposite_eos", "other_numeric_eos", "missing_numeric_eos",
                                  "cap", "correct_without_eos") if row[k])
    d = len(rows)
    return {"n": d, **counts, "delta": (counts["correct_eos"]-counts["opposite_eos"])/d}


def read_scores(paths, fingerprints):
    loaded = []
    for path, fingerprint in zip(paths, fingerprints, strict=True):
        if sha(path) != fingerprint:
            raise ValueError("delayed query score receipt changed")
        loaded.append(load(path))
    return loaded


def correct_eos_contrasts(rows):
    keyed = {(r["id"],r["draw"],r["variant"],r["mode"],r["arm"]): int(r["correct_eos"]) for r in rows}
    cells = {(p,d,v) for p,d,v,_,_ in keyed}
    if len(keyed) != len(cells)*4 or len(rows) != len(keyed):
        raise ValueError("incorrect matched read/apply arm matrix")
    differences = {m: {key: keyed[(*key,m,RESET)]-keyed[(*key,m,KEEP)] for key in cells} for m in MODES}
    return keyed, differences | {"read_minus_apply": {key: differences["read"][key]-differences["apply"][key]
                                                      for key in cells}}


def interval_by_id(values):
    grouped = defaultdict(list)
    for (pid,_,_), value in sorted(values.items()):
        grouped[pid].append(value)
    return paired_interval(dict(grouped))


def original_reference(args, plan, first):
    if args.reference_reports is None and args.reference_fingerprints is None:
        return None
    if (plan.get("protocol") not in (MATCHED, PADDED) or args.split != "eval"
            or not args.reference_reports or not args.reference_fingerprints):
        raise ValueError("original references are for a complete length-matched eval only")
    loaded = read_scores(args.reference_reports, args.reference_fingerprints)
    for report in loaded:
        h = report["header"]
        if (h.get("protocol") is not None or h["panel_sha256"] != plan["source_panel_sha256"]
                or h["split"] != "eval" or tuple(h["arms"]) != ARMS["eval"] or h["draws"] != 2 or h["cap"] != 512
                or any(h[key] != first[key] for key in ("tokenizer_sha256","checkpoint","runtime","sampling"))):
            raise ValueError("original score reference is from a different protocol or runtime")
    return loaded


def summarize(args):
    loaded = read_scores(args.reports, args.fingerprints)
    plan = load(args.panel)
    expected = {c["id"] for c in plan["panels"][args.split]}
    first = loaded[0]["header"]
    arms = protocol_arms(plan, args.split)
    reference = original_reference(args, plan, first)
    boundary = loaded[0].get("answer_boundary")
    if any(r.get("answer_boundary") != boundary for r in loaded + (reference or [])):
        raise ValueError("cannot compare scores with mixed answer boundaries")
    result = {"split": args.split, "panel_sha256": sha(args.panel), "score_shas": args.fingerprints,
              "protocol": plan.get("protocol"), "reference_score_shas": args.reference_fingerprints,
              "answer_boundary": boundary, "windows": {}}
    if any(r["header"]["split"] != args.split or r["header"]["panel_sha256"] != result["panel_sha256"]
           or tuple(r["header"]["arms"]) != arms or r["header"].get("protocol") != plan.get("protocol")
           or r["header"]["draws"] != first["draws"] or r["header"]["cap"] != first["cap"] for r in loaded):
        raise ValueError("mixed score protocols")
    for cap in loaded[0]["windows"]:
        rows = [row for report in loaded for row in report["windows"][cap]]
        keys = [(r["id"],r["draw"],r["mode"],r["arm"],r["variant"]) for r in rows]
        if len(set(keys)) != len(keys) or set(keys) != {(p,d,m,a,v) for p in expected for m in MODES
                for a in arms for v in (0,1) for d in range(first["draws"])}:
            raise ValueError("expected all disjoint scored rows on the frozen panel")
        tables, ci, vs_keep = {}, {}, {}
        for mode in MODES:
            block = [r for r in rows if r["mode"] == mode]
            tables[mode] = {arm: tally([r for r in block if r["arm"] == arm]) for arm in arms}
            ci[mode] = {}
            paired = {}
            for arm in arms:
                paired[arm] = {(r["id"],r["draw"],r["variant"]): int(r["correct_eos"])-int(r["opposite_eos"])
                               for r in block if r["arm"] == arm}
                by_id = defaultdict(list)
                for key, value in paired[arm].items():
                    by_id[key[0]].append(value)
                ci[mode][arm] = paired_interval(dict(by_id))
            vs_keep[mode] = {}
            for arm, values in paired.items():
                if arm == KEEP:
                    continue
                by_id = defaultdict(list)
                for key, value in values.items():
                    by_id[key[0]].append(value - paired[KEEP][key])
                vs_keep[mode][arm] = paired_interval(dict(by_id))
        window = {"tables": tables, "signed_ci": ci, "delta_vs_keep_pp": vs_keep}
        if plan.get("protocol") in (MATCHED, PADDED):
            matched_rows = [r for r in rows if r["arm"] in (KEEP,RESET)]
            paired, contrasts = correct_eos_contrasts(matched_rows)
            window["correct_eos_reset_minus_keep"] = {k: interval_by_id(v) for k,v in contrasts.items()}
            if reference is not None:
                original = [r for report in reference for r in report["windows"][cap] if r["arm"] in (KEEP,RESET)]
                reference_cells, reference_effects = correct_eos_contrasts(original)
                if reference_cells.keys() != paired.keys():
                    raise ValueError("reference and ablation are not paired by prompt, mode, draw and value")
                window["original_correct_eos_reset_minus_keep"] = {k: interval_by_id(v) for k,v in reference_effects.items()}
                window["matched_minus_original_interaction"] = interval_by_id({
                    k: v-reference_effects["read_minus_apply"][k] for k,v in contrasts["read_minus_apply"].items()})
                window["unchanged_apply_correct_eos_agreement"] = sum(
                    paired[k] == reference_cells[k] for k in paired if k[3] == "apply")
        result["windows"][cap] = window
    primary = result["windows"][str(first["cap"])]["tables"]
    result["gate_passed"] = True
    if args.split == "pilot":
        if plan.get("protocol") == PADDED:
            keep_read = primary["read"][KEEP]
            result["gate_passed"] = keep_read.get("correct_eos", 0) >= .75 * keep_read["n"]
        else:
            result["gate_passed"] = all(primary[m][FRESH]["delta"] >= .25 and primary[m][KEEP]["delta"] >= .125
                                        and primary[m][NONE]["delta"] == 0 for m in MODES)
    if plan.get("protocol") in (MATCHED, PADDED) and args.split == "canary":
        rows = [row for report in loaded for row in report["windows"][str(first["cap"])]]
        correct_read = {r["id"] for r in rows if r["arm"] == KEEP and r["mode"] == "read" and r["correct_eos"]}
        result["gate_passed"] = correct_read == expected
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("freeze", "freeze-match", "run", "score", "rescore", "summarize"))
    for key in ("panel", "source-panel", "archive-worker", "model", "checkpoint", "data", "raw", "output"):
        p.add_argument("--"+key, type=Path)
    p.add_argument("--split", choices=tuple(ARMS))
    p.add_argument("--fingerprint")
    p.add_argument("--padding", choices=("wording", "whitespace"), default="wording")
    p.add_argument("--archive-worker-sha256")
    p.add_argument("--reports", type=Path, nargs="+")
    p.add_argument("--fingerprints", nargs="+")
    p.add_argument("--reference-reports", type=Path, nargs="+")
    p.add_argument("--reference-fingerprints", nargs="+")
    for k,v in (("start",0), ("count",2), ("draws",1), ("cap",512), ("deadline",None)):
        p.add_argument("--"+k, type=int, default=v)
    args = p.parse_args()
    if args.action in ("freeze", "freeze-match", "score", "rescore"):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if args.action == "freeze":
        freeze(tok, args.panel, sha(args.model/"tokenizer.json"))
    elif args.action == "freeze-match":
        freeze_match(tok, args.source_panel, args.fingerprint, args.panel, sha(args.model/"tokenizer.json"), args.padding)
    elif args.action == "run":
        if args.start < 0 or args.count < 1 or args.draws < 1 or not 64 <= args.cap <= 512 or args.deadline is None:
            p.error("run requires a range, a 64–512 post-marker cap and the allocation deadline")
        run(args)
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        if args.action == "rescore":
            if args.archive_worker is None or sha(args.archive_worker) != args.archive_worker_sha256:
                raise ValueError("archive worker bytes do not match the pinned rollout worker")
            result = score(args, tok, args.archive_worker_sha256)
        elif args.action == "score":
            result = score(args, tok)
        else:
            result = summarize(args)
        save(args.output, result)
        if args.action == "summarize" and not result["gate_passed"]:
            raise SystemExit("record-use check did not meet the preregistered gate")


if __name__ == "__main__":
    main()
