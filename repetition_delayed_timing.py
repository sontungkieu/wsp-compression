"""Move a convolution-cache intervention while keeping the READ query fixed."""

import argparse
from collections import defaultdict
from contextlib import contextmanager
import json
import math
from pathlib import Path
from statistics import mean
import time
from types import SimpleNamespace

from repetition_baseline import sha
from repetition_budget_run import packed_hash
from repetition_cache_swap import replace_states, state_digest
from repetition_delayed_query import (
    FIELDS, KEEP, KV, NONE, PAD_SPLIT, THINK_END, assess, interval_by_id, prefixes, query_policy, tally,
)
from repetition_followup_baseline import runtime_for
from repetition_mediation_queue import load, save
from repetition_sampler_reach import pack_at


OFFSETS = (0, 7, 17)
RESETS = tuple(f"reset_{offset}" for offset in OFFSETS)
SELFS = tuple(f"self_{offset}" for offset in OFFSETS)
ARMS = {"canary": (KEEP, *RESETS, *SELFS), "eval": (KEEP, *RESETS)}
PROTOCOL = "fixed-read-offset-v1"


def frozen_panel(path, pin, tok, tokenizer_pin):
    if sha(path) != pin:
        raise ValueError("fixed READ panel checksum changed")
    panel = load(path)
    if panel["protocol"] != PAD_SPLIT or panel["tokenizer_sha256"] != tokenizer_pin:
        raise ValueError("expected the pinned 6+6 READ panel")
    for split, expected in (("canary", 2), ("eval", 64)):
        cases = panel["panels"][split]
        if len({c["id"] for c in cases}) != len(cases) or len(cases) != expected:
            raise ValueError("unexpected question count")
        for c in cases:
            query = c["query"]["read"]
            if (len(query) != 23 or len(c["query"]["apply"]) != 23
                    or not tok.decode(query[:1]).isspace()
                    or not tok.decode(query[1:7]).isspace()
                    or not tok.decode(query[7:17]).startswith("Later query: Return the saved ")
                    or not tok.decode(query[17:]).isspace()):
                raise ValueError("READ timing offsets do not delimit the archived query")
    return panel


def at_offset(kwargs, width, expected):
    cache = kwargs.get("past_key_values")
    if cache is None:
        return None
    offset = cache.get_seq_length() - width
    if offset not in expected:
        return None
    pos = width + offset
    ids = kwargs["input_ids"]
    if (not cache.has_previous_state or ids.shape != (2, 1) or not bool((ids == expected[offset]).all())
            or kwargs["cache_position"].tolist() != [pos] or kwargs["attention_mask"].shape != (2, pos + 1)
            or not bool(kwargs["attention_mask"][:, -1].all())):
        raise ValueError(f"did not observe the expected cached token at offset {offset}")
    return offset, cache


def donor_states(runtime, case, width, query):
    torch = runtime.torch
    snapshots = {}
    expected = {offset: query[offset] for offset in OFFSETS}

    def capture(_module, _args, kwargs):
        hit = at_offset(kwargs, width, expected)
        if hit is None:
            return
        offset, cache = hit
        if offset in snapshots:
            raise ValueError("duplicate public-donor capture")
        snapshots[offset] = SimpleNamespace(
            conv_states=[x.detach().clone() if x is not None else None for x in cache.conv_states],
        )

    class ForcedPrefix(runtime.transformers.LogitsProcessor):
        def __call__(self, ids, scores):
            token = query[ids.shape[1] - width]
            return torch.full_like(scores, -torch.inf).scatter_(1, ids.new_full((2, 1), token), 0.)

    hook = runtime.model.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        cfg = runtime.transformers.GenerationConfig(
            do_sample=False, use_cache=True, max_new_tokens=max(OFFSETS) + 2,
            eos_token_id=sorted(runtime.eos), pad_token_id=runtime.pad,
        )
        out = runtime.model.generate(
            **pack_at(runtime, prefixes(case, NONE), width), generation_config=cfg,
            logits_processor=[ForcedPrefix()], **runtime.defaults,
        )
    finally:
        hook.remove()
    forced = query[:max(OFFSETS) + 2]
    if set(snapshots) != set(OFFSETS) or out[:, width:].tolist() != [forced, forced]:
        raise ValueError("public donor did not follow the same cached query prefix")
    return snapshots


@contextmanager
def receiver(runtime, width, query, arm, donors, audit, logits):
    torch = runtime.torch
    modified = None if arm == KEEP else int(arm.rsplit("_", 1)[1])
    positions = OFFSETS if arm == KEEP else (modified,)
    expected = {offset: query[offset] for offset in positions}
    expected[len(query)] = THINK_END
    marker_seen = False

    def pre(_module, _args, kwargs):
        nonlocal marker_seen
        hit = at_offset(kwargs, width, expected)
        if hit is None:
            return
        offset, cache = hit
        if offset == len(query):
            marker_seen = True
            return
        key = str(offset)
        if key in audit:
            raise ValueError("more than one receiver forward at a forced position")
        kv_before = state_digest(cache, KV, torch)
        entry = audit[key] = {
            "kv": kv_before, "mask": packed_hash(kwargs["attention_mask"].tolist()),
            "token": query[offset], "cache_position": width + offset,
        }
        if offset != modified:
            return
        donor = donors[offset] if arm in RESETS else cache
        entry["conv_before"] = state_digest(cache, FIELDS, torch)
        entry["expected_conv"] = state_digest(donor, FIELDS, torch)
        matrices = tuple(cache.recurrent_states)
        entry["layers_changed"] = replace_states(cache, donor, torch, FIELDS)[FIELDS[0]]
        if (tuple(map(id, matrices)) != tuple(map(id, cache.recurrent_states))
                or kv_before != state_digest(cache, KV, torch)
                or entry["expected_conv"] != state_digest(cache, FIELDS, torch)):
            raise ValueError("convolution-cache replacement broke the isolated-state contract")

    def post(_module, _args, _kwargs, output):
        nonlocal marker_seen
        if marker_seen:
            marker_seen = False
            if logits:
                raise ValueError("duplicate marker forward")
            logits.append(output.logits[:, -1].float().detach().clone())

    before = runtime.model.register_forward_pre_hook(pre, with_kwargs=True)
    after = runtime.model.register_forward_hook(post, with_kwargs=True)
    try:
        yield
    finally:
        before.remove()
        after.remove()
    if set(audit) != set(map(str, positions)) or len(logits) != 1:
        raise ValueError("receiver did not execute the requested offsets and marker")


def run_case(runtime, case, draw, cap, arms):
    torch = runtime.torch
    query = case["query"]["read"]
    width = max(len(ids) for arm in (KEEP, NONE, "fresh_record") for ids in prefixes(case, arm))
    donors = donor_states(runtime, case, width, query)
    ids = prefixes(case, KEEP)
    packed = pack_at(runtime, ids, width)
    policy = query_policy(runtime, query)
    group = [{"row": {"id": case["id"], "task": "math", "gold": str(gold)}, "pending": True,
              "presence_seed_ids": []} for gold in case["targets"]["read"]]
    audits, traces, marker_logp = {}, {}, {}
    for arm in arms:
        audit, logits = {}, []
        hook = receiver(runtime, width, query, arm, donors, audit, logits)
        outputs = runtime.generate(group, "native", draw, len(query) + 1 + cap, packed=packed, policy=policy, hook=hook)
        logp = logits[0].log_softmax(-1)
        marker_logp[arm] = logp
        if arm != KEEP:
            offset = arm.rsplit("_", 1)[1]
            if any(audit[offset][key] != audits[KEEP][offset][key] for key in ("kv", "mask", "token", "cache_position")):
                raise ValueError("receiver differed before the only intervention")
            if arm in SELFS and (audit[offset]["layers_changed"] != 0
                                 or not torch.equal(marker_logp[KEEP], logp)
                                 or any(row["tokens"] != keep["tokens"] or row["eos_token"] != keep["eos_token"]
                                        for row, keep in zip(outputs, traces[KEEP], strict=True))):
                raise ValueError("self-copy changed logits or the sampled trajectory")
        audits[arm] = audit
        ref = marker_logp[KEEP]
        kl = (ref.exp() * (ref - logp)).sum(-1).clamp_min(0).tolist()
        traces[arm] = [row | {"variant": variant, "arm": arm, "mode": "read",
                              "prefix_sha256": packed_hash(ids[variant]), "marker_kl_from_keep": kl[variant]}
                       for variant, row in enumerate(outputs)]
    return audits, traces


def run(args):
    runtime = runtime_for(args)
    panel = frozen_panel(args.panel, args.panel_sha256, runtime.tokenizer, runtime.manifest["tokenizer_sha256"])
    cases = panel["panels"][args.split][args.start:args.start + args.count]
    if len(cases) != args.count or not cases or args.output.exists():
        raise ValueError("expected a fresh complete shard")
    arms = ARMS[args.split]
    before = time.monotonic()
    partial = args.output.with_suffix(".partial.jsonl")
    with partial.open("x") as stream, runtime.torch.inference_mode():
        def emit(row):
            print(json.dumps(row, allow_nan=False), file=stream, flush=True)

        emit({"kind": "manifest", "protocol": PROTOCOL, "worker_sha256_timing": sha(Path(__file__)),
              "panel_sha256": args.panel_sha256, "split": args.split, "start": args.start, "count": args.count,
              "draws": args.draws, "cap": args.cap, "arms": arms, "offsets": OFFSETS, **runtime.manifest})
        for case in cases:
            for draw in range(args.draws):
                if time.time() >= args.deadline - 120:
                    raise TimeoutError("READ timing shard nearing the allocation deadline")
                audits, traces = run_case(runtime, case, draw, args.cap, arms)
                emit({"kind": "audit", "id": case["id"], "draw": draw, "arms": audits})
                for pair in traces.values():
                    for trace in pair:
                        emit(trace)
        emit({"kind": "complete", "seconds": time.monotonic() - before})
    partial.replace(args.output)


def score(args):
    from transformers import AutoTokenizer

    if sha(args.raw) != args.fingerprint:
        raise ValueError("READ timing raw checksum changed")
    head, *body, end = map(json.loads, args.raw.read_text().splitlines())
    expected_arms = ARMS[head["split"]]
    tokenizer_pin = sha(args.model / "tokenizer.json")
    if (head["protocol"] != PROTOCOL or head["worker_sha256_timing"] != sha(Path(__file__))
            or head["panel_sha256"] != args.panel_sha256 or head["tokenizer_sha256"] != tokenizer_pin
            or tuple(head["arms"]) != expected_arms or tuple(head["offsets"]) != OFFSETS
            or end["kind"] != "complete"):
        raise ValueError("wrong READ timing provenance")
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    panel = frozen_panel(args.panel, args.panel_sha256, tok, tokenizer_pin)
    items = {item["id"]: item for item in panel["panels"][head["split"]][head["start"]:head["start"] + head["count"]]}
    [turn] = tok.encode("<|im_end|>", add_special_tokens=False)
    audits, cells = set(), {}
    for row in body:
        key = row["id"], row["draw"]
        if row["kind"] == "audit":
            if key in audits or set(row["arms"]) != set(expected_arms):
                raise ValueError("invalid receiver audit matrix")
            audits.add(key)
            continue
        key += row["variant"], row["arm"]
        lead = items[key[0]]["query"]["read"] + [THINK_END]
        if (row["kind"] != "rollout" or key in cells or row["tokens"][:len(lead)] != lead
                or not math.isfinite(row["marker_kl_from_keep"])):
            raise ValueError("wrong rollout identity, query or marker")
        cells[key] = row
    expected = {(pid, draw) for pid in items for draw in range(head["draws"])}
    if (len(items) != head["count"] or audits != expected
            or cells.keys() != {(*pair, v, arm) for pair in expected for v in (0, 1) for arm in expected_arms}):
        raise ValueError("incomplete READ timing shard")
    for pid, draw in expected:
        for v in (0, 1):
            keep = cells[pid, draw, v, KEEP]
            if any(any(cells[pid, draw, v, arm][key] != keep[key] for key in ("seed", "post_seed", "prefix_sha256"))
                   for arm in expected_arms):
                raise ValueError("receiver used a different prefix or draw")
    return {"header": head, "raw_sha256": args.fingerprint, "seconds": end["seconds"], "technical_pass": True,
            "windows": {str(cap): [{"id": pid, "draw": d, "variant": v, "arm": arm,
                                    "marker_kl_from_keep": row["marker_kl_from_keep"],
                                    **assess(tok, items[pid], row, cap, turn)} for (pid, d, v, arm), row in cells.items()]
                        for cap in sorted({head["cap"], min(head["cap"], 256)})}}


def summarize(args):
    reports = []
    for path, pin in zip(args.reports, args.fingerprints, strict=True):
        if sha(path) != pin:
            raise ValueError("READ timing score checksum changed")
        reports.append(load(path))
    header = reports[0]["header"]
    identity = ("protocol", "panel_sha256", "checkpoint", "runtime", "sampling", "split", "cap", "draws", "arms")
    if any(not report["technical_pass"] or any(report["header"][k] != header[k] for k in identity)
           for report in reports):
        raise ValueError("mixed READ timing cells")
    results = {}
    for cap in reports[0]["windows"]:
        rows = defaultdict(dict)
        for report in reports:
            for row in report["windows"][cap]:
                key = row["id"], row["draw"], row["variant"]
                if key in rows[row["arm"]]:
                    raise ValueError("overlapping READ timing score shards")
                rows[row["arm"]][key] = row
        keep = rows[KEEP]
        prompts = len({key[0] for key in keep})
        if (any(cells.keys() != keep.keys() for cells in rows.values())
                or prompts != sum(report["header"]["count"] for report in reports)
                or len(keep) != prompts * header["draws"] * 2):
            raise ValueError("unpaired READ timing arms")

        def diff(left, right):
            return interval_by_id({key: int(rows[left][key]["correct_eos"]) - int(rows[right][key]["correct_eos"])
                                   for key in keep})

        results[cap] = {
            "prompts": prompts, "counts": {arm: tally(cells.values()) for arm, cells in rows.items()},
            "correct_eos_vs_keep_pp": {arm: diff(arm, KEEP) for arm in rows if arm != KEEP},
            "offset_contrasts_pp": {f"{a}_minus_{b}": diff(a, b) for a, b in
                                    ((RESETS[1], RESETS[0]), (RESETS[2], RESETS[0]), (RESETS[2], RESETS[1]))},
            "mean_marker_kl_from_keep": {arm: mean(row["marker_kl_from_keep"] for row in cells.values())
                                         for arm, cells in rows.items()},
        }
    return {"protocol": PROTOCOL, "split": header["split"], "panel_sha256": header["panel_sha256"],
            "score_shas": args.fingerprints, "windows": results}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("run", "score", "summarize"))
    for key in ("panel", "model", "data", "checkpoint", "raw", "output"):
        p.add_argument("--" + key, type=Path, required=key == "output")
    p.add_argument("--panel-sha256")
    p.add_argument("--split", choices=tuple(ARMS))
    for key, default in (("start", 0), ("count", 1), ("draws", 1), ("cap", 512), ("deadline", None)):
        p.add_argument("--" + key, type=int, default=default)
    p.add_argument("--fingerprint")
    p.add_argument("--reports", type=Path, nargs="+")
    p.add_argument("--fingerprints", nargs="+")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.action == "run":
        if (args.split is None or args.deadline is None or args.start < 0 or args.count <= 0
                or (args.cap, args.draws) != ((128, 1) if args.split == "canary" else (512, 2))):
            p.error("expected canary 128/1 or eval 512/2 and a valid scope")
        run(args)
    else:
        save(args.output, score(args) if args.action == "score" else summarize(args))


if __name__ == "__main__":
    main()
