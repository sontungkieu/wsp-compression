"""Reuse frozen thinking prefixes for a paired check of the published Astra edit."""
import argparse
import json
from pathlib import Path
import time

from repetition_baseline import sha
from repetition_budget_run import packed_hash, tokens
from repetition_budget_score import cut
from repetition_canonical import paired_interval, seed_for
from repetition_followup_baseline import runtime_for, verified
from repetition_mediation_queue import save
from repetition_published_astra import MANIFEST_SHA, PATCH_SHA, PUBLIC_COMMIT, patch_arrays, patch_tensors
from repetition_replay import terminal_cycle
from repetition_restart import score_policy
from repetition_sampler_reach import pack_at
from repetition_state_data import keyed
from repetition_state_value import inputs_for, post_seed

SWITCH = 512
VARIANTS = ("base", "astra")
MODES = ("native", "fixed")


def context(rows, upstream, marker):
    plain, kept = [], []
    for ident, row in rows.items():
        source = upstream[ident, "native", 0]
        item = {"row": row, "render_sha256": source["render_sha256"], "completion_prefix": []}
        plain.append(item)
        if len(source["tokens"]) >= SWITCH and marker not in source["tokens"][:SWITCH]:
            kept.append(item | {"completion_prefix": source["tokens"][:SWITCH]})
    return plain, kept


def copy_panel(rows, count):
    salt = "wsp-followup-astra-copy-v1:" + min(rows)
    words = {a+b+c+d for a in "ptnr" for b in "eou" for c in "kbvs" for d in "aoi"}
    nonce = sorted(words, key=lambda word: keyed(word, salt))[:count]
    cases = []
    for i, word in enumerate(nonce):
        gold = " ".join([word] * 3) if i % 2 == 0 else f"[{word} | {word}]"
        prompt = (f"Write the word {word} exactly three times separated by one space."
                  if i % 2 == 0 else f"Copy this string character for character: {gold}")
        cases.append({"row": {"id": f"{salt}:{i:02d}", "task": "copy", "gold": gold,
                              "prompt": f"{prompt} Output only the requested string."},
                      "completion_prefix": []})
    return cases


def run(args):
    header, rows, upstream, _ = verified(args.raw, args.fingerprint, args.panel)
    r = runtime_for(args)
    [marker] = r.groups["think"]
    if marker != header["marker"] or r.manifest["tokenizer_sha256"] != header["tokenizer_sha256"]:
        raise ValueError("Astra replay requires the baseline tokenizer and marker")
    plain, kept = context(rows, upstream, marker)
    if args.cap <= SWITCH + 1 or (args.canary and not kept):
        raise ValueError("invalid replay budget or canary prefix")
    batches = {"native": inputs_for(r.tokenizer, plain), "fixed": inputs_for(r.tokenizer, kept)}
    copy_items = inputs_for(r.tokenizer, copy_panel(rows, args.copy_count))
    edits = patch_arrays(args.artifact, args.publisher_manifest)
    partial = args.output.with_suffix(".partial.jsonl")
    if args.output.exists():
        raise FileExistsError(args.output)
    with partial.open("x") as stream, r.torch.inference_mode():
        def emit(item):
            print(json.dumps(item, allow_nan=False), file=stream, flush=True)
        emit({"kind": "manifest", "source": str(args.raw), "raw_source_sha256": args.fingerprint,
              "panel_sha256": sha(args.panel), "worker_sha256_astra": sha(Path(__file__)),
              "cap": args.cap, "draws": args.draws, "split": header["split"], "dataset": header["dataset"],
              "turn": header["turn"], "marker": marker, "copy_count": args.copy_count,
              "publisher_commit": PUBLIC_COMMIT, "patch_sha256": PATCH_SHA,
              "publisher_manifest_sha256": MANIFEST_SHA, **r.manifest})
        start = time.monotonic()
        for variant in VARIANTS:
            if variant == "astra":
                emit({"kind": "verified_edit", "tensors": patch_tensors(
                    r.torch, dict(r.model.named_parameters()), edits, apply=True)})
            for mode, items in batches.items():
                spent = SWITCH if mode == "fixed" else 0
                for i in range(0, len(items), args.batch):
                    group = items[i:i+args.batch]
                    group = [x | {"presence_seed_ids": []} for x in group]
                    width = max(len(x["native_ids"]) for x in group)
                    packed = pack_at(r, [x["native_ids"] for x in group], width)
                    for draw in range(args.draws):
                        if time.time() >= args.deadline - 150:
                            raise TimeoutError("Astra shard reached the reserved end window")
                        action, cap = ("e0", args.cap-spent-1) if spent else ("native", args.cap)
                        traces = r.generate(group, action, draw, cap, packed=packed)
                        for item, trace in zip(group, traces, strict=True):
                            emit(trace | {"variant": variant, "mode": mode, "spent": spent,
                                          "input_sha256": packed_hash(item["native_ids"])})
            for i in range(0, len(copy_items), args.batch):
                if time.time() >= args.deadline - 150:
                    raise TimeoutError("Astra copy reached the reserved end window")
                group = copy_items[i:i+args.batch]
                for trace in r.generate(group, "no_think", 0, 192):
                    emit(trace | {"kind": "copy", "variant": variant, "mode": "copy", "spent": 0})
        emit({"kind": "complete", "seconds": time.monotonic()-start})
    partial.replace(args.output)


def outcome(row, trace, budget, tok, header):
    ids, eos = cut(trace, "total", budget)
    answer = score_policy(row, trace | {"tokens": ids, "eos_token": trace["eos_token"] if eos else None},
                          tok, header["turn"])
    return answer | {"closed": header["marker"] in ids, "terminal_exact": terminal_cycle(ids) is not None,
                     "tokens": len(ids) + int(eos)}


def report(path, fingerprint, panel, model, source_worker_sha=None):
    from transformers import AutoTokenizer
    if sha(path) != fingerprint:
        raise ValueError("Astra shard checksum mismatch")
    with path.open() as stream:
        header, *body, end = map(json.loads, stream)
    worker_sha = source_worker_sha or sha(Path(__file__))
    if (header["panel_sha256"] != sha(panel) or header["worker_sha256_astra"] != worker_sha
            or header["patch_sha256"] != PATCH_SHA or end["kind"] != "complete"
            or sum(x["kind"] == "verified_edit" for x in body) != 1):
        raise ValueError("Astra shard attestation is incomplete")
    _, rows, upstream, _ = verified(Path(header["source"]), header["raw_source_sha256"], panel)
    _, kept = context(rows, upstream, header["marker"])
    eligible = {"native": set(rows), "fixed": {x["row"]["id"] for x in kept}}
    copies = {x["row"]["id"]:x["row"] for x in copy_panel(rows, header["copy_count"])}
    data = {}
    for trace in body:
        if trace["kind"] == "verified_edit":
            continue
        ident, draw, variant, mode = (trace[x] for x in ("id", "draw", "variant", "mode"))
        key = ident, draw, mode, variant
        spent = SWITCH if mode == "fixed" else 0
        limit = 192 if mode == "copy" else header["cap"] - spent
        eos = trace["eos_token"]
        known_eos = eos is None or eos in header["eos_ids"] or (mode=="copy" and eos==header["turn"])
        if (key in data or len(tokens(trace)) > limit or trace["spent"] != spent
                or (mode == "fixed" and trace["tokens"][0] != header["marker"])
                or trace["seed"] != seed_for(ident, draw) or trace["post_seed"] != post_seed(ident, draw)
                or not known_eos):
            raise ValueError("invalid paired Astra rollout")
        data[key] = trace
    draws = range(header["draws"])
    if data.keys() != ({(p,d,m,v) for m in MODES for p in eligible[m] for d in draws for v in VARIANTS}
                       | {(p,0,"copy",v) for p in copies for v in VARIANTS}):
        raise ValueError("incomplete paired Astra matrix")
    if any(data[p,d,m,"astra"]["input_sha256"] != data[p,d,m,"base"]["input_sha256"]
           for m in MODES for p in eligible[m] for d in draws):
        raise ValueError("Astra and base prompts differ within a paired comparison")
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    windows = {}
    for cap in sorted({n for n in (2048, 4096, header["cap"]) if n <= header["cap"]}):
        modes = {}
        for mode in MODES:
            cells = {p:{v:[outcome(rows[p], data[p,d,mode,v], cap-(SWITCH if mode=="fixed" else 0),
                                  tok, header) for d in draws] for v in VARIANTS} for p in eligible[mode]}
            for p in cells:
                cells[p]["direct_baseline"] = [outcome(rows[p], upstream[p,"no_think",0], cap, tok, header)]
            modes[mode] = cells
        windows[str(cap)] = modes
    copy = {p:{v:bool(data[p,0,"copy",v]["success"]) for v in VARIANTS} for p in copies}
    return {"header":header, "windows":windows, "copy":copy, "raw_sha256":fingerprint,
            "copy_astra_minus_base": paired_interval({p:[int(y["astra"])-int(y["base"])]
                                                     for p,y in copy.items()}) if copy else None,
            "seconds":end["seconds"]}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("run", "score"))
    for name in ("model", "data", "checkpoint", "panel", "raw", "output", "artifact", "publisher-manifest"):
        cli.add_argument("--"+name, type=Path)
    cli.add_argument("--fingerprint")
    cli.add_argument("--source-worker-sha256")
    for name, default in (("cap",4096),("draws",2),("copy-count",8),("batch",8),("deadline",None)):
        cli.add_argument("--"+name,type=int,default=default)
    cli.add_argument("--canary",action="store_true")
    args = cli.parse_args()
    if args.action == "run":
        run(args)
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        save(args.output, report(args.raw, args.fingerprint, args.panel, args.model, args.source_worker_sha256))


if __name__ == "__main__":
    main()
