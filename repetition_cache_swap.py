"""Swap Qwen3.5 recurrent memory at the first real post-prefill marker forward."""
import argparse
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
import time

from repetition_baseline import sha
from repetition_budget_run import packed_hash, tokens
from repetition_mediation_queue import load, save
from repetition_sampler_reach import pack_at
from repetition_state_data import STRATA, THINK_END, keyed
from repetition_state_value import Runtime, inputs_for

FIELDS = ("conv_states", "recurrent_states")
BATCH = 4
ARMS = ("keep_native", "keep_self_copy", "keep_kv_swap_recurrent", "question_only")


def freeze(source, output):
    panel = load(source)
    groups = [
        sorted((c for c in panel["cases"] if c["stratum"] == group),
               key=lambda c: keyed(c["row"]["id"], "marker-cache-swap-v1"))[:4]
        for group in STRATA
    ]
    if any(len(g) != 4 for g in groups) or output.exists():
        raise ValueError("require a fresh, balanced 12-prompt development cohort")
    cases = [c for batch in zip(*groups, strict=True) for c in batch]
    save(output, {"cases": cases, "source_cohort_sha256": sha(source), "selection": "hash rank within strata"})


def state_digest(cache, fields, torch):
    h = hashlib.sha256()
    for field in fields:
        for tensor in getattr(cache, field):
            if tensor is None:
                h.update(b"none")
            else:
                item = tensor.detach().contiguous()
                h.update(str((item.shape, item.dtype)).encode())
                h.update(item.cpu().view(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def replace_states(cache, donor, torch, fields=FIELDS):
    changed = {}
    for field in fields:
        target = getattr(cache, field)
        origin = getattr(donor, field)
        changed[field] = 0
        populated = 0
        for i, (a, b) in enumerate(zip(target, origin, strict=True)):
            if a is None and b is None:
                continue
            if a is None or b is None or (a.shape, a.dtype, a.device) != (b.shape, b.dtype, b.device):
                raise ValueError(f"incompatible {field} at layer {i}")
            populated += 1
            changed[field] += not torch.equal(a, b)
            target[i] = b.clone()
            if b.data_ptr() == target[i].data_ptr():
                raise ValueError("replacement aliases the donor")
        if not populated:
            raise ValueError(f"no populated {field}")
    return changed


@contextmanager
def marker_hook(runtime, width, mode, donor=None, digest_kv=False, fields=FIELDS):
    torch = runtime.torch
    audit = {"calls": 0}

    def pre(_module, _args, kwargs):
        ids = kwargs.get("input_ids")
        cache = kwargs.get("past_key_values")
        if cache is None or cache.get_seq_length() != width:
            return
        if ids is None or ids.shape[1] != 1 or not bool((ids == THINK_END).all()):
            raise ValueError("first cached decode did not receive the real closing marker")
        audit["calls"] += 1
        if audit["calls"] != 1 or not cache.has_previous_state:
            raise ValueError("marker hook did not run once with populated recurrent state")
        mask, pos = kwargs["attention_mask"], kwargs["cache_position"]
        if mask.shape[1] != width+1 or pos.tolist() != [width] or not bool(mask[:, -1].all()):
            raise ValueError("unexpected marker mask or cache position")
        audit["cache_position"] = pos.tolist()
        audit["mask_sha256"] = packed_hash(mask.tolist())
        keys, values = tuple(cache.key_cache), tuple(cache.value_cache)
        if digest_kv:
            audit["kv_sha256"] = state_digest(cache, ("key_cache", "value_cache"), torch)
        if mode != "native":
            audit["layers_changed"] = replace_states(cache, cache if mode == "self" else donor, torch, fields)
        if any(a is not b for old, current in ((keys, cache.key_cache), (values, cache.value_cache))
               for a, b in zip(old, current, strict=True)):
            raise ValueError("a recurrence swap replaced attention KV")
        if digest_kv and audit["kv_sha256"] != state_digest(cache, ("key_cache", "value_cache"), torch):
            raise ValueError("a recurrence swap changed attention tensor bytes")
        audit["capturing"] = True

    def post(_module, _args, _kwargs, output):
        if audit.pop("capturing", False):
            audit["logits"] = output.logits[:, -1].float().detach().clone()

    before = runtime.model.register_forward_pre_hook(pre, with_kwargs=True)
    after = runtime.model.register_forward_hook(post, with_kwargs=True)
    try:
        yield audit
    finally:
        before.remove()
        after.remove()
    if audit["calls"] != 1 or "logits" not in audit:
        raise ValueError("did not capture logits after executing the marker token")


def prefill_donor(runtime, group):
    question = [x | {"native_ids": x["native_ids"][:-len(x["completion_prefix"])]} for x in group]
    small = runtime.pack([x["native_ids"] for x in question])
    positions = runtime.model._prepare_position_ids_for_generation(small["input_ids"], dict(small))
    donor = runtime.model(**small, position_ids=positions, use_cache=True, logits_to_keep=1).past_key_values
    if not donor.has_previous_state or donor.get_seq_length() != small["input_ids"].shape[1]:
        raise ValueError("question-only donor cache did not prefill")
    return question, donor


def run_batch(runtime, group, draw, cap, canary):
    torch = runtime.torch
    group = [x | {"presence_seed_ids": []} for x in group]
    question, donor = prefill_donor(runtime, group)
    packed = runtime.pack([x["native_ids"] for x in group])
    width = packed["input_ids"].shape[1]
    donor_width = donor.get_seq_length()
    initial = state_digest(donor, FIELDS, torch) if canary else None

    trajectories, audits = {}, {}
    for arm, mode in zip(ARMS[:3], ("native", "self", "swap"), strict=True):
        with marker_hook(runtime, width, mode, donor, digest_kv=canary) as note:
            trajectories[arm] = runtime.generate(group, "e0", draw, cap, packed=packed)
        audits[arm] = note
    trajectories["question_only"] = runtime.generate(
        question, "e0", draw, cap,
        packed=pack_at(runtime, [x["native_ids"] for x in question], width))
    control, self_copy, swap = (audits[a] for a in ARMS[:3])
    if not torch.equal(control["logits"], self_copy["logits"]) or any(
            tokens(a) != tokens(b) for a, b in zip(trajectories[ARMS[0]], trajectories[ARMS[1]], strict=True)):
        raise ValueError("A11 self-copy differs from the unswapped baseline")
    if {x["mask_sha256"] for x in audits.values()} != {control["mask_sha256"]}:
        raise ValueError("receiver mask differs across recurrence treatments")
    distance = (control["logits"]-swap["logits"]).abs().amax(dim=-1).tolist()
    if max(distance) == 0 or min(swap["layers_changed"].values()) == 0:
        raise ValueError("swapping the recurrent state did not affect the marker readout")
    if canary and (len({x["kv_sha256"] for x in audits.values()}) != 1 or state_digest(donor, FIELDS, torch) != initial):
        raise ValueError("attention receiver or original recurrent donor changed")
    for note in audits.values():
        note.pop("logits")
    return trajectories, {"receiver_width": width, "donor_width": donor_width,
                          "max_logit_change": distance, "audit": audits, "donor_sha256": initial}


def run(a):
    cohort = a.root/"cohort.json"
    selected = load(cohort)["cases"]
    selected = selected[:BATCH] if a.phase == "smoke" else selected[a.part*len(selected)//a.partitions:(a.part+1)*len(selected)//a.partitions]
    if not selected or len(selected) % BATCH or any(not x["completion_prefix"] for x in selected):
        raise ValueError("cache-swap shards require complete nonempty batches")
    r = Runtime(a)
    items = inputs_for(r.tokenizer, selected)
    config_path = Path(inspect.getfile(type(r.model)))
    with a.output.open("x") as out, r.torch.inference_mode():
        def emit(row):
            print(json.dumps(row, allow_nan=False), file=out, flush=True)
        emit({"kind": "manifest", "cohort_sha256": sha(cohort), "worker_sha256_cache": sha(Path(__file__)),
              "modeling_sha256": sha(config_path), "ids": [c["row"]["id"] for c in items],
              "draw": a.draw, "cap": a.cap, "arms": ARMS, "phase": a.phase, **r.manifest})
        for i in range(0, len(items), BATCH):
            if time.time() >= a.deadline-120:
                raise TimeoutError("insufficient time for another cache-swap batch")
            records, audit = run_batch(r, items[i:i+BATCH], a.draw, a.cap, a.phase == "smoke")
            emit({"kind": "audit", "ids": [c["row"]["id"] for c in items[i:i+BATCH]], **audit})
            for arm, rows in records.items():
                for row in rows:
                    emit(row | {"arm": arm})
        emit({"kind": "complete"})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("smoke", "dev"), required=True)
    for name in ("root", "model", "data", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--deadline", type=int, required=True)
    for name, default in (("draw", 0), ("part", 0), ("partitions", 1), ("cap", 512)):
        p.add_argument("--"+name, type=int, default=default)
    run(p.parse_args())
