"""Capture the block-four direction and test whether it transports to native prefixes."""
import argparse
from collections import Counter
from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
import statistics
import time

from repetition_baseline import sha
from repetition_canonical import seed_for
from repetition_exit_localize import capture, decoder_layers, hidden
from repetition_mediation import INSERTIONS
from repetition_mediation_queue import load, save
from repetition_replay import with_newline_tags
from repetition_state_data import THINK_END, copies
from repetition_state_value import CAP, Runtime, delivery_header, inputs_for, read_matrix, summarize

FRACTIONS = (.25, .5, 1.)
DIRECTION_ARMS = ("native", "fixed_0.25", "fixed_0.5", "fixed_1", "gaussian_1")


def block_and_down(runtime):
    block = decoder_layers(runtime.model, runtime.torch)[3]
    down = block.mlp.down_proj
    if not isinstance(down, runtime.torch.nn.Linear) or down.weight.ndim != 2:
        raise ValueError("block four down_proj is not an output-by-input linear layer")
    return block, down


def capture_x(runtime, module, parts, offsets):
    values = []

    def hook(_module, args):
        x = args[0]
        if x.ndim != 3 or x.shape[0] != len(parts) or x.shape[-1] != module.in_features:
            raise ValueError("down_proj received an incompatible hidden tensor")
        values.extend(x[i, offsets[i]].float().cpu().tolist() for i in range(len(parts)))

    reg = module.register_forward_pre_hook(hook)
    try:
        logits = runtime.model(**runtime.pack(parts), use_cache=False, logits_to_keep=1).logits[:, -1]
    finally:
        reg.remove()
    if len(values) != len(parts):
        raise ValueError("down_proj probe did not fire exactly once")
    return values, logits


@contextmanager
def residual_direction(block, direction, torch):
    applied = []

    def hook(_module, _args, result):
        if applied:
            return result
        tensor = hidden(result)
        changed = tensor.clone()
        changed[:, -1] = (tensor[:, -1].float() + direction).to(tensor.dtype)
        applied.append(True)
        return (changed, *result[1:]) if isinstance(result, tuple) else changed

    reg = block.register_forward_hook(hook)
    try:
        yield
    finally:
        reg.remove()
    if not applied:
        raise ValueError("block-four prefill hook was not applied")


def gaussian(direction, seeds, torch):
    vectors = [torch.randn(direction.shape, device=direction.device, generator=torch.Generator(
        device=direction.device).manual_seed(seed ^ 0x13C27BF902C5D034)) for seed in seeds]
    g = torch.stack(vectors)
    return g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12) * direction.norm()


def extract(runtime, down, block, rows, teacher, out):
    def emit(record):
        print(json.dumps(record, allow_nan=False), file=out)

    for start in range(0, len(rows), 2):
        group = rows[start:start + 2]
        if group[0]["stratum"] == "copy_zero":
            parts = [x["no_think_ids"] + runtime.tokenizer.encode(x["row"]["gold"], add_special_tokens=False) for x in group]
            offsets = [[-1, -min(3, len(ids))] for ids in parts]
            x, _ = capture_x(runtime, down, parts, offsets)
            for item, values in zip(group, x, strict=True):
                emit({"kind": "features", "id": item["row"]["id"], "stratum": "copy_zero", "zeros": values})
        else:
            parts = [x["native_ids"] for x in group]
            offsets = [[-min(len(ids), offset) for offset in (1, 2, 8, 32, 128)] for ids in parts]
            native, _ = capture_x(runtime, down, parts, offsets)
            post, post_offsets = [], []
            for x, ids in zip(group, parts, strict=True):
                past = teacher[x["row"]["id"], 0, "e0"]["tokens"][:4]
                if not past or past[0] != THINK_END:
                    raise ValueError("post-close regularization lacks a verified E0 tag")
                post.append(ids + past)
                post_offsets.append([-1, -2] if len(past) >= 2 else [-1])
            finished, _ = capture_x(runtime, down, post, post_offsets)
            controls = [with_newline_tags(runtime.tokenizer, ids, 41, INSERTIONS) for ids in parts]
            activations = {}
            for variant in ("blank", "left"):
                _, store = capture(runtime.model, runtime.pack([x[variant] for x in controls]), {3: block})
                activations[variant] = store[3].float()
            difference = (activations["left"] - activations["blank"]).cpu().tolist()
            for item, x, zeros, delta in zip(group, native, finished, difference, strict=True):
                emit({"kind": "features", "id": item["row"]["id"], "stratum": item["stratum"], "x": x[0],
                      "zeros": x[1:] + zeros, "direction_sample": delta})
        out.flush()
        print(f"features prompts={start + len(group)}/{len(rows)}", flush=True)


def run(args):
    base = args.root / "cohort.json"
    dev = load(base)["dev"]
    seed_copy = copies("dev")[:8]
    if args.phase == "features":
        cases = dev + [item | {"stratum": "copy_zero"} for item in seed_copy]
        arms, draws, cap = (), 0, 0
    else:
        cases, arms, draws, cap = dev, DIRECTION_ARMS, 4, CAP
    runtime = Runtime(args)
    torch = runtime.torch
    block, down = block_and_down(runtime)
    items = inputs_for(runtime.tokenizer, cases[args.part::args.partitions])
    out = delivery_header(runtime, args, items, arms, cap, draws, len(cases), sha(base))
    out["rank_worker_sha256"] = sha(Path(__file__))
    start = time.monotonic()
    with args.output.open("x") as stream, torch.inference_mode():
        print(json.dumps(out), file=stream, flush=True)
        if args.phase == "features":
            _, _, teacher = read_matrix(args.root, "labels-dev")
            normal = [x for x in items if x["stratum"] != "copy_zero"]
            copy = [x for x in items if x["stratum"] == "copy_zero"]
            extract(runtime, down, block, normal, teacher, stream)
            extract(runtime, down, block, copy, teacher, stream)
        else:
            gate = load(args.root / "direction.json")
            vector = torch.tensor(gate["direction"], device="cuda", dtype=torch.float32)
            for draw in range(draws):
                for begin in range(0, len(items), 6):
                    batch = items[begin:begin + 6]
                    for arm in arms:
                        if time.time() >= args.deadline - 60:
                            raise TimeoutError("direction phase deadline")
                        hook = nullcontext()
                        if arm.startswith("fixed_"):
                            v = vector * gate["reference_norm"] * float(arm.removeprefix("fixed_"))
                            hook = residual_direction(block, v, torch)
                        elif arm == "gaussian_1":
                            v = gaussian(vector * gate["reference_norm"],
                                         [seed_for(x["row"]["id"], draw) for x in batch], torch)
                            hook = residual_direction(block, v, torch)
                        for row in runtime.generate(batch, arm, draw, cap, hook=hook):
                            print(json.dumps(row, allow_nan=False), file=stream)
                    stream.flush()
                    print(f"direction draw={draw} prompts={begin + len(batch)}/{len(items)}", flush=True)
        print(json.dumps({"kind": "complete", "seconds": time.monotonic() - start}), file=stream)


def feature_rows(root):
    seen = {}
    receipt = load(root / "queue/features-done.json")
    for filename, checksum in receipt["outputs"]:
        path = Path(filename)
        if sha(path) != checksum:
            raise ValueError("feature shard checksum mismatch")
        with path.open() as stream:
            head = json.loads(next(stream))
            expected = {x["id"] for x in head["cases"]}
            done = False
            for line in stream:
                r = json.loads(line)
                if r["kind"] == "complete":
                    done = True
                    continue
                if r["id"] not in expected or r["id"] in seen:
                    raise ValueError("feature record missing from shard manifest or duplicated")
                expected.remove(r["id"])
                seen[r["id"]] = r
            if expected or not done:
                raise ValueError("feature shard incomplete")
    return seen


def summarize_features(root):
    import numpy as np

    rows = feature_rows(root)
    d = np.asarray([x["direction_sample"] for x in rows.values() if x["stratum"] != "copy_zero"])
    v = d.mean(0)
    norm = np.linalg.norm(v)
    magnitude = float(np.median(np.linalg.norm(d, axis=-1)))
    if not (norm > 0 and magnitude > 0 and np.isfinite(d).all()):
        raise ValueError("empty or nonfinite residual direction")
    save(root / "direction.json", {"direction": (v / norm).tolist(), "reference_norm": magnitude,
                                   "cohort_sha256": sha(root / "cohort.json"), "block": 3})
    save(root / "features-summary.json", {"prompts": dict(Counter(r["stratum"] for r in rows.values())),
                                         "mean_diff_norm": float(norm), "reference_norm": magnitude,
                                         "direction_sha256": sha(root / "direction.json")})


def summarize_direction(root):
    summarize(root, "direction", tuple((arm, "native") for arm in DIRECTION_ARMS if arm != "native"))
    data = load(root / "direction-summary.json")
    counts = data["strata"]["all"]["arms"]
    arm = max((f"fixed_{n:g}" for n in FRACTIONS), key=lambda k: (counts[k]["success"], -counts[k]["wrong_parsed_eos"],
                                                                -float(k.removeprefix("fixed_"))))
    reference = load(root / "direction.json")["reference_norm"]
    dose = reference * float(arm.removeprefix("fixed_"))
    threshold = max(2, round(.025 * data["strata"]["all"]["per_arm"]))
    passed = counts[arm]["success"] - counts["native"]["success"] >= threshold
    save(root / "direction-choice.json", {"selected_arm": arm, "dose": dose, "transport_gate": passed,
                                          "native_success": counts["native"]["success"],
                                          "fixed_success": counts[arm]["success"], "threshold_rollouts": threshold,
                                          "summary_sha256": sha(root / "direction-summary.json")})


def continue_run(queue):
    if not queue.stage("features", worker=queue.rank, max_reserve=900):
        queue.stage("labels-eval", max_reserve=700)
        return
    if not queue.stage("direction", worker=queue.rank, max_reserve=1400):
        queue.stage("labels-eval", max_reserve=700)
        return
    choice = load(queue.root / "direction-choice.json")
    queue.status(direction_gate=choice["transport_gate"])
    if not choice["transport_gate"]:
        queue.stage("labels-eval", max_reserve=700)
        return
    marker = queue.src / "ridge-ready.json"
    wait_until = min(queue.args.deadline - 1500, time.time() + 180)
    while not marker.is_file() and time.time() < wait_until:
        queue.status("waiting_for_ridge_release")
        time.sleep(8)
    if not marker.is_file():
        queue.stage("labels-eval", max_reserve=700)
        queue.status(ridge_release="not_available")
        return
    contracts = load(marker)["sources"]
    if "repetition_rank_eval.py" not in contracts or any(sha(queue.src / name) != digest for name, digest in contracts.items()):
        raise ValueError("ridge continuation SHA mismatch")
    from repetition_rank_eval import continue_ridge
    continue_ridge(queue)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("run", "summarize"))
    p.add_argument("--phase", choices=("features", "direction"), required=True)
    p.add_argument("--part", type=int)
    p.add_argument("--partitions", type=int)
    p.add_argument("--deadline", type=int)
    for k in ("output", "root", "model", "data"):
        p.add_argument("--" + k, type=Path)
    args = p.parse_args()
    if args.action == "run":
        run(args)
    elif args.phase == "features":
        summarize_features(args.root)
    else:
        summarize_direction(args.root)


if __name__ == "__main__":
    main()
