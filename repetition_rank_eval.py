"""Evaluate a development-locked gate and, if supported, its BF16 materialization."""
import argparse
from contextlib import contextmanager, nullcontext
import json
import math
from pathlib import Path
import statistics
import time

from repetition_baseline import first_period, sha
from repetition_canonical import seed_for
from repetition_mediation import SAMPLING
from repetition_mediation_queue import load, save
from repetition_rank_fit import simple_features
from repetition_rank_one import block_and_down, capture_x, gaussian, residual_direction
from repetition_state_data import THINK_END, copies
from repetition_state_value import CAP, Runtime, delivery_header, inputs_for, read_matrix, summarize

BETAS = tuple(n / 4 for n in range(-32, 65))
CORE = ("native", "fixed", "gate_last", "bias", "gaussian", "e0", "e32")
ALL = ("native", "gate_last", "gate_all", "bias")
EDIT = ("native", "gate_all", "edit", "random_edit", "bias")
PHASES = ("match", "rank-eval", "rank-all", "rank-parity", "rank-edit", "rank-full", "rank-copy")


@contextmanager
def down_gate(module, d, v, all_positions, torch):
    seen = []

    def hook(_module, inputs, output):
        if seen and not all_positions:
            return output
        x = inputs[0]
        if all_positions:
            amplitude = x.float() @ v
            changed = (output.float() + amplitude[..., None] * d).to(output.dtype)
        else:
            amplitude = x[:, -1].float() @ v
            changed = output.clone()
            changed[:, -1] = (output[:, -1].float() + amplitude[:, None] * d).to(output.dtype)
        if not seen:
            seen.extend((amplitude[:, -1] if all_positions else amplitude).detach().float().cpu().tolist())
        return changed

    registered = module.register_forward_hook(hook)
    try:
        yield seen
    finally:
        registered.remove()
    if not seen:
        raise ValueError("down_proj gate did not fire")


def phases(root, phase):
    cohort = root / "cohort.json"
    bundle = load(cohort)
    test = bundle["eval"]
    if phase == "match":
        return bundle["dev"], (), 0, 0, sha(cohort)
    if phase == "rank-eval":
        return test, CORE, CAP, 4, sha(cohort)
    if phase == "rank-all":
        return test[:18], ALL, CAP, 4, sha(cohort)
    if phase == "rank-parity":
        return test[:4], (), 0, 0, sha(cohort)
    if phase == "rank-edit":
        return test[:18], EDIT, CAP, 4, sha(cohort)
    if phase == "rank-full":
        panel = [x | {"completion_prefix": []} for x in test[:12]]
        return panel, ("native", "edit", "random_edit", "bias"), 4096, 2, sha(cohort)
    if phase == "rank-copy":
        plan = root / "copy-dev-summary.json"
        n = load(plan)["copy_selection"]["repetitions"]
        return copies("test", n), ("copy_base", "copy_hook", "copy_edit", "copy_random"), 192, 2, sha(plan)
    raise ValueError("unknown rank evaluation phase")


def match_rows(runtime, down, d, v, items):
    torch, tf = runtime.torch, runtime.transformers
    warp_k = tf.TopKLogitsWarper(top_k=SAMPLING["top_k"])
    warp_p = tf.TopPLogitsWarper(top_p=SAMPLING["top_p"])
    for start in range(0, len(items), 3):
        batch = items[start:start + 3]
        ids = [x["native_ids"] for x in batch]
        packed = runtime.pack(ids)
        kwargs = {"use_cache": False, "logits_to_keep": 1}
        before = runtime.model(**packed, **kwargs).logits[:, -1].float()
        with down_gate(down, d, v, False, torch) as coefficient:
            after = runtime.model(**packed, **kwargs).logits[:, -1].float()
        for i, x in enumerate(batch):
            prior = torch.tensor(sorted(set(x["completion_prefix"])), device="cuda", dtype=torch.long)
            before[i, prior] -= 1.5
            after[i, prior] -= 1.5
        t = .6
        warm = after / t
        goal = warp_p(packed["input_ids"], warp_k(packed["input_ids"], warm)).softmax(-1)[:, THINK_END]
        beta = torch.tensor(BETAS, device="cuda", dtype=before.dtype)
        for i, x in enumerate(batch):
            logits = before[i].expand(len(BETAS), -1).clone()
            logits[:, THINK_END] += beta
            logits /= t
            input_ids = packed["input_ids"][i].expand(len(BETAS), -1)
            p = warp_p(input_ids, warp_k(input_ids, logits)).softmax(-1)[:, THINK_END]
            yield {"kind": "match", "id": x["row"]["id"], "grid": p.tolist(),
                   "gate": float(goal[i]), "coefficient": coefficient[i]}


def apply_columns(weight, columns, values):
    return weight.index_copy(1, columns, values)


def edit_components(module, d, v, support, torch):
    base = module.weight.detach().clone()
    cols = torch.tensor(support, dtype=torch.long, device=base.device)
    stored = base.index_select(1, cols)
    exact = d[:, None] * v[cols][None]
    new_columns = (stored.float() + exact).to(base.dtype)
    effective = new_columns.float() - stored.float()
    metrics = {"support_columns": len(support), "effective_columns": int((effective != 0).any(0).sum()),
               "scalar_changes": int((effective != 0).sum()), "norm": float(effective.norm()),
               "pre_round_norm": float(exact.norm()), "dtype": str(base.dtype)}
    return base, cols, stored, new_columns, effective, metrics


def matched_random(stored, edited, effective, torch):
    # Randomize signs on the exact changed scalar coordinates. BF16 spacing
    # around a power of two can make the reflected change unrepresentable.
    engine = torch.Generator(device=stored.device).manual_seed(26551001)
    changed = effective != 0
    reflected = (stored.float() - effective).to(stored.dtype)
    reversible = changed & ((stored.float() - reflected.float()) == effective)
    flip = reversible & (torch.rand(stored.shape, generator=engine, device=stored.device) < .5)
    new = torch.where(flip, reflected, edited)
    actual = new.float() - stored.float()
    if not torch.equal(new != stored, changed) or not torch.equal(actual.abs(), effective.abs()):
        raise ValueError("randomized edit failed BF16 footprint matching")
    desired_norm = float(effective.norm())
    if not desired_norm:
        raise ValueError("rank-one edit disappears after BF16 rounding")
    norm = float(actual.norm())
    n = int(changed.sum())
    flipped = int(flip.sum())
    return new, {"matched": flipped >= .25 * n, "relative_norm_gap": abs(norm / desired_norm - 1.),
                 "norm": norm, "scalar_changes": n, "reversible_scalars": int(reversible.sum()),
                 "flipped_scalars": flipped}


@contextmanager
def use_weight(module, base, edited):
    module.weight.copy_(edited)
    try:
        yield
    finally:
        module.weight.copy_(base)


def parity_rows(runtime, down, d, v, edit, items):
    torch = runtime.torch
    base, modified, _, footprint = edit
    for x in items:
        options = {"use_cache": False, "logits_to_keep": 1}
        packed = runtime.pack([x["native_ids"]])
        held, _ = capture_x(runtime, down, [x["native_ids"]], [[-1]])
        data = torch.tensor(held[0][0], device="cuda", dtype=torch.float32)
        left = base.float() @ data + d * (v @ data)
        right = (base.float() + d[:, None] * v[None]) @ data
        algebra = float((left - right).abs().max())
        with down_gate(down, d, v, True, torch):
            hook_logits = runtime.model(**packed, **options).logits[0, -1].float()
        with use_weight(down, base, modified):
            edit_logits = runtime.model(**packed, **options).logits[0, -1].float()
        err = (hook_logits - edit_logits).abs()
        yield {"kind": "parity", "id": x["row"]["id"], "pre_round_algebra_max": algebra,
               "bf16_logit_p99": float(torch.quantile(err, .99)), "bf16_logit_max": float(err.max()),
               "bf16_think_abs": float(err[THINK_END]), "argmax_equal": bool(hook_logits.argmax() == edit_logits.argmax()),
               "footprint": footprint}


def prepare_edit(down, d, v, support, torch):
    base, cols, stored, new, actual, footprint = edit_components(down, d, v, support, torch)
    random, matched = matched_random(stored, new, actual, torch)
    return base, apply_columns(base, cols, new), apply_columns(base, cols, random), footprint | {"random": matched}


def run(args):
    cases, arms, cap, draws, source = phases(args.root, args.phase)
    runtime = Runtime(args)
    torch = runtime.torch
    block, down = block_and_down(runtime)
    fit = load(args.root / "fit.json")
    direction = load(args.root / "direction.json")["direction"]
    d, v = (torch.tensor(w, device="cuda", dtype=torch.float32) for w in (direction, fit["v"]))
    if down.weight.shape != (len(d), len(v)):
        raise ValueError("rank-one factor does not match down_proj output-by-input shape")
    items = inputs_for(runtime.tokenizer, cases[args.part::args.partitions])
    header = delivery_header(runtime, args, items, arms, cap, draws, len(cases), source)
    header.update(rank_eval_sha256=sha(Path(__file__)), fit_sha256=sha(args.root / "fit.json"))
    start = time.monotonic()
    with args.output.open("x") as stream, torch.inference_mode():
        def emit(record):
            print(json.dumps(record, allow_nan=False), file=stream)
        emit(header)
        if args.phase == "match":
            for r in match_rows(runtime, down, d, v, items):
                emit(r)
        elif args.phase == "rank-parity":
            edit = prepare_edit(down, d, v, fit["support"], torch)
            for r in parity_rows(runtime, down, d, v, edit, items):
                emit(r)
        else:
            beta = load(args.root / "match-summary.json")["bias"]
            weights = None
            if any(a in arms for a in ("edit", "copy_edit")):
                weights = prepare_edit(down, d, v, fit["support"], torch)
                if not weights[3]["random"]["matched"]:
                    raise ValueError("BF16 randomized match not available")
            for draw in range(draws):
                for begin in range(0, len(items), 6):
                    batch = items[begin:begin + 6]
                    for arm in arms:
                        if time.time() >= args.deadline - 60:
                            raise TimeoutError("rank eval deadline")
                        hook = nullcontext()
                        if arm == "fixed":
                            hook = residual_direction(block, d * fit["dose"], torch)
                        elif arm == "gaussian":
                            fixed = d * fit["dose"]
                            hook = residual_direction(block, gaussian(
                                fixed, [seed_for(x["row"]["id"], draw) for x in batch], torch), torch)
                        elif arm in ("gate_last", "gate_all", "copy_hook"):
                            hook = down_gate(down, d, v, arm != "gate_last", torch)
                        elif arm in ("edit", "random_edit", "copy_edit", "copy_random"):
                            base, edited, random, _ = weights
                            hook = use_weight(down, base, random if arm in ("random_edit", "copy_random") else edited)
                        arm_arg = "no_think" if arm.startswith("copy_") else arm
                        # Enter the hook here to preserve activation coefficients for readout.
                        with hook as values:
                            rows = runtime.generate(batch, arm_arg, draw, cap, bias=beta if arm == "bias" else 0.)
                        for i, row in enumerate(rows):
                            record = row | {"arm": arm, "coefficient": values[i] if values is not None else None}
                            if args.phase == "rank-full":
                                cycle = first_period(row["tokens"])
                                record["periodic_onset"] = cycle[0] if cycle else None
                            emit(record)
                    stream.flush()
                    print(f"{args.phase} draw={draw} prompts={begin + len(batch)}/{len(items)}", flush=True)
        emit({"kind": "complete", "seconds": time.monotonic() - start})


def small_matrix(root, phase):
    out = {}
    receipt = load(root / "queue" / f"{phase}-done.json")
    for filename, checksum in receipt["outputs"]:
        path = Path(filename)
        if sha(path) != checksum:
            raise ValueError("rank probe digest mismatch")
        with path.open() as f:
            header = json.loads(next(f))
            ids = {x["id"] for x in header["cases"]}
            done = False
            for line in f:
                r = json.loads(line)
                if r["kind"] == "complete":
                    done = True
                    continue
                if r["id"] not in ids or r["id"] in out:
                    raise ValueError("invalid rank probe matrix")
                ids.remove(r["id"])
                out[r["id"]] = r
            if ids or not done:
                raise ValueError("rank probe is incomplete")
    return out


def choose_bias(root):
    by = list(small_matrix(root, "match").values())
    target = statistics.fmean(r["gate"] for r in by)
    grid = [statistics.fmean(r["grid"][i] for r in by) for i in range(len(BETAS))]
    j = min(range(len(grid)), key=lambda i: (abs(grid[i] - target), abs(BETAS[i])))
    save(root / "match-summary.json", {"bias": BETAS[j], "expected_gate_close": target, "expected_bias_close": grid[j],
                                      "expected_gap": grid[j] - target, "prompts": len(by)})


def summarize_parity(root):
    rows = small_matrix(root, "rank-parity")
    footprint = next(iter(rows.values()))["footprint"]
    fp = max(r["pre_round_algebra_max"] for r in rows.values())
    p99 = max(r["bf16_logit_p99"] for r in rows.values())
    think = max(r["bf16_think_abs"] for r in rows.values())
    passed = fp <= .02 and p99 <= .5 and think <= 1. and footprint["random"]["matched"]
    result = {"passed": passed, "rows": rows, "footprint": footprint, "algebra_max": fp,
              "p99_max": p99, "think_max": think, "tolerances": [.02, .5, 1.]}
    save(root / "rank-parity-summary.json", result)
    if passed:
        d, v = load(root / "direction.json"), load(root / "fit.json")
        partial = {"d": d["direction"], "v": v["v"], "support": v["support"], "block": 3,
                   "formula": "down_proj BF16(FP32(weight) + outer(d,v))", "footprint": footprint,
                   "direction_sha256": sha(root / "direction.json"), "fit_sha256": sha(root / "fit.json")}
        save(root / "delta-factor.json", partial)


def evaluate_predictions(root):
    r = load(root / "rank-eval-summary.json")
    fit = load(root / "fit.json")
    _, _, raw = read_matrix(root, "rank-eval")
    simple = fit["simple_baseline"]
    by = {}
    for p, q in r["labels"].items():
        score = raw[p, 0, "gate_last"]["coefficient"] / fit["dose"]
        features = simple_features(q)
        guess = simple["coef"][0] + sum(w * (x - mu) / std for x, mu, std, w in
                                        zip(features, simple["mean"], simple["std"], simple["coef"][1:], strict=True))
        by[p] = {"observed_delta": q["delta"], "gate": score, "logodds_length_detector": guess}
    mse = {m: statistics.fmean((x[m] - max(0., x["observed_delta"])) ** 2 for x in by.values())
           for m in ("gate", "logodds_length_detector")}
    save(root / "rank-prediction-summary.json", {"by_prompt": by, "mse_positive_advantage": mse})


def summarize_phase(root, phase):
    if phase == "match":
        choose_bias(root)
    elif phase == "rank-parity":
        summarize_parity(root)
    else:
        contrasts = {
            "rank-eval": (("gate_last", "native"), ("gate_last", "bias"), ("gate_last", "fixed"),
                          ("fixed", "gaussian"), ("e0", "e32")),
            "rank-all": (("gate_all", "gate_last"), ("gate_all", "native"), ("gate_all", "bias")),
            "rank-edit": (("edit", "native"), ("edit", "gate_all"), ("edit", "random_edit"), ("edit", "bias")),
            "rank-full": (("edit", "native"), ("edit", "random_edit"), ("edit", "bias")),
            "rank-copy": (("copy_edit", "copy_base"), ("copy_hook", "copy_base"), ("copy_edit", "copy_random")),
        }[phase]
        summarize(root, phase, contrasts)
        if phase == "rank-eval":
            evaluate_predictions(root)
        elif phase == "rank-full":
            h, _, r = read_matrix(root, phase)
            by = {a: sum(row["periodic_onset"] is not None for row in r.values() if row["arm"] == a) for a in h["arms"]}
            save(root / "rank-entry-summary.json", {"periodic_entry": by, "per_arm": h["total"] * h["draws"]})


def continue_ridge(queue):
    worker = queue.src / "repetition_rank_eval.py"
    queue.cpu(queue.src / "repetition_rank_fit.py", "fit", "--root", queue.root)
    fit = load(queue.root / "fit.json")
    if not fit["positive_rows"]:
        queue.status(ridge_gate=False, ridge_reason="no positive development labels")
        queue.stage("labels-eval", max_reserve=700)
        return
    if not queue.stage("match", worker=worker, max_reserve=1500):
        queue.stage("labels-eval", max_reserve=700)
        return
    replay = queue.stage("rank-eval", worker=worker, max_reserve=1400)
    if not replay:
        queue.stage("labels-eval", max_reserve=700)
        return
    part = replay["strata"]["all"]
    a, b, n = (part["arms"][x] for x in ("gate_last", "bias", "native"))
    rate_gap = (a["think_at_zero"] - b["think_at_zero"]) / part["per_arm"]
    selectivity = a["success"] > max(b["success"], n["success"]) and abs(rate_gap) <= .125
    queue.status(ridge_gate=selectivity, heldout_close_rate_gap=rate_gap)
    if not selectivity:
        return
    full = queue.stage("rank-all", worker=worker, max_reserve=1200)
    if not full:
        return
    s = full["strata"]["all"]["arms"]
    preserved = s["gate_all"]["success"] >= s["native"]["success"] and (
        s["gate_all"]["wrong_parsed_eos"] <= s["gate_last"]["wrong_parsed_eos"] + 3)
    queue.status(all_position_gate=preserved)
    if not preserved:
        return
    parity = queue.stage("rank-parity", worker=worker, max_reserve=800)
    if not parity or not parity["passed"]:
        return
    if not queue.stage("rank-edit", worker=worker, max_reserve=1100):
        return
    if load(queue.root / "copy-dev-summary.json")["copy_selection"] is not None:
        queue.stage("rank-copy", worker=worker, max_reserve=480)
    queue.stage("rank-full", worker=worker, max_reserve=1900)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("run", "summarize"))
    p.add_argument("--phase", choices=PHASES, required=True)
    p.add_argument("--part", type=int)
    p.add_argument("--partitions", type=int)
    p.add_argument("--deadline", type=int)
    for k in ("output", "root", "model", "data"):
        p.add_argument("--" + k, type=Path)
    args = p.parse_args()
    if args.action == "run":
        run(args)
    else:
        summarize_phase(args.root, args.phase)


if __name__ == "__main__":
    main()
