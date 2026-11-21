"""Probe native-context actuator reachability and compare matched first-token policies."""
import argparse
from collections import defaultdict
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import statistics
import time

from repetition_baseline import sha
from repetition_canonical import seed_for
from repetition_canonical_queue import paired_interval
from repetition_exit_localize import capture, decoder_layers, hidden, replace_first_position
from repetition_mediation import INSERTIONS
from repetition_mediation_queue import load, save
from repetition_rank_one import gaussian
from repetition_replay import with_newline_tags
from repetition_sampler import BOUND_TOL, Boundary, match_marker, temperature_scores
from repetition_state_data import THINK_END
from repetition_state_value import Runtime, delivery_header, inputs_for, post_seed, read_matrix, summarize

PHASES = ("probe-dev", "higher-dev", "oracle-dev", "probe-eval", "replay")
INSERTED = 41
CAP = 512
DRAWS = 4
BLOCK = 3
LATER = 11
ARM_SEPARATOR = ":"
MIN_DEV_SUPPORT = 3
MIN_DEV_MEAN_Q = .05
ORACLE_SCALES = (.25, .5, 1.)


def arm_name(layer, name):
    return f"{layer}{ARM_SEPARATOR}{name}"


def pack_at(runtime, parts, width):
    packed = runtime.pack(parts)
    pad = width - packed["input_ids"].shape[1]
    fn = runtime.torch.nn.functional.pad
    return {"input_ids": fn(packed["input_ids"], (pad, 0), value=runtime.pad),
            "attention_mask": fn(packed["attention_mask"], (pad, 0), value=0)}


def id_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def stats(runtime, boundary, logits, items):
    logits = logits.float()
    scores = temperature_scores(logits, items, runtime.torch)
    rows = boundary.measures(logits, scores)
    for row, raw in zip(rows, logits, strict=True):
        values, tokens = raw.topk(8)
        row["top_logits"] = list(zip(tokens.tolist(), values.tolist(), strict=True))
    return rows


def parity(runtime, packed, items, logits, measures):
    width = packed["input_ids"].shape[-1]
    seeds = [seed_for(x["row"]["id"], 0) for x in items]
    post = [post_seed(x["row"]["id"], 0) for x in items]
    worker = runtime.policy(items, width, seeds, post, None, 0.)
    worker(packed["input_ids"], logits.float())
    deltas = [abs(m["q"] - q[2]) for m, q in zip(measures, worker.first_step["think"], strict=True)]
    if max(deltas) > 2e-7:
        raise ValueError("prefill sampler disagrees with the production harness")
    return deltas


def probe_specs(phase, plan, final):
    if phase == "probe-dev":
        return (BLOCK,), False, True
    if phase == "higher-dev":
        return (LATER, final), False, False
    if phase == "oracle-dev":
        return (), True, False
    tested = (BLOCK, LATER, final) if plan["higher"] else (BLOCK,)
    return tested, plan["oracle"], True


def donor_arms(layer, original, directions, group, short, torch):
    n = original["native"]
    blank, left = original["blank"], original["left"]
    kind = {
        "left_to_blank": ("blank", left),
        "left_to_native": ("native", left),
        "blank_to_native": ("native", blank),
        "same_delta_native": ("native", (n.float() + left.float() - blank.float()).to(n.dtype)),
    }
    seeds = [seed_for(x["row"]["id"], 0) for x in group]
    if layer == BLOCK and short:
        kind["old_fixed"] = "native", (n.float() + directions).to(n.dtype)
        kind["old_gaussian"] = "native", (n.float() + gaussian(directions, seeds, torch)).to(n.dtype)
        # Other-state donor direction uses the same per-state norm as the full donor.
        delta = left.roll(1, dims=0).float() - n.float()
        own_norm = (left.float() - n.float()).norm(dim=-1, keepdim=True)
        other = n.float() + delta * own_norm / delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        kind["other_donor"] = "native", other.to(n.dtype)
    else:
        own = left.float() - n.float()
        normed = gaussian(torch.ones_like(own[0]), seeds, torch)
        noise = normed / normed.norm(dim=-1, keepdim=True).clamp_min(1e-12) * own.norm(dim=-1, keepdim=True)
        kind["donor_gaussian"] = "native", (n.float() + noise).to(n.dtype)
    return kind


def measure_patch(runtime, boundary, layer, packed, target, source, group, stats_base):
    with replace_first_position(layer, target):
        raw, _ = capture(runtime.model, packed, {}, use_cache=True)
    measured = stats(runtime, boundary, raw, group)
    norms = (target.float() - source.float()).norm(dim=-1).tolist()
    for row, norm, base in zip(measured, norms, stats_base, strict=True):
        row.update(norm=norm, reference_q=base["q"], log_odds_change=row["log_odds"] - base["log_odds"])
    return measured


def oracle_gradient(runtime, module, packed, group):
    torch = runtime.torch
    leaf = []

    def replace(_module, _args, output):
        base = hidden(output)
        h = base[:, -1].detach().float().requires_grad_()
        leaf.append(h)
        mixed = torch.cat((base[:, :-1], h[:, None].to(base.dtype)), dim=1)
        return (mixed, *output[1:]) if isinstance(output, tuple) else mixed

    with torch.enable_grad():
        handle = module.register_forward_hook(replace)
        try:
            raw = runtime.model(**packed, use_cache=False, logits_to_keep=1).logits[:, -1]
            adjusted = temperature_scores(raw, group, torch)
            rest = adjusted.clone()
            rest[:, THINK_END] = -torch.inf
            objective = adjusted[:, THINK_END] - rest.logsumexp(-1)
            if len(leaf) != 1:
                raise ValueError("oracle patch was not applied at exactly one prefill")
            (g,) = torch.autograd.grad(objective.sum(), leaf)
        finally:
            handle.remove()
    return g.detach()


def bounded_cast(h, delta, bound, torch):
    lo = torch.zeros_like(delta[:, :1])
    hi = torch.ones_like(lo)
    normal = (h.float() + delta).to(h.dtype)
    if bool(((normal.float() - h.float()).norm(dim=-1) <= bound).all()):
        return normal
    for _ in range(14):
        weight = (lo + hi) / 2
        candidate = (h.float() + weight * delta).to(h.dtype)
        ok = (candidate.float() - h.float()).norm(dim=-1, keepdim=True) <= bound
        lo, hi = torch.where(ok, weight, lo), torch.where(ok, hi, weight)
    return (h.float() + lo * delta).to(h.dtype)


def prefill_group(runtime, boundary, layers, group, phase, plan, direction):
    torch = runtime.torch
    final = len(layers) - 1
    indices = (BLOCK, LATER, final)
    if len(set(indices)) != 3:
        raise ValueError("unexpected Qwen block count")
    pack_ids = {"native": [x["native_ids"] for x in group]}
    inserted = [with_newline_tags(runtime.tokenizer, ids, INSERTED, INSERTIONS) for ids in pack_ids["native"]]
    pack_ids.update({name: [x[name] for x in inserted] for name in ("blank", "left")})
    width = max(map(len, pack_ids["blank"]))
    packs = {k: pack_at(runtime, v, width) for k, v in pack_ids.items()}
    requested = {i: layers[i] for i in indices}
    base, state, parity_diff = {}, {}, []
    for k, packed in packs.items():
        raw, state[k] = capture(runtime.model, packed, requested, use_cache=True)
        base[k] = stats(runtime, boundary, raw, group)
        parity_diff.extend(parity(runtime, packed, group, raw, base[k]))
    tested, oracle, short = probe_specs(phase, plan, final)
    rows = [{"kind": "probe", "id": x["row"]["id"], "stratum": x["stratum"],
             "eval_origin": x.get("eval_origin"), "render_sha256": x["render_sha256"], "batch_width": width,
             "prefix_sha256": {k: id_hash(v[i]) for k, v in pack_ids.items()},
             "base": {k: v[i] for k, v in base.items()}, "patches": {},
             "last_native_donor_norm": float((state["left"][final][i].float()
                                             - state["native"][final][i].float()).norm()),
             "sampler_parity_max": max(parity_diff)} for i, x in enumerate(group)]
    def add(layer, name, receiver, values):
        result = measure_patch(runtime, boundary, layers[layer], packs[receiver], values,
                               state[receiver][layer], group, base[receiver])
        for i, row in enumerate(rows):
            applied = result[i]
            applied["receiver"] = receiver
            if receiver == "native":
                applied["replacement"] = values[i].float().tolist()
            if name == "other_donor":
                applied["other_id"] = group[i-1]["row"]["id"]
            row["patches"][arm_name(layer, name)] = applied

    for layer in tested:
        original = {k: v[layer] for k, v in state.items()}
        for name, (receiver, values) in donor_arms(layer, original, direction, group, short, torch).items():
            add(layer, name, receiver, values)
    if oracle:
        bound = plan["oracle_norm"]
        grad = oracle_gradient(runtime, layers[final], packs["native"], group)
        if not bool(torch.isfinite(grad).all()):
            raise ValueError("nonfinite gradient in last-block oracle")
        need = torch.tensor([x["b_half"][1] for x in base["native"]], device=grad.device)
        delta = need[:, None] * grad / (grad.square().sum(-1, keepdim=True) + 1e-12)
        cap_ratio = (bound / delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp_max(1.)
        delta *= cap_ratio
        h = state["native"][final]
        for scale in ORACLE_SCALES:
            name = f"oracle_{scale:g}"
            add(final, name, "native", bounded_cast(h, scale * delta, bound, torch))
            for i, row in enumerate(rows):
                row["patches"][arm_name(final, name)].update(
                    norm_bound=bound, was_clipped=bool(cap_ratio[i] < 1), grad_norm=float(grad[i].norm()))
    return rows


def iter_probe(root, phase):
    receipt = load(root / "queue" / f"{phase}-done.json")
    ids, seen, common = set(), set(), None
    for path, digest in receipt["outputs"]:
        path = Path(path)
        if sha(path) != digest:
            raise ValueError("reachability shard SHA mismatch")
        with path.open() as stream:
            h = json.loads(next(stream))
            contract = {k: v for k, v in h.items() if k not in ("part", "cases")}
            if common not in (None, contract) or h["phase"] != phase or h["part"] in seen:
                raise ValueError("invalid reachability shard header")
            common = contract
            seen.add(h["part"])
            pane = {r["id"]: r for r in h["cases"]}
            if pane.keys() & ids:
                raise ValueError("overlapping reachability shards")
            ids.update(pane)
            done = False
            for line in stream:
                r = json.loads(line)
                if r["kind"] == "complete":
                    done = True
                    continue
                item = pane.pop(r["id"])
                if item["render_sha256"] != r["render_sha256"]:
                    raise ValueError("rendered prompt identity changed")
                yield r
            if pane or not done:
                raise ValueError("incomplete reachability shard")
    if len(ids) != common["total"] or seen != set(range(common["partitions"])):
        raise ValueError("incomplete reachability matrix")


def candidate_stats(rows):
    patches = defaultdict(list)
    for row in rows:
        for arm, values in row["patches"].items():
            if values["receiver"] == "native" and not any(c in arm for c in ("gaussian", "other_donor")):
                patches[arm].append(values)
    stats = {}
    for k, observations in patches.items():
        baseline = sum(v["reference_q"] > 0 for v in observations)
        improved = sum(v["q"] > 0 and v["reference_q"] == 0 for v in observations)
        stats[k] = {"support": sum(v["in_p"] for v in observations), "native_support": baseline,
                    "new_support": improved, "mean_q": statistics.fmean(v["q"] for v in observations),
                    "norm": statistics.median(v["norm"] for v in observations)}
    return stats


def choose(rows):
    diagnostic = candidate_stats(rows)
    good = [a for a, s in diagnostic.items() if s["new_support"] >= MIN_DEV_SUPPORT and s["mean_q"] >= MIN_DEV_MEAN_Q]
    picked = sorted(good, key=lambda a: (-diagnostic[a]["new_support"], -diagnostic[a]["mean_q"],
                                         diagnostic[a]["norm"], a))
    return picked, diagnostic


def report_probe(root, phase):
    rows = list(iter_probe(root, phase))
    subsets = {"all": rows}
    for k in sorted({r["stratum"] for r in rows}):
        subsets[k] = [r for r in rows if r["stratum"] == k]
    report = {}
    for name, group in subsets.items():
        report[name] = {"prompts": len(group), "arms": {}}
        arms = {**group[0]["base"], **group[0]["patches"]}
        for arm in arms:
            obs = [(r["base"] | r["patches"])[arm] for r in group]
            stats = {f"{k}_median": statistics.median(r[k] for r in obs) for k in
                     ("raw_logit", "adjusted_logit", "log_odds", "rank", "topk_gap", "p_before", "p_after_k", "q")}
            stats.update({k: sum(r[k] for r in obs) for k in ("in_k", "in_p")})
            stats.update(mean_q=statistics.fmean(r["q"] for r in obs), ties=sum(r["ties"] > 0 for r in obs),
                         b_support_median=statistics.median(r["b_support"][1] for r in obs),
                         b_half_median=statistics.median(r["b_half"][1] for r in obs),
                         b_support_q_median=statistics.median(r["b_support"][3] for r in obs),
                         b_half_q_median=statistics.median(r["b_half"][3] for r in obs))
            if "norm" in obs[0]:
                stats["median_norm"] = statistics.median(r["norm"] for r in obs)
                stats["paired_q"] = paired_interval({r["id"]: [v["q"] - v["reference_q"]] for r, v in zip(group, obs)})
            report[name]["arms"][arm] = stats
    save(root / f"{phase}-summary.json",
         {"strata": report, "bisection_tolerance": BOUND_TOL, "candidate_stats": candidate_stats(rows),
          "max_harness_q_error": max(r["sampler_parity_max"] for r in rows)})


def replay_group(runtime, layers, group, probe, picked, draw):
    torch = runtime.torch
    expected = [probe[x["row"]["id"]] for x in group]
    padded_widths = {r["batch_width"] for r in expected}
    if len(padded_widths) != 1 or any(id_hash(x["native_ids"]) != r["prefix_sha256"]["native"]
                                     for x, r in zip(group, expected)):
        raise ValueError("prefill and replay differ on native receiver")
    packed = pack_at(runtime, [x["native_ids"] for x in group], padded_widths.pop())
    def run(arm, hook=nullcontext(), transform=None):
        return runtime.generate(group, arm, draw, CAP, hook, packed=packed,
                                first_transform=transform, matched_budget=True)

    for arm in ("native", "e0"):
        yield from run(arm)
    def decoder(q, label):
        fallback = []
        def transform(probs, base):
            changed, cases = match_marker(probs, base, THINK_END, q, torch)
            fallback.extend(cases)
            return changed
        rows = run(label, transform=transform)
        for i, row in enumerate(rows):
            if abs(q[i] - row["first_think_mass"][2]) > 2e-7:
                raise ValueError("decoder control did not match activation marker mass")
            yield row | {"target_q": q[i], "fallback": fallback[i]}

    for a in picked:
        layer = int(a.split(ARM_SEPARATOR, 1)[0])
        measurements = [r["patches"][a] for r in expected]
        replacement = torch.tensor([r["replacement"] for r in measurements], dtype=torch.bfloat16, device="cuda")
        lifted = run(f"act:{a}", replace_first_position(layers[layer], replacement))
        q = [r["first_think_mass"][2] for r in lifted]
        for i, row in enumerate(lifted):
            yield row | {"measured_q": measurements[i]["q"], "prefill_q_error": q[i] - measurements[i]["q"],
                         "activation_norm": measurements[i]["norm"]}
        yield from decoder(q, f"match:{a}")
    yield from decoder([.5] * len(group), "output_half")


def run(args):
    root = args.root
    cohort_path = root / "cohort.json"
    cohort = load(cohort_path)
    plan_path = root / "decision.json"
    plan = load(plan_path) if plan_path.exists() else {}
    if args.phase == "oracle-dev":
        plan = load(root / "oracle-budget.json")
    panel = "eval" if args.phase in ("probe-eval", "replay") else "dev"
    cases = cohort[panel]
    if args.smoke:
        cases = cases[:3]
    runtime = Runtime(args)
    torch = runtime.torch
    runtime.model.requires_grad_(False)
    layers = decoder_layers(runtime.model, torch)
    boundary = Boundary(torch, runtime.transformers)
    prior = root / "old-direction.json"
    old = load(prior)
    d = torch.tensor(old["direction"], dtype=torch.float32, device="cuda") * old["reference_norm"]
    items = inputs_for(runtime.tokenizer, cases[args.part::args.partitions])
    arms = ["native", "e0"] + [f"{kind}:{a}" for a in plan.get("picked", []) for kind in ("act", "match")] + ["output_half"]
    replay = args.phase == "replay"
    header = delivery_header(runtime, args, items, arms if replay else (), CAP if replay else 0,
                             DRAWS if replay else 0, len(cases), sha(cohort_path))
    header.update(source_commit=args.base_commit, reach_sha256=sha(Path(__file__)),
                  old_direction_sha256=sha(prior), plan_sha256=sha(plan_path) if plan_path.exists() else None,
                  layer_count=len(layers), batch_size=3, tolerance=BOUND_TOL,
                  shared_answer_budget=CAP if replay else None,
                  separate_reasoning_cap=CAP if replay else None)
    start = time.monotonic()
    with args.output.open("x") as stream, torch.no_grad():
        def emit(record):
            print(json.dumps(record, allow_nan=False), file=stream)
        emit(header)
        lookup = {r["id"]: r for r in iter_probe(root, "probe-eval")} if replay else None
        for draw in range(DRAWS if replay else 1):
            for begin in range(0, len(items), 3):
                if time.time() >= args.deadline - 60:
                    raise TimeoutError("reachability hard deadline")
                group = items[begin:begin + 3]
                if replay:
                    records = replay_group(runtime, layers, group, lookup, plan["picked"], draw)
                else:
                    records = prefill_group(runtime, boundary, layers, group, args.phase, plan, d)
                for row in records:
                    emit(row)
                stream.flush()
                print(f"{args.phase} draw={draw} prompts={begin + len(group)}/{len(items)}", flush=True)
        emit({"kind": "complete", "seconds": time.monotonic() - start})


def report_replay(root):
    plan = load(root / "decision.json")
    contrast = [("e0", "native"), ("output_half", "native")]
    for a in plan["picked"]:
        contrast.extend(((f"act:{a}", f"match:{a}"), (f"act:{a}", "native"), (f"match:{a}", "native")))
    summarize(root, "replay", contrast)
    h, _, rows = read_matrix(root, "replay")
    status = {a: {"mean_first_q": 1. if a == "e0" else statistics.fmean(
                      r["first_think_mass"][2] for r in rows.values() if r["arm"] == a),
                  "fallbacks": sum(bool(r.get("fallback")) for r in rows.values() if r["arm"] == a),
                  "max_prefill_q_error": max((abs(r.get("prefill_q_error", 0.)) for r in rows.values()
                                              if r["arm"] == a), default=0.)} for a in h["arms"]}
    save(root / "replay-q-diagnostic.json", status)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("run", "summarize"))
    p.add_argument("--phase", choices=PHASES, required=True)
    for name in ("root", "model", "data", "output"):
        p.add_argument("--" + name, type=Path)
    for name in ("part", "partitions", "deadline"):
        p.add_argument("--" + name, type=int)
    p.add_argument("--base-commit", default="d03d487bf8e679b21c3c52d85dc6905ca12dfd03")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.action == "run":
        run(args)
    elif args.phase == "replay":
        report_replay(args.root)
    else:
        report_probe(args.root, args.phase)


if __name__ == "__main__":
    main()
