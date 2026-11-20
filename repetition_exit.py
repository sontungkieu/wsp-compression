"""Measure whether state-gated Qwen reasoning exits preserve useful continuation."""
import argparse
from collections import Counter, defaultdict
import inspect
import json
from pathlib import Path
import random
import socket
import statistics
import time
from contextlib import nullcontext

from repetition_baseline import assess, render, sha
from repetition_exit_localize import capture, decoder_layers, random_replacement, replace_first_position, scan
from repetition_mediation import INSERTIONS, SAMPLING, identity, processor_factory, token_groups
from repetition_mediation_queue import load, save
from repetition_replay import terminal_cycle, with_newline_tags

DOSES = (.75, 1.5, 3.)
LAYERS = (0, 3, 6, 9, 12, 15, 18, 19, 21, 23)
CAPS = {"dev": 512, "localize": 0, "heldout": 512, "long": 2048, "patch": 512}
DRAWS = {"dev": 4, "localize": 0, "heldout": 6, "long": 4, "patch": 4}
CORE = ("native", "blank", "left", "think_transfer")
TREATMENTS = ("static", "fixed_pulse", "periodic_pulse")
PATCHES = ("patch_forward", "patch_reverse", "patch_random")
METRICS = ("success", "correct", "eos", "wrong_eos", "terminal_cycle", "thought_closed", "cap", "gate_fired")
PULSE = 32
WINDOW = 256


def copy_cases():
    rng = random.Random(41025)
    cases = []
    for i in range(32):
        nonce = "".join(rng.choices("bcdfghjklmnprstvz", k=7))
        copies = (4, 6, 9, 12)[i % 4]
        row = {"id": f"copy-exit:{i}", "task": "copy", "gold": " | ".join([nonce] * copies),
               "prompt": f"Write {nonce} exactly {copies} times. Separate copies with space, vertical bar, space. "
                         "In your final answer write only the requested string."}
        cases.append({"row": row, "completion_prefix": [], "cycle": [], "period": 0,
                      "origin": "synthetic-nonce-v1", "stratum": "copy", "budget": 41})
    return cases


def cohort(args):
    discovery = args.phase in ("dev", "localize")
    path = args.old / "holdout.json" if discovery else args.root / "cohort.json"
    rows = load(path)["cases"]
    if discovery:
        if len(rows) != 64:
            raise ValueError("valid development cohort must have 64 old loop states")
        rows = [{**row, "stratum": "loop", "budget": 41} for row in rows]
    else:
        rows += copy_cases()
    limits = {"long": {"loop": 48, "clean": 24, "copy": 16},
              "patch": {"loop": 40, "clean": 16, "copy": 8}}
    if args.phase in limits:
        count = Counter()
        selected = []
        for row in rows:
            key = row["stratum"]
            if count[key] < limits[args.phase][key]:
                selected.append(row)
                count[key] += 1
        rows = selected
    if len({r["row"]["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate prompt in cohort")
    return rows[args.part::args.partitions], len(rows), sha(path)


def inputs_for(tokenizer, cases, think_token):
    items = []
    for case in cases:
        row = case["row"]
        prior = case["completion_prefix"]
        prefix = tokenizer.encode(render(tokenizer, row, "think"), add_special_tokens=False) + prior
        if think_token in prior:
            raise ValueError("expected a prefix still inside reasoning")
        inputs = {"native": prefix, **with_newline_tags(tokenizer, prefix, case["budget"], INSERTIONS)}
        items.append({**case, "inputs": inputs, "pending": True,
                      "wait_loop": False})
    return items


def padded(parts, pad, torch):
    width = max(map(len, parts))
    return {"input_ids": torch.tensor([[pad] * (width - len(p)) + p for p in parts], device="cuda"),
            "attention_mask": torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in parts], device="cuda")}


def periodic_suffix(tail, lengths, torch):
    """Four exact trailing repeats, span at least 24 tokens, periods 1 through 64."""
    lags = torch.arange(1, 65, device=tail.device)[:, None]
    recent = torch.arange(192, device=tail.device)[None, :]
    span = (4 * lags).clamp_min(24)
    required = recent < span - lags
    left = (WINDOW - 1 - recent - lags).clamp_min(0)
    tip = tail[:, WINDOW - 1 - recent[0]][:, None, :]
    matches = tail[:, left] == tip
    return ((matches | ~required).all(-1) & (lengths[:, None] >= span[:, 0])).any(-1)


def make_policy(torch, transformers, groups):
    base = processor_factory(torch, transformers, groups)
    token = groups["think"][0]

    class ExitPolicy(base):
        def __init__(self, batch, width, mode, amount, seed):
            super().__init__(batch, width, ("think",) if mode == "think_transfer" else (), seed)
            self.previous = [values.long() for values in self.previous]
            self.mode = mode
            self.amount = amount
            self.trigger = torch.full((self.n,), -1, dtype=torch.long, device="cuda")
            self.prior = torch.tensor(
                [[-1] * max(0, WINDOW - len(p := x["completion_prefix"])) + p[-WINDOW:] for x in batch],
                dtype=torch.long, device="cuda")
            self.lengths = torch.tensor([len(x["completion_prefix"]) for x in batch], device="cuda")

        def __call__(self, ids, scores):
            if self.amount:
                history = ids[:self.n, self.width:]
                step = history.shape[-1]
                pending = ~(history == token).any(-1)
                if self.mode == "periodic_pulse":
                    if step % 4 == 0:
                        tail = torch.cat((self.prior, history[:, -WINDOW:]), -1)[:, -WINDOW:]
                        fired = pending & (self.trigger < 0) & periodic_suffix(tail, self.lengths + step, torch)
                        self.trigger = torch.where(fired, step, self.trigger)
                    pending &= (self.trigger >= 0) & ((step - self.trigger) < PULSE)
                elif self.mode == "fixed_pulse":
                    pending &= step < PULSE
                scores = scores.float().clone()
                scores[:, token] += self.amount * pending
            return super().__call__(ids, scores)

    return ExitPolicy


def arms_for(args):
    if args.phase == "dev":
        return {**dict.fromkeys(CORE, 0.), **{f"bias_{dose:g}": dose for dose in DOSES}}, None
    if args.phase == "patch" or args.smoke == "patch":
        layer = 15 if args.smoke else load(args.root / "localize-summary.json")["selected_layer"]
        return dict.fromkeys(("blank", "left", *PATCHES), 0.), layer
    dose = 1.5 if args.smoke else load(args.root / "dev-summary.json")["selected_dose"]
    core = CORE if args.phase != "long" else ("native", "blank", "left")
    return {**dict.fromkeys(core, 0.), **dict.fromkeys(TREATMENTS, dose)}, None


def patch_context(arm, model, layer_map, batch, pad, seed, torch):
    if arm not in PATCHES:
        return nullcontext()
    receiver = "left" if arm == "patch_reverse" else "blank"
    donor = "blank" if arm == "patch_reverse" else "left"
    idx, module = next(iter(layer_map.items()))
    donor_inputs = padded([x["inputs"][donor] for x in batch], pad, torch)
    _, d = capture(model, donor_inputs, layer_map)
    replacement = d[idx]
    if arm == "patch_random":
        receiver_inputs = padded([x["inputs"][receiver] for x in batch], pad, torch)
        _, base = capture(model, receiver_inputs, layer_map)
        replacement = random_replacement(base[idx], replacement, seed + 9287, torch)
    return replace_first_position(module, replacement)


def run(args):
    import torch
    import transformers
    import tokenizers

    if torch.cuda.device_count() != 1:
        raise ValueError("one CUDA device required")
    identity(args)
    helper = Path(__file__).with_name("repetition_exit_localize.py")
    helper_sha = sha(helper)
    if not args.smoke and load(helper.with_name("exit-ready.json"))["helper_sha256"] != helper_sha:
        raise ValueError("localization source digest changed")
    if args.smoke in ("holdout", "patch"):
        args.phase = "dev"
        selected, _, fingerprint = cohort(args)
        selected = selected[:1] + copy_cases()[:1]
        total = 2
        args.phase = "patch" if args.smoke == "patch" else "heldout"
    else:
        selected, total, fingerprint = cohort(args)
        if args.smoke:
            selected, total = selected[:2], 2
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    groups = token_groups(tokenizer)
    think = groups["think"][0]
    if think != 248069:
        raise ValueError("thinking delimiter differs from cohort construction")
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16).to("cuda:0").eval()
    defaults = {"use_model_defaults": False} if "use_model_defaults" in inspect.signature(model.generate).parameters else {}
    end = model.generation_config.eos_token_id
    eos = set(end if isinstance(end, (list, tuple)) else [end])
    if None in eos:
        raise ValueError("model lacks EOS")
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else min(eos)
    items = inputs_for(tokenizer, selected, think)
    for item in items:
        item["wait_loop"] = bool(set(item["cycle"]) & set(groups["wait"]))
    role = args.phase
    cap, n_draws = (12, 1) if args.smoke else (CAPS[role], DRAWS[role])
    if role == "localize":
        arms, layer = {}, None
        layer_ids = (0, 15, 23) if args.smoke else LAYERS
        all_layers = decoder_layers(model, torch)
        layer_map = {idx: all_layers[idx] for idx in layer_ids}
    else:
        arms, layer = arms_for(args)
        layer_map = {} if layer is None else {layer: decoder_layers(model, torch)[layer]}
    current = {"host": socket.gethostname(), "torch": str(torch.__version__),
               "transformers": transformers.__version__, "tokenizers": tokenizers.__version__}
    header = {"kind": "manifest", "phase": role, "part": args.part, "partitions": args.partitions,
              "total": total, "input_sha256": fingerprint, "source_sha256": sha(Path(__file__)),
              "helper_sha256": helper_sha, "sampling": SAMPLING, "cap": cap, "draws": n_draws, "arms": arms,
              "layer": layer, "runtime": current,
              "cases": [{k: x[k] for k in ("stratum", "wait_loop", "origin")} | {"id": x["row"]["id"]} for x in items]}
    started = time.monotonic()
    with args.output.open("x") as out, torch.inference_mode():
        def write(record):
            print(json.dumps(record, allow_nan=False), file=out)
        write(header)
        if role == "localize":
            for i in range(0, len(items), 4):
                if time.time() >= args.deadline - 60:
                    raise TimeoutError("localization deadline")
                batch = items[i:i + 4]
                packed = {k: padded([x["inputs"][k] for x in batch], pad, torch) for k in ("blank", "left", "round")}
                odds, deltas = scan(model, packed, layer_map, think, [x["completion_prefix"] for x in batch], torch)
                raw = {a: t.tolist() for a, t in odds.items()}
                patch = {k: {a: v.tolist() for a, v in s.items()} for k, s in deltas.items()}
                for j, row in enumerate(batch):
                    write({"kind": "trace", "id": row["row"]["id"], "odds": {k: v[j] for k, v in raw.items()},
                           "deltas": {k: {a: v[j] for a, v in s.items()} for k, s in patch.items()}})
                out.flush()
                print(f"localize prompts={i + len(batch)}/{len(items)}", flush=True)
        else:
            policy = make_policy(torch, transformers, groups)
            cfg = transformers.GenerationConfig(do_sample=False, max_new_tokens=cap, eos_token_id=sorted(eos),
                                                pad_token_id=pad, use_cache=True)
            for draw in range(n_draws):
                for i in range(0, len(items), 8):
                    batch = items[i:i + 8]
                    rng = 71231 + 100000 * args.part + 1000 * draw + i
                    for arm, amount in arms.items():
                        if time.time() >= args.deadline - 60:
                            raise TimeoutError("generation deadline")
                        receiver = arm if arm in CORE[:3] else "left" if arm == "patch_reverse" else "blank"
                        parts = [x["inputs"][receiver] for x in batch]
                        if arm == "think_transfer":
                            parts += [x["inputs"]["left"] for x in batch]
                        packed = padded(parts, pad, torch)
                        width = packed["input_ids"].shape[1]
                        processor = policy(batch, width, arm, amount, rng)
                        with patch_context(arm, model, layer_map, batch, pad, rng, torch):
                            result = model.generate(**packed, generation_config=cfg, logits_processor=[processor], **defaults)
                        tokens = result[:, width:].tolist()
                        if arm == "think_transfer" and tokens[:len(batch)] != tokens[len(batch):]:
                            raise ValueError("coupled donor and receiver diverged")
                        fired = processor.trigger.tolist()
                        for j, (case, raw) in enumerate(zip(batch, tokens[:len(batch)], strict=True)):
                            stop = next((k for k, t in enumerate(raw) if t in eos), None)
                            continuation = raw[:stop] if stop is not None else raw
                            text = tokenizer.decode(continuation, skip_special_tokens=False)
                            quality = assess(case["row"], text, continuation, stop is not None, cap, True)
                            repeating = terminal_cycle(continuation) is not None
                            write({"kind": "rollout", "id": case["row"]["id"], "arm": arm, "draw": draw,
                                   "success": quality["correct"] and quality["eos"] and not repeating,
                                   "correct": quality["correct"], "eos": quality["eos"],
                                   "wrong_eos": quality["eos"] and not quality["correct"],
                                   "terminal_cycle": repeating, "thought_closed": quality["think_closed"],
                                   "cap": quality["cap"], "gate_fired": fired[j] >= 0,
                                   "trigger_step": fired[j], "tokens": continuation,
                                   "first_think_mass": processor.first_step["think"][j]})
                    out.flush()
                    print(f"{role} draw={draw} prompts={i + len(batch)}/{len(items)}", flush=True)
        write({"kind": "complete", "seconds": time.monotonic() - started})


def result_matrix(args):
    phase = args.phase
    roles = ("mechanism",) if phase in ("dev", "localize") else ("collection", "mechanism")
    headers, cases, rows, parts = [], {}, {}, set()
    for role in roles:
        for name, checksum in load(args.root / role / f"{phase}-done.json")["outputs"]:
            path = Path(name)
            if sha(path) != checksum:
                raise ValueError("worker digest mismatch")
            with path.open() as stream:
                header = json.loads(next(stream))
                if header["part"] in parts:
                    raise ValueError("duplicate partition")
                parts.add(header["part"])
                ids = {x["id"]: x for x in header["cases"]}
                if ids.keys() & cases.keys():
                    raise ValueError("prompt partitions overlap")
                cases.update(ids)
                contract = {k: v for k, v in header.items() if k not in ("part", "cases", "runtime")}
                if headers and headers[0] != contract:
                    raise ValueError("worker contracts differ")
                headers.append(contract)
                seen = False
                for line in stream:
                    item = json.loads(line)
                    if item["kind"] == "complete":
                        seen = True
                        continue
                    key = (item["id"], item.get("draw"), item.get("arm"))
                    if key in rows or item["id"] not in ids:
                        raise ValueError("unexpected worker row")
                    rows[key] = item
                if not seen:
                    raise ValueError("worker missing completion footer")
    contract = headers[0]
    expected = ({(p, None, None) for p in cases} if phase == "localize" else
                {(p, draw, arm) for p in cases for draw in range(contract["draws"]) for arm in contract["arms"]})
    if rows.keys() != expected or len(cases) != contract["total"] or parts != set(range(contract["partitions"])):
        raise ValueError("incomplete experiment matrix")
    return contract, cases, rows


def summarize(args):
    contract, cases, rows = result_matrix(args)
    phase = args.phase
    result = {"protocol": contract, "rows": len(rows), "prompts": len(cases)}
    if phase == "localize":
        strata = {}
        for layer in LAYERS:
            point = {k: statistics.fmean(r["deltas"][str(layer)][k] for r in rows.values())
                     for k in ("forward", "reverse", "round")}
            point["symmetric"] = min(point["forward"], -point["reverse"])
            strata[layer] = point
        last = LAYERS[-1]
        reference = strata[last]
        admissible = [idx for idx in LAYERS[:-1]
                      if reference["symmetric"] > 0 and strata[idx]["forward"] >= .75 * reference["forward"]
                      and -strata[idx]["reverse"] >= -.75 * reference["reverse"]]
        winner = admissible[0] if admissible else last
        result.update(selected_layer=winner, layers=strata,
                      selection="earliest layer reaching 75% of the last-layer effect in both directions",
                      left_margin=statistics.fmean(r["odds"]["left"] - r["odds"]["blank"] for r in rows.values()))
    else:
        arms = contract["arms"]
        draws = contract["draws"]
        groups = defaultdict(list)
        for key, item in cases.items():
            category = item["stratum"]
            groups[category].append(key)
            if category == "loop":
                groups["loop_wait" if item["wait_loop"] else "loop_other"].append(key)
        rng = random.Random(83011)
        summary = {}
        for label, prompt_ids in groups.items():
            counts = {arm: {metric: sum(int(rows[(p, d, arm)][metric]) for p in prompt_ids for d in range(draws))
                            for metric in METRICS} for arm in arms}
            pairs = [(arm, "blank") for arm in arms if arm != "blank"]
            if "periodic_pulse" in arms:
                pairs += [("periodic_pulse", "fixed_pulse"), ("periodic_pulse", "static")]
            if "patch_forward" in arms:
                pairs += [("patch_forward", "patch_random"), ("left", "patch_reverse")]
            intervals = {}
            for a, b in pairs:
                metrics = {}
                for metric in ("success", "wrong_eos", "terminal_cycle"):
                    diff = [sum(int(rows[p, d, a][metric]) - int(rows[p, d, b][metric]) for d in range(draws))
                            * 100. / draws for p in sorted(prompt_ids)]
                    boot = sorted(statistics.fmean(rng.choices(diff, k=len(diff))) for _ in range(2000))
                    metrics[metric] = [round(statistics.fmean(diff), 2), round(boot[50], 2), round(boot[1949], 2)]
                intervals[f"{a}_minus_{b}"] = metrics
            summary[label] = {"prompts": len(prompt_ids), "total_per_arm": len(prompt_ids) * draws,
                              "arms": counts, "paired_pp": intervals}
        result["strata"] = summary
        if phase == "dev":
            ranked = summary["loop"]["arms"]
            result["selected_dose"] = max(DOSES, key=lambda d: (
                ranked[f"bias_{d:g}"]["success"], -ranked[f"bias_{d:g}"]["wrong_eos"], -d))
    save(args.root / f"{phase}-summary.json", result)


def self_test():
    import torch
    p = torch.tensor([[-1] * 224 + list(range(8)) * 4, list(range(256)), [7] * 256])
    verdict = periodic_suffix(p, torch.tensor([32, 256, 12]), torch)
    assert verdict.tolist() == [True, False, False]
    samples = copy_cases()
    assert len(samples) == 32 and len({r["row"]["id"] for r in samples}) == 32
    print("periodicity and copy-panel checks passed")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("run", "summarize", "self-test"))
    p.add_argument("--phase", choices=tuple(CAPS))
    p.add_argument("--part", type=int)
    p.add_argument("--partitions", type=int)
    p.add_argument("--deadline", type=int)
    p.add_argument("--smoke", choices=("dev", "holdout", "localize", "patch"))
    for key in ("root", "output", "old", "model", "data"):
        p.add_argument("--" + key, type=Path)
    args = p.parse_args()
    {"run": run, "summarize": summarize, "self-test": lambda _: self_test()}[args.action](args)


if __name__ == "__main__":
    main()
