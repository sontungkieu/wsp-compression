"""Compare one-token and trajectory gradients with exact column edits."""
import argparse
import json
import os
from pathlib import Path
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def examples(ids, start, count, period, rng, copy_instruction):
    width = 72
    if start + width * count > len(ids):
        raise ValueError('insufficient input tokens')
    clean, positive, copy, motifs = [], [], [], []
    for i in range(count):
        row = ids[start + i * width:start + (i + 1) * width]
        motif = row[-period:]
        repeated = row[:-3 * period] + motif * 3
        clean.append(row)
        positive.append(repeated)
        copy.append(copy_instruction + repeated[len(copy_instruction):])
        motifs.append(motif)
    order = list(range(count))
    rng.shuffle(order)
    return [[rows[i] for i in order] for rows in (clean, positive, copy, motifs)]


def run(args):
    if torch.cuda.device_count() != 1:
        raise SystemExit('requires exactly one assigned CUDA device')
    device = torch.device('cuda:0')
    started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.bfloat16).to(device).eval()
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    train = tokenizer(args.train.read_text(), add_special_tokens=False,
                      truncation=True, max_length=100000).input_ids
    test = tokenizer(args.test.read_text(), add_special_tokens=False,
                     truncation=True, max_length=60000).input_ids
    instruction = tokenizer('Continue copying this exact repeated sequence: ',
                            add_special_tokens=False).input_ids
    rng = random.Random(args.seed)
    _, calibration, _, motifs = examples(train, args.offset, args.rows, args.period, rng, instruction)
    test_start = args.test_offset % max(1, len(test) - 72 * args.test_rows + 1)
    clean, heldout, copy, test_motifs = examples(test, test_start, args.test_rows,
                                                args.period, rng, instruction)
    writers = [layer.mlp.down_proj for layer in model.model.layers]
    sizes = [module.in_features for module in writers]
    norms = torch.cat([module.weight.detach().float().norm(dim=0) for module in writers]).cpu()
    ends = model.generation_config.eos_token_id or tokenizer.eos_token_id
    ends = [ends] if isinstance(ends, int) else list(ends)
    temp = .8

    def array(rows):
        return torch.tensor(rows, device=device, dtype=torch.long)

    def hidden_logits(ids, positions):
        hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state
        return model.lm_head(hidden[:, positions]).float()

    def generate(rows, steps, seed, sample=True):
        torch.manual_seed(seed)
        config = GenerationConfig(
            max_new_tokens=steps, do_sample=sample, temperature=temp if sample else 1.,
            top_k=0 if sample else None, top_p=1., eos_token_id=ends,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else ends[0],
            use_cache=True,
        )
        with torch.inference_mode():
            pieces = []
            for i in range(0, len(rows), args.batch):
                xs = array(rows[i:i + args.batch])
                ys = model.generate(xs, attention_mask=torch.ones_like(xs), generation_config=config)
                suffix = ys[:, xs.size(1):].tolist()
                pieces.extend([row + [ends[0]] * (steps - len(row)) for row in suffix])
            return pieces

    def rewards(outputs, motifs):
        results = []
        for suffix, motif in zip(outputs, motifs):
            length = next((i + 1 for i, token in enumerate(suffix) if token in ends), len(suffix))
            seq, period = suffix[:length], len(motif)
            # Fixed loss denominator keeps the reward defined for stopped rollouts.
            matches = sum(t == motif[i % period] for i, t in enumerate(seq))
            repeats = sum(seq[i - 2 * period:i - period] == seq[i - period:i]
                          and seq[i] == seq[i - period] for i in range(2 * period, length))
            results.append({'aligned': matches / len(suffix), 'recurrence': repeats / len(suffix),
                            'motif_completed': seq[:period] == motif,
                            'length': length, 'stopped': seq[-1] in ends})
        return results

    def score(kind, suffixes=None, centered=None):
        gates = [torch.ones(size, device=device, requires_grad=True) for size in sizes]
        def hook(gate):
            def scale(_module, inputs):
                return (inputs[0] * gate.to(inputs[0].dtype),)
            return scale
        handles = [w.register_forward_pre_hook(hook(g)) for w, g in zip(writers, gates)]
        prompts = calibration if kind == 'one_token' else calibration * args.draws
        count = len(prompts)
        try:
            for start in range(0, count, args.grad_batch):
                end = min(start + args.grad_batch, count)
                batch = prompts[start:end]
                if kind == 'one_token':
                    z = hidden_logits(array(batch), slice(-1, None))[:, 0]
                    target = array([row[0] for row in motifs[start:end]])
                    loss = z.log_softmax(-1).gather(-1, target[:, None]).sum() / count
                else:
                    ys = array(suffixes[start:end])
                    full = torch.cat((array(batch), ys), dim=-1)
                    width = len(batch[0])
                    z = hidden_logits(full, slice(width - 1, -1)) / temp
                    logp = z.log_softmax(-1).gather(-1, ys[..., None]).squeeze(-1)
                    stop = torch.isin(ys, array(ends))
                    live = (stop.long().cumsum(-1) - stop.long()) == 0
                    loss = ((logp * live).sum(-1) * centered[start:end]).sum() / count
                loss.backward()
        finally:
            for handle in handles:
                handle.remove()
        return torch.cat([g.grad.detach().cpu() for g in gates])

    suffixes = [row for draw in range(args.draws)
                for row in generate(calibration, args.horizon, args.seed + 100 + draw)]
    trial = rewards(suffixes, motifs * args.draws)
    risk = torch.tensor([row[args.objective] for row in trial], device=device).reshape(args.draws, -1)
    # Leave-one-out baseline is independent of the scored sample for each prompt.
    centered = ((risk - risk.mean(0)) * args.draws / (args.draws - 1)).flatten()
    scores = {'one_token': score('one_token'), 'trajectory': score('trajectory', suffixes, centered)}
    if any(not torch.isfinite(s).all() for s in scores.values()):
        raise RuntimeError('non-finite score')
    selections = {k: (s / norms.clamp_min(1e-8)).topk(args.columns).indices for k, s in scores.items()}
    offsets = [0]
    for size in sizes:
        offsets.append(offsets[-1] + size)
    by_layer = [int(((selections['trajectory'] >= lo) & (selections['trajectory'] < hi)).sum())
                for lo, hi in zip(offsets, offsets[1:])]
    selections['random'] = torch.tensor([j for lo, hi, n in zip(offsets, offsets[1:], by_layer)
                                         for j in rng.sample(range(lo, hi), n)], dtype=torch.long)
    radius = args.strength * min(float(norms[ix].norm()) for ix in selections.values())

    @torch.inference_mode()
    def first(rows):
        return torch.cat([hidden_logits(array(rows[i:i + args.batch]), slice(-1, None))[:, 0].cpu()
                          for i in range(0, len(rows), args.batch)])

    before = first(clean).log_softmax(-1)
    metrics = {}
    for name, selected in [('original', None), *selections.items()]:
        saved = []
        scale = radius / float(norms[selected].norm()) if selected is not None else 0.
        with torch.no_grad():
            if selected is not None:
                for w, lo, hi in zip(writers, offsets, offsets[1:]):
                    ids = selected[(selected >= lo) & (selected < hi)].to(device) - lo
                    if ids.numel():
                        old = w.weight[:, ids].clone()
                        saved.append((w, ids, old))
                        w.weight[:, ids] = old * (1. - scale)
        try:
            arms = {policy: rewards(generate(heldout, args.eval_horizon, args.seed + 900,
                                             sample=policy == 'sample'), test_motifs)
                    for policy in ('greedy', 'sample')}
            copied = rewards(generate(copy, args.period, args.seed + 901, sample=False), test_motifs)
            after = first(clean).log_softmax(-1)
            kl = (before.exp() * (before - after)).sum(-1).tolist()
            metrics[name] = {'rollout': arms, 'copy': copied, 'clean_kl': kl, 'scale': scale,
                             'nominal_frobenius': radius if selected is not None else 0.,
                             'columns': selected.tolist() if selected is not None else []}
        finally:
            with torch.no_grad():
                for w, ids, old in saved:
                    w.weight[:, ids] = old
        print(json.dumps({'arm': name, 'greedy_match': sum(r['aligned'] for r in arms['greedy']) / len(heldout),
                          'sample_match': sum(r['aligned'] for r in arms['sample']) / len(heldout)}), flush=True)

    result = {'spec': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'effective_test_offset': test_start,
              'arms': metrics, 'calibration_risk_mean': float(risk.mean()),
              'centered_risk_sd': float(centered.std()), 'column_overlap': len(set(selections['trajectory'].tolist()) &
                                                                              set(selections['one_token'].tolist())),
              'peak_bytes': torch.cuda.max_memory_allocated(), 'elapsed_seconds': time.monotonic() - started,
              'torch_version': torch.__version__, 'device': torch.cuda.get_device_name()}
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'train', 'test', 'output'):
        p.add_argument('--' + name, required=True, type=Path)
    for name, default in dict(seed=4, offset=24576, test_offset=0, rows=16, test_rows=24,
                              horizon=16, eval_horizon=40, period=8, draws=4, columns=32,
                              batch=8, grad_batch=4).items():
        p.add_argument('--' + name.replace('_', '-'), type=int, default=default)
    p.add_argument('--strength', type=float, default=.5)
    p.add_argument('--objective', choices=('aligned', 'recurrence'), default='aligned')
    args = p.parse_args()
    if args.draws < 2 or args.horizon < args.period or not 0 < args.strength <= 1.:
        p.error('invalid draws, horizon, or strength')
    run(args)


if __name__ == '__main__':
    main()
