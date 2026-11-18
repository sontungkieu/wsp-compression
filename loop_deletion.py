"""Compare learned binary deletion and Astra-style projection on held-out loop probes."""
import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import random
import statistics
import time


def bootstrap_delta(edited, base, seed, draws=1500):
    """Paired prompt bootstrap for an edit-minus-base mean."""
    delta = [a - b for a, b in zip(edited, base, strict=True)]
    if not delta:
        raise ValueError('empty comparison panel')
    rng = random.Random(seed)
    n = len(delta)
    means = sorted(sum(rng.choices(delta, k=n)) / n for _ in range(draws))
    return {'mean': statistics.fmean(delta),
            'interval_95': [means[int(.025 * draws)], means[min(draws - 1, int(.975 * draws))]]}


def summarize(rows):
    return {key: statistics.fmean(row[key] for row in rows)
            for key in ('aligned', 'recurrence', 'motif_completed', 'length', 'stopped')}


def find_projection(model, loop, clean, batch_size, layer_count, columns):
    """Locate the MLP output directions and build a hard parameter-budget reference."""
    import torch

    modules = [layer.mlp.down_proj for layer in model.model.layers]
    device = model.device
    captures = [{index: [torch.zeros(w.in_features, device=device),
                         torch.zeros(w.out_features, device=device),
                         torch.zeros((), device=device)]
                 for index, w in enumerate(modules)} for _ in range(2)]

    def hook(sums):
        def collect(_module, inputs, output):
            incoming = inputs[0][:, -1].float()
            outgoing = output[:, -1].float()
            sums[0].add_(incoming.sum(0))
            sums[1].add_(outgoing.sum(0))
            sums[2].add_(outgoing.square().sum())
        return collect

    for panel, sums in zip((loop, clean), captures, strict=True):
        handles = [w.register_forward_hook(hook(sums[i])) for i, w in enumerate(modules)]
        try:
            with torch.no_grad():
                for start in range(0, len(panel), batch_size):
                    ids = torch.tensor(panel[start:start + batch_size], dtype=torch.long, device=device)
                    model.model(input_ids=ids, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()

    contrasts = []
    for i in range(len(modules)):
        act_loop, out_loop, _ = captures[0][i]
        act_clean, out_clean, sq_clean = captures[1][i]
        direction = out_loop / len(loop) - out_clean / len(clean)
        magnitude = direction.norm()
        if torch.isfinite(magnitude) and float(magnitude) > 0.:
            layer_score = float(magnitude / (sq_clean / len(clean)).sqrt().clamp_min(1e-8))
            contrasts.append((layer_score, i, direction / magnitude,
                              act_loop / len(loop) - act_clean / len(clean)))
    if len(contrasts) < layer_count:
        raise RuntimeError('too few layers with a finite repetition contrast')

    patches = []
    for layer_score, i, direction, contrast in sorted(contrasts, key=lambda row: (-row[0], row[1]))[:layer_count]:
        w = modules[i].weight
        writer = direction @ w.detach().float()
        selected = (contrast * writer).topk(columns).indices
        old = w.detach()[:, selected]
        new = (old.float() - direction[:, None] * writer[selected]).to(w.dtype)
        count = int((old != new).sum())
        if not count:
            raise RuntimeError(f'zero displacement after rounding in layer {i}')
        patches.append((i, selected, new, count, layer_score))
    return patches


def candidate_pool(weight, count, multiple, generator):
    """Draw a fixed task-independent candidate pool shared by every binary mask."""
    import torch

    locations = weight.detach().flatten().nonzero().flatten()
    size = min(multiple * count, locations.numel())
    if size <= count:
        raise ValueError('the candidate pool must be larger than its deletion count')
    perm = torch.randperm(locations.numel(), device=weight.device, generator=generator)
    return locations[perm[:size]]


def learn_mask(model, calibration, targets, protected, pools, args):
    import torch
    from torch.nn import functional as F
    from torch.nn.utils import parametrize

    class FixedBudgetMask(torch.nn.Module):
        def __init__(self, candidates, count, generator):
            super().__init__()
            self.register_buffer('candidates', candidates)
            self.count = count
            initial = math.log(count / (candidates.numel() - count))
            jitter = .02 * torch.randn(candidates.numel(), device=candidates.device, generator=generator)
            self.scores = torch.nn.Parameter(jitter + initial)

        def selected(self):
            top = self.scores.topk(self.count, sorted=False).indices
            return self.candidates[top]

        def forward(self, original):
            soft = self.scores.sigmoid()
            top = self.scores.topk(self.count, sorted=False).indices
            hard = torch.zeros_like(soft).scatter_(0, top, 1.)
            # Exactly hard in the forward, with the sigmoid gradient in the backward.
            deletion = hard + (soft - soft.detach())
            flat = original.reshape(-1)
            kept = (flat[self.candidates].float() * (1. - deletion)).to(original.dtype)
            return flat.scatter(0, self.candidates, kept).reshape_as(original)

    def logits(ids, positions):
        hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state
        return model.lm_head(hidden[:, positions]).float()

    device = model.device
    x_loop = torch.tensor(calibration, device=device, dtype=torch.long)
    n, width = x_loop.shape
    y_loop = torch.tensor(targets, device=device, dtype=torch.long)[:, :args.probe_tokens]
    x_loop = torch.cat((x_loop, y_loop[:, :-1]), dim=-1)
    x_clean = torch.tensor(protected, device=device, dtype=torch.long)
    generator = torch.Generator(device=device).manual_seed(args.seed + 113)
    reference = []
    with torch.no_grad():
        for batch in x_clean.split(args.batch):
            reference.append(logits(batch, -1).log_softmax(-1).cpu())
    reference = torch.cat(reference)
    masks, initial, registered, history = {}, {}, [], []
    try:
        for i, (module, candidates, count) in pools.items():
            mask = FixedBudgetMask(candidates, count, generator)
            parametrize.register_parametrization(module, 'weight', mask)
            masks[i] = mask
            with torch.no_grad():
                initial[i] = mask.selected()
            registered.append(module)
        optim = torch.optim.Adam([mask.scores for mask in masks.values()], lr=args.lr)
        for step in range(args.steps):
            batch = min(args.batch, n)
            ix = torch.randperm(n, generator=generator, device=device)[:batch]
            iy = torch.randperm(len(protected), generator=generator, device=device)[:batch]
            optim.zero_grad(set_to_none=True)
            z = logits(x_loop[ix], slice(width - 1, None))
            other = z.scatter(-1, y_loop[ix, :, None], -torch.inf)
            # Negative log probability of choosing any token outside the known repeat.
            unlikelihood = (z.logsumexp(-1) - other.logsumexp(-1)).mean()
            unlikelihood.backward()
            ref = reference[iy.cpu()].to(device)
            protect = F.kl_div(logits(x_clean[iy], -1).log_softmax(-1), ref,
                               reduction='batchmean', log_target=True)
            (args.preserve_weight * protect).backward()
            if not bool(torch.isfinite(unlikelihood) & torch.isfinite(protect)):
                raise RuntimeError('nonfinite calibration loss')
            grad_norm = torch.nn.utils.clip_grad_norm_([mask.scores for mask in masks.values()],
                                                      max_norm=1., error_if_nonfinite=True)
            optim.step()
            record = {'step': step + 1, 'unlikelihood': float(unlikelihood.detach()),
                      'protection_kl': float(protect.detach()), 'grad_norm': float(grad_norm)}
            history.append(record)
            if step == 0 or (step + 1) % 8 == 0 or step + 1 == args.steps:
                print(json.dumps(record), flush=True)
        with torch.no_grad():
            selected = {i: mask.selected() for i, mask in masks.items()}
            history[-1]['changed_fraction_by_layer'] = {
                i: float((~torch.isin(indices, initial[i])).float().mean()) for i, indices in selected.items()
            }
        return selected, history
    finally:
        for module in reversed(registered):
            parametrize.remove_parametrizations(module, 'weight', leave_parametrized=False)


@contextmanager
def edited(changes):
    """Apply one static edit and restore the original weights before the next arm."""
    import torch

    saved, by_layer = [], {}
    try:
        with torch.no_grad():
            for layer, weight, indices, values in changes:
                old = weight[indices].clone()
                replacement = torch.zeros_like(old) if values is None else values
                changed = int((old != replacement).sum())
                square = float((old.float() - replacement.float()).square().sum())
                by_layer[layer] = {'count': changed, 'squared_frobenius': square}
                saved.append((weight, indices, old))
                weight[indices] = replacement
        yield {'count': sum(row['count'] for row in by_layer.values()),
               'frobenius': math.sqrt(sum(row['squared_frobenius'] for row in by_layer.values())),
               'by_layer': by_layer}
    finally:
        with torch.no_grad():
            for weight, indices, original in reversed(saved):
                weight[indices] = original


def run(args):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from loop_gradient import examples, rollout_tokens, score_rollout
    from six_gpu_campaign import save
    from wsp_functional import last_logits

    started = time.monotonic()
    if torch.cuda.device_count() != 1:
        raise SystemExit('requires exactly one assigned CUDA device')
    device = torch.device('cuda:0')
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
                                                torch_dtype=torch.bfloat16).to(device).eval()
    if model.config.model_type != 'qwen2':
        raise SystemExit('this pilot supports Qwen2.5 dense checkpoints')
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    train = tokenizer(args.train.read_text(), add_special_tokens=False,
                      truncation=True, max_length=100000).input_ids
    test = tokenizer(args.test.read_text(), add_special_tokens=False,
                     truncation=True, max_length=60000).input_ids
    copy_prefix = tokenizer('Continue copying this exact repeated sequence: ',
                            add_special_tokens=False).input_ids
    if len(copy_prefix) > 72 - 3 * args.period:
        raise ValueError('copy instruction overlaps the repeated motif')
    rng = random.Random(args.seed)
    clean, loop, copy, target = examples(train, args.offset, args.rows, args.period, rng, copy_prefix)
    test_start = args.test_offset % max(1, len(test) - 72 * args.test_rows + 1)
    clean_test, loop_test, copy_test, target_test = examples(
        test, test_start, args.test_rows, args.period, rng, copy_prefix)

    projection = find_projection(model, loop, clean, args.batch, args.layers, args.columns)
    layers = model.model.layers
    budget = {i: count for i, _, _, count, _ in projection}
    generator = torch.Generator(device=device).manual_seed(args.seed + 17)
    pools = {}
    for i, count in budget.items():
        module = getattr(layers[i].mlp, args.mask_module)
        pools[i] = (module, candidate_pool(module.weight, count, args.candidate_multiple, generator), count)
    learned, history = learn_mask(model, loop, target, clean + copy, pools, args)

    def deletions(selection):
        return [(i, pools[i][0].weight.view(-1), ix, None) for i, ix in selection.items()]

    arms = {'original': [], 'learned_deletion': deletions(learned),
            'astra_projection': [(i, layers[i].mlp.down_proj.weight, (slice(None), columns), values)
                                 for i, columns, values, _, _ in projection]}
    for trial in range(args.random_draws):
        selected = {i: pool[torch.randperm(pool.numel(), device=device, generator=generator)[:count]]
                    for i, (_, pool, count) in pools.items()}
        arms[f'random_deletion_{trial}'] = deletions(selected)

    eos = model.generation_config.eos_token_id or tokenizer.eos_token_id
    ends = [eos] if isinstance(eos, int) else list(eos)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else ends[0]

    def assess(rows, steps, seed, sample):
        suffix = rollout_tokens(model, rows, steps, seed, batch_size=args.batch, ends=ends,
                                pad=pad, sample=sample)
        return score_rollout(suffix, target_test, ends)

    base = last_logits(model, clean_test, batch_size=args.batch).log_softmax(-1)
    measurements = {}
    for arm, changes in arms.items():
        with edited(changes) as footprint:
            observed = {i: row['count'] for i, row in footprint['by_layer'].items()}
            if changes and observed != budget:
                raise RuntimeError(f'{arm} changed {observed}; expected actual edit budget {budget}')
            greedy = assess(loop_test, args.eval_horizon, args.seed + 900, False)
            sample = [assess(loop_test, args.eval_horizon, args.seed + 901 + draw, True)
                      for draw in range(args.eval_draws)]
            copy_rows = assess(copy_test, args.period, args.seed + 950, False)
            clean_logp = last_logits(model, clean_test, batch_size=args.batch).log_softmax(-1)
            kl = (base.exp() * (base - clean_logp)).sum(-1).tolist()
            measurements[arm] = {'edit': footprint, 'greedy': greedy, 'sampling': sample,
                                 'copy': copy_rows, 'clean_kl': kl}
        print(json.dumps({'arm': arm, 'actual_edits': footprint['count'],
                          'greedy': summarize(greedy), 'copy': summarize(copy_rows)}), flush=True)

    # Average samples within each prompt before bootstrapping across prompts.
    def per_prompt(result, policy, metric):
        if policy != 'sampling':
            return [row[metric] for row in result[policy]]
        return [statistics.fmean(row[metric] for row in draws) for draws in zip(*result[policy])]

    comparisons = {}
    random_arms = [arm for arm in measurements if arm.startswith('random_deletion_')]
    for control in ['original', 'astra_projection', 'random_mean']:
        metrics = {}
        for policy, metric in [('greedy', 'aligned'), ('greedy', 'recurrence'),
                               ('greedy', 'length'), ('greedy', 'stopped'),
                               ('sampling', 'aligned'), ('sampling', 'recurrence'),
                               ('sampling', 'length'), ('sampling', 'stopped'),
                               ('copy', 'motif_completed')]:
            if control == 'random_mean':
                all_random = [per_prompt(measurements[arm], policy, metric) for arm in random_arms]
                reference = [statistics.fmean(values) for values in zip(*all_random)]
            else:
                reference = per_prompt(measurements[control], policy, metric)
            learned_rows = per_prompt(measurements['learned_deletion'], policy, metric)
            metrics[f'{policy}_{metric}'] = bootstrap_delta(learned_rows, reference, args.seed)
        comparisons[control] = metrics

    original = measurements['original']
    original_rollouts = [original['greedy'], *original['sampling']]
    result = {'spec': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              'data_contract': 'synthetic repeated WikiText motifs, separate train/test files',
              'eligible': any(row['motif_completed'] or row['recurrence'] > 0.
                              for draw in original_rollouts for row in draw),
              'edit_budget_by_layer': budget, 'candidate_counts': {i: pool.numel() for i, (_, pool, _) in pools.items()},
              'learned_mask': {i: {'shape': list(pools[i][0].weight.shape),
                                   'zeroed_flat_indices': indices.sort().values.tolist()}
                               for i, indices in learned.items()},
              'projection_layers': {i: {'columns': cols.tolist(), 'contrast_score': score}
                                    for i, cols, _, _, score in projection},
              'effective_test_offset': test_start, 'training': history, 'arms': measurements,
              'paired_deltas': comparisons, 'peak_bytes': torch.cuda.max_memory_allocated(),
              'elapsed_seconds': time.monotonic() - started, 'torch_version': torch.__version__,
              'transformers_version': transformers.__version__, 'device': torch.cuda.get_device_name()}
    save(args.output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'train', 'test', 'output'):
        parser.add_argument(f'--{name}', required=True, type=Path)
    defaults = {'seed': 4, 'offset': 24576, 'test_offset': 0, 'rows': 64, 'test_rows': 40,
                'eval_horizon': 48, 'period': 8, 'probe_tokens': 4, 'columns': 32, 'layers': 2,
                'candidate_multiple': 8, 'steps': 48, 'batch': 8, 'random_draws': 3, 'eval_draws': 2}
    for name, default in defaults.items():
        parser.add_argument('--' + name.replace('_', '-'), type=int, default=default)
    parser.add_argument('--lr', type=float, default=.05)
    parser.add_argument('--preserve-weight', type=float, default=1.)
    parser.add_argument('--mask-module', choices=('gate_proj', 'down_proj'), default='gate_proj')
    args = parser.parse_args()
    if any(getattr(args, name) <= 0 for name in defaults if name not in ('seed', 'offset', 'test_offset')):
        parser.error('counts must be positive')
    if args.offset < 0 or args.test_offset < 0 or not (1 <= args.probe_tokens <= args.period <= 16):
        parser.error('invalid offsets, period, or calibration horizon')
    if args.eval_horizon <= 2 * args.period or args.candidate_multiple < 2:
        parser.error('rollout must cover at least two motif spans and the candidate pool must exceed budget')
    if not math.isfinite(args.lr) or args.lr <= 0. or not math.isfinite(args.preserve_weight) or args.preserve_weight < 0.:
        parser.error('invalid learning rate or preservation weight')
    if args.train.resolve() == args.test.resolve():
        parser.error('separate calibration and evaluation files are required')
    if args.output.exists():
        parser.error('a fresh output path is required')
    run(args)


if __name__ == '__main__':
    main()
