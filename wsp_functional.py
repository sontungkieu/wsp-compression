"""Forward-only fitting of physical weight patches against measured outputs."""
from dataclasses import dataclass

import torch


def groups(size, count):
    return [list(range(start, stop))
            for start, stop in zip([i * size // count for i in range(count)],
                                   [i * size // count for i in range(1, count + 1)])]


def least_squares_ball(matrix, target, radius, ridge=1e-6):
    """Solve min ||Ax-y||² + ridge ||x||² on the Euclidean ball."""
    left, values, right = torch.linalg.svd(matrix.double(), full_matrices=False)
    factors = values * (left.T @ target.double())

    def solution(shift):
        return right.T @ (factors / (values.square() + ridge + shift))

    if solution(0).norm() <= radius:
        return solution(0)
    low, high = 0., 1.
    while solution(high).norm() > radius:
        high *= 2
    for _ in range(45):
        midpoint = (low + high) / 2
        if solution(midpoint).norm() > radius:
            low = midpoint
        else:
            high = midpoint
    return solution(high)


def reachability(target, protected, requested, rtol=1e-5):
    """Exact projection for the measured linear map and chosen rank tolerance."""
    _, spectrum, right = torch.linalg.svd(protected, full_matrices=True)
    protected_rank = int((spectrum > spectrum[0] * rtol).sum())
    null = right[protected_rank:].T
    restricted = target @ null
    left, singular, _ = torch.linalg.svd(restricted, full_matrices=False)
    rank = int((singular > target.norm() * rtol).sum())
    projection = left[:, :rank] @ (left[:, :rank].T @ requested)
    return {'protected_rank': protected_rank, 'null_dim': null.shape[1],
            'reachable_rank': rank, 'singular_values': singular.tolist(),
            'relative_unreachable': float((requested - projection).norm() /
                                          requested.norm().clamp_min(1e-12))}


@dataclass
class Patch:
    """Each group has disjoint tensors. Coefficients use Frobenius geometry."""
    groups: list

    def __post_init__(self):
        self.references = [[(dst, dst.detach().clone(), delta)
                            for dst, delta in group] for group in self.groups]
        self.norms = torch.tensor(
            [sum(float(delta.double().square().sum()) for _, _, delta in group) ** .5
             for group in self.references], dtype=torch.float64)
        if bool((self.norms <= 1e-12).any()):
            raise ValueError('zero patch direction')

    @torch.no_grad()
    def apply(self, coefficients):
        for group, coefficient, norm in zip(self.references, coefficients, self.norms):
            scale = float(coefficient / norm)
            for target, base, delta in group:
                target.copy_(base)
                target.add_(delta, alpha=scale)


@torch.inference_mode()
def last_logits(model, rows, batch_size=8):
    device = next(model.parameters()).device
    result = []
    for offset in range(0, len(rows), batch_size):
        x = torch.as_tensor(rows[offset:offset + batch_size], device=device)
        hidden = model.model(input_ids=x, use_cache=False).last_hidden_state[:, -1]
        result.append(model.lm_head(hidden).float().cpu())
    return torch.cat(result)


def margins(logits, tokens, reference):
    return (logits.gather(1, tokens) - logits.gather(1, reference)).flatten().double()


def mean_kl(reference, measured):
    p = reference.log_softmax(-1)
    return float((p.exp() * (p - measured.log_softmax(-1))).sum(-1).mean())


def compile_patch(patch, measure, edit_target, preserve_target, radius, preserve_weight):
    """Measure a finite-difference Jacobian and solve with the actual weight budget."""
    k = len(patch.groups)
    zero = torch.zeros(k, dtype=torch.float64)
    patch.apply(zero)
    before_e, before_p = measure()
    bump = max(radius / 25, 1e-3)
    edit_columns, protect_columns = [], []
    for col in range(k):
        direction = zero.clone()
        direction[col] = bump
        patch.apply(direction)
        plus_e, plus_p = measure()
        patch.apply(-direction)
        minus_e, minus_p = measure()
        edit_columns.append((plus_e - minus_e) / (2 * bump))
        protect_columns.append((plus_p - minus_p) / (2 * bump))
    a_e = torch.stack(edit_columns, dim=1)
    a_p = torch.stack(protect_columns, dim=1)
    desired_e = edit_target - before_e
    desired_p = preserve_target - before_p
    ew = a_e.shape[0] ** -.5
    pw = (preserve_weight / a_p.shape[0]) ** .5
    joint = torch.cat((ew * a_e, pw * a_p))
    requested = torch.cat((ew * desired_e, pw * desired_p))
    answer = least_squares_ball(joint, requested, radius)
    # Wholly held-out rows are not touched during either solve.
    patch.apply(answer)
    after_e, after_p = measure()
    patch.apply(zero)
    edit_prediction = a_e @ answer
    return answer, {
        'forward_probes_per_panel': 2 * k + 2,
        'radius': radius,
        'finite_difference_step': bump,
        'fit_mse_before': float(desired_e.square().mean()),
        'fit_mse_linear': float((edit_prediction - desired_e).square().mean()),
        'fit_mse_actual': float((after_e - edit_target).square().mean()),
        'fit_protected_mse_actual': float((after_p - preserve_target).square().mean()),
        'linearization_rmse': float((after_e - before_e - edit_prediction).square().mean().sqrt()),
        'strict_zero_drift': reachability(a_e, a_p, desired_e),
        'unprotected': reachability(a_e, torch.zeros_like(a_p), desired_e),
        'direction_norms': patch.norms.tolist(),
        'coefficients': answer.tolist(),
    }


def calibration_rows(ids, start, count, length):
    data = ids[start:start + count * length]
    if len(data) != count * length:
        raise ValueError('too few calibration tokens')
    return [data[i:i + length] for i in range(0, len(data), length)]


def prepare_compression(model, install, choices, windows, config):
    """Construct physical patches on the pruned layout and hold out logit panels."""
    program = config['functional']
    start = program['start']
    count = program['rows']
    seq = program['length']
    edit, protect, test = [windows(start + d * count * seq, count, seq).tolist()
                           for d in (0, 1, 2)]
    teacher = [last_logits(model, panel) for panel in (edit, protect, test)]
    selected = [v.topk(9, dim=1).indices for v in teacher[:2]]
    teach_e, teach_p = [margins(v, s[:, 1:], s[:, :1]) for v, s in zip(teacher, selected)]
    install('drop')
    physical = [layer.mlp.down_proj for layer in model.model.layers]
    blocks = groups(len(physical), program['directions'])
    patch = Patch([
        [(physical[i].weight, choices[i]['arms']['ridge'][0]) for i in block]
        for block in blocks
    ])

    def measure():
        return tuple(margins(last_logits(model, rows), tokens[:, 1:], tokens[:, :1])
                     for rows, tokens in zip((edit, protect), selected))

    coefficients, record = compile_patch(patch, measure, teach_e, teach_p,
                                         float(patch.norms.norm()), program['preserve_weight'])
    zero = torch.zeros_like(coefficients)
    for name, values in [('drop', zero), ('ridge', patch.norms), ('function_fit', coefficients)]:
        patch.apply(values)
        record[name + '_heldout_teacher_kl'] = mean_kl(teacher[2], last_logits(model, test))
    for i, block in enumerate(blocks):
        factor = float(coefficients[i] / patch.norms[i])
        for j in block:
            delta, bias = choices[j]['arms']['ridge']
            choices[j]['arms']['function_fit'] = (delta * factor, bias)
    return record


def repeat_rows(rows):
    repeat, clean, target = [], [], []
    for row in rows:
        if len(row) < 32:
            raise ValueError('loop probes need at least 32 tokens')
        motif = row[-8:]
        repeat.append(row[:-24] + motif * 3)
        clean.append(row)
        target.append(motif[0])
    return repeat, clean, torch.tensor(target, dtype=torch.long)[:, None]


@torch.inference_mode()
def rollout(model, prompts, eos, steps, penalty=0.):
    """Evaluate excess recurrence separately from stopping and output length."""
    histories = [list(x) for x in prompts]
    repeats = [0] * len(histories)
    lengths = [0] * len(histories)
    done = [False] * len(histories)
    for _ in range(steps):
        logits = last_logits(model, histories)
        blocked = [h[-8] if h[-16:-8] == h[-8:] else None for h in histories]
        if penalty:
            for i, token in enumerate(blocked):
                if token is not None:
                    logits[i, token] -= penalty
        for i, token in enumerate(logits.argmax(-1).tolist()):
            if done[i]:
                histories[i].append(eos)
                continue
            lengths[i] += 1
            repeats[i] += int(token == blocked[i])
            histories[i].append(token)
            done[i] = token == eos
        if all(done):
            break
    return {'per_example_recurrence': repeats, 'per_example_length': lengths,
            'per_example_eos': done, 'recurrence_per_generated_token': sum(repeats) / sum(lengths)}


def run_repetition(model, tokenizer, config, source_digest):
    """Compile a context-dependent, token-specific penalty into down projections."""
    from recovery import fit_correction

    spec = config['functional']
    ids = tokenizer(open(config['train']).read(), add_special_tokens=False,
                    truncation=True, max_length=100000).input_ids
    n, seq, start = spec['rows'], spec['length'], spec['start']
    positive, negative, target = repeat_rows(calibration_rows(ids, start, n, seq))
    test_pos, test_neg, test_target = repeat_rows(calibration_rows(ids, start + 2 * n * seq, n, seq))
    blocks = groups(len(model.model.layers), spec['directions'])
    device = next(model.parameters()).device
    captures = {}
    handles = []

    def hook(index):
        def collect(_module, args):
            captures.setdefault(index, []).append(args[0][:, -1].detach().float())
        return collect

    for block in blocks:
        for i in block:
            handles.append(model.model.layers[i].mlp.down_proj.register_forward_pre_hook(hook(i)))
    before_e = last_logits(model, positive)
    before_p = last_logits(model, negative)
    for handle in handles:
        handle.remove()
    base_reference = before_e.topk(2, dim=1).indices
    ref = torch.where(base_reference[:, :1] == target, base_reference[:, 1:], base_reference[:, :1])
    protect_ids = before_p.topk(5, dim=1).indices
    requested = margins(before_e, target, ref) - spec['penalty']
    preserved = margins(before_p, protect_ids[:, 1:], protect_ids[:, :1])
    directions = []
    unembedding = model.lm_head.weight.detach()
    desired = unembedding[ref.flatten().to(device)].float() - unembedding[target.flatten().to(device)].float()
    desired = desired / desired.norm(dim=1, keepdim=True).clamp_min(1e-12)
    targets = torch.cat((desired, torch.zeros_like(desired)))
    genuine_norm_sq = 0.
    for block in blocks:
        changes = []
        for i in block:
            weight = model.model.layers[i].mlp.down_proj.weight
            act = torch.cat(captures[i])
            delta, _, _ = fit_correction(act, targets, 1.0, False)
            genuine_norm_sq += float(delta.double().square().sum())
            if spec.get('shuffled_basis'):
                delta, _, _ = fit_correction(act, targets.roll(n // 2, dims=0), 1.0, False)
            changes.append((weight, delta))
        directions.append(changes)
    del captures
    patch = Patch(directions)
    radius = genuine_norm_sq ** .5 * spec['radius_scale']

    def measure():
        ep = margins(last_logits(model, positive), target, ref)
        pp = margins(last_logits(model, negative), protect_ids[:, 1:], protect_ids[:, :1])
        return ep, pp

    coefficients, diagnostics = compile_patch(patch, measure, requested, preserved,
                                               radius, spec['preserve_weight'])
    test_before_pos = last_logits(model, test_pos)
    test_before_neg = last_logits(model, test_neg)
    top = test_before_pos.topk(2, dim=1).indices
    test_ref = torch.where(top[:, :1] == test_target, top[:, 1:], top[:, :1])
    train_tokens = set(target.flatten().tolist())
    initial_margin = margins(test_before_pos, test_target, test_ref)
    initial_negative = margins(test_before_neg, test_target, test_ref)
    seen = torch.tensor([int(x) in train_tokens for x in test_target.flatten()])
    records = {}
    generation_prompts = test_pos[:spec.get('generation_rows', 8)]

    for arm, c in [('original', torch.zeros_like(coefficients)),
                   ('local', patch.norms * (coefficients.norm() / patch.norms.norm())),
                   ('function_fit', coefficients)]:
        patch.apply(c)
        p = last_logits(model, test_pos)
        q = last_logits(model, test_neg)
        shift = margins(p, test_target, test_ref) - initial_margin
        false_shift = margins(q, test_target, test_ref) - initial_negative
        records[arm] = {
            'weight_delta_frobenius': float(c.norm()),
            'target_shift': float(shift.mean()),
            'target_shift_unseen_token': float(shift[~seen].mean()) if (~seen).any() else None,
            'unseen_token_count': int((~seen).sum()),
            'false_target_shift': float(false_shift.mean()),
            'preservation_kl': mean_kl(test_before_neg, q),
            'target_argmax_rate': float((p.argmax(-1) == test_target.flatten()).float().mean()),
            'eos_probability': float(p.softmax(-1)[:, tokenizer.eos_token_id].mean()),
            'per_example_shift': shift.tolist(),
            'per_example_false_shift': false_shift.tolist(),
            'greedy_rollout': rollout(model, generation_prompts, tokenizer.eos_token_id, 24),
        }
    patch.apply(torch.zeros_like(coefficients))
    oracle = rollout(model, generation_prompts, tokenizer.eos_token_id, 24, spec['penalty'])
    return {'spec': config, 'source_sha256': source_digest, 'functional': diagnostics,
            'oracle_rollout': oracle, 'results': records}


if __name__ == '__main__':
    a = torch.eye(2, dtype=torch.float64)
    target = torch.tensor([3., 4.], dtype=torch.float64)
    torch.testing.assert_close(least_squares_ball(a, target, 2., ridge=0), target * .4)
    residual = reachability(a, a[:1], target)
    assert residual['null_dim'] == residual['reachable_rank'] == 1
    assert abs(residual['relative_unreachable'] - .6) < 1e-10
    parameter = torch.nn.Parameter(torch.tensor([2., -1.]))
    patch = Patch([[(parameter, torch.tensor([0., 2.]))]])
    patch.apply([3.])
    torch.testing.assert_close(parameter, torch.tensor([2., 2.]))
    patch.apply([0.])
    torch.testing.assert_close(parameter, torch.tensor([2., -1.]))

    @torch.inference_mode()
    def fake_logits(_model, rows):
        return torch.tensor([[1., 2., 0.] for _ in rows])

    last_logits = fake_logits
    assert rollout(None, [[1] * 16], 2, 1)['per_example_recurrence'] == [1]
    assert rollout(None, [[1] * 16], 2, 1, 2.)['per_example_recurrence'] == [0]
    print('PASS: trust-region optimum, nullspace, patch reset and recurrence oracle')
