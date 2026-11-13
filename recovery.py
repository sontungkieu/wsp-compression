#!/usr/bin/env python3
"""Fixed-mask channel pruning with matched analytical and proposed corrections.

Offline, one assigned GPU; no gradient optimization or external LLM calls.
Each run writes exactly one new receipt plus a checkpoint-free event stream.
"""
import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()

def fit_correction(x, y, relative_lambda, affine):
    import torch
    # Use the same absolute regularizer for both fits, so centering does not
    # silently change ridge strength. Intercepts are not penalized.
    regularizer = relative_lambda * x.square().sum(dim=1).mean().clamp_min(1e-12)
    mx = x.mean(0) if affine else torch.zeros(x.shape[1], device=x.device)
    my = y.mean(0) if affine else torch.zeros(y.shape[1], device=y.device)
    xx, yy = x - mx, y - my
    gram = xx @ xx.T
    gram.diagonal().add_(regularizer)
    delta = torch.linalg.solve(gram, yy).T @ xx
    bias = my - mx @ delta.T
    if not torch.isfinite(delta).all() or not torch.isfinite(bias).all():
        raise RuntimeError('nonfinite reconstruction')
    return delta, bias, float(regularizer)

def self_test():
    import torch
    torch.manual_seed(39)
    x, y = torch.randn(13, 5), torch.randn(13, 3)
    lam = .2
    reg = lam * x.square().sum(1).mean()
    for affine in (False, True):
        d, b, r = fit_correction(x, y, lam, affine)
        xx = x - x.mean(0) if affine else x
        yy = y - y.mean(0) if affine else y
        expected = torch.linalg.solve(xx.T @ xx + reg*torch.eye(5), xx.T @ yy).T
        assert torch.allclose(d, expected, atol=3e-6)
        assert abs(r-float(reg)) < 1e-6
        if affine:
            assert torch.allclose((y-x@d.T-b).mean(0), torch.zeros(3), atol=1e-6)
    # The best constant output is exactly its mean.
    assert (y-y.mean(0)).square().sum() <= (y-torch.zeros(3)).square().sum()
    print(json.dumps({'self_test': 'passed', 'checks': ['dual_vs_primal', 'shared_lambda', 'affine_normal_equation', 'mean_optimum']}))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--spec')
    p.add_argument('--output')
    p.add_argument('--self-test', action='store_true')
    a = p.parse_args()
    if a.self_test:
        self_test()
        return
    cfg = json.loads(Path(a.spec).read_text())
    output = Path(a.output)
    if output.exists():
        raise SystemExit('fresh output required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(cfg['gpu']):
        raise SystemExit('CUDA device differs from the assigned device in the spec')
    started = time.monotonic()
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if torch.cuda.device_count() != 1:
        raise SystemExit('exactly one CUDA device required')
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dtype = torch.float32 if cfg.get('dtype', 'float32') == 'float32' else torch.bfloat16
    device = torch.device('cuda:0')
    model_path = Path(cfg['model'])
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype=dtype).to(device).eval()
    if model.config.model_type != 'qwen2':
        raise SystemExit('only verified qwen2 architecture supported')
    layers = model.model.layers
    originals = [(l.mlp.gate_proj, l.mlp.up_proj, l.mlp.down_proj) for l in layers]
    dense_parameters = sum(p.numel() for p in model.parameters())
    seq = cfg['context']
    cal_seq = cfg.get('cal_context', 256)
    train_path = Path(cfg['train'])
    train_ids = tokenizer(train_path.read_text(), add_special_tokens=False,
                          truncation=True, max_length=cfg.get('max_train_tokens', 100000)).input_ids
    def windows(offset, count, length=cal_seq):
        ids = train_ids[offset:offset+count*length]
        if len(ids) != count*length:
            raise RuntimeError('insufficient calibration tokens')
        return torch.tensor(ids).reshape(count, length)

    def capture(rows):
        captured = [[] for _ in layers]
        handles = []
        for i, (_, _, down) in enumerate(originals):
            def hook(_module, inputs, layer_index=i):
                captured[layer_index].append(inputs[0].detach().float().reshape(-1, originals[layer_index][2].in_features))
            handles.append(down.register_forward_pre_hook(hook))
        try:
            with torch.inference_mode():
                for row in rows:
                    model.model(input_ids=row[None].to(device), use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        return [torch.cat(x, dim=0) for x in captured]

    mask_rows = windows(cfg['mask_offset'], cfg['mask_windows'])
    fit_rows = windows(cfg['fit_offset'], cfg['fit_windows'])
    hold_rows = windows(cfg['local_hold_offset'], cfg.get('local_hold_windows', 4))
    mask_act = capture(mask_rows)
    fit_act = mask_act if (cfg['mask_offset'], cfg['mask_windows']) == (cfg['fit_offset'], cfg['fit_windows']) else capture(fit_rows)
    hold_act = capture(hold_rows)
    masks, choices, diagnostics = [], [], []
    for i, (_, _, down) in enumerate(originals):
        width = down.in_features
        keep_n = width - int(width * cfg['remove_fraction'])
        metric = mask_act[i].var(dim=0, correction=0) if cfg.get('mask_metric', 'energy') == 'variance' else mask_act[i].square().mean(0)
        score = metric * down.weight.detach().float().square().sum(0)
        keep = torch.argsort(score, descending=True)[:keep_n].sort().values
        mask = torch.ones(width, dtype=torch.bool, device=device)
        mask[keep] = False
        removed = torch.arange(width, device=device)[mask]
        x = fit_act[i][:, keep]
        wd = down.weight.detach().float()[:, removed]
        y = fit_act[i][:, removed] @ wd.T
        xh = hold_act[i][:, keep]
        yh = hold_act[i][:, removed] @ wd.T
        zero_d = None
        zero_b = torch.zeros(down.out_features, device=device)
        mean_b = y.mean(0)
        ridge_d, ridge_b, reg = fit_correction(x, y, cfg['ridge_lambda'], False)
        affine_d, affine_b, reg2 = fit_correction(x, y, cfg['ridge_lambda'], True)
        assert reg == reg2
        arm = {'drop': (zero_d, zero_b), 'mean': (zero_d, mean_b),
               'ridge': (ridge_d, ridge_b), 'affine': (affine_d, affine_b)}
        candidate = cfg.get('candidate')
        if candidate is not None:
            basis = candidate.get('basis', 'ridge')
            if basis not in ('ridge', 'affine'):
                raise ValueError('unsupported correction basis')
            delta, bias = arm[basis]
            position = i / max(1, len(layers)-1)
            profile = candidate.get('profile', 'flat')
            profiles = {'flat': 1.0, 'front': 1.5-position,
                        'back': .5+position, 'middle': 1.5-2*abs(position-.5),
                        'protect_late': 1.0 if position < .7 else .25,
                        'protect_early': .25 if position < .3 else 1.0}
            if profile not in profiles:
                raise ValueError('unsupported layer profile')
            scale = float(candidate.get('gain', 1.0))*profiles[profile]
            delta = delta*scale
            mix = float(candidate.get('mean_mix', 0.0))
            bias = (1-mix)*bias*scale + mix*mean_b
            cap = candidate.get('row_norm_cap')
            if cap is not None:
                retained_norm = down.weight.detach().float()[:,keep].norm(dim=1)
                limit = float(cap)*retained_norm
                factor = (limit/delta.norm(dim=1).clamp_min(1e-12)).clamp(max=1)
                delta = delta*factor[:,None]
            arm['candidate'] = (delta, bias)

        stats = {}
        for name, (delta, bias) in arm.items():
            pred = bias[None] if delta is None else x @ delta.T + bias
            predh = bias[None] if delta is None else xh @ delta.T + bias
            stats[name] = {'train_mse': float((y-pred).square().mean()),
                           'heldout_mse': float((yh-predh).square().mean()),
                           'delta_relative_norm': 0 if delta is None else float(delta.norm()/down.weight.detach().float()[:,keep].norm().clamp_min(1e-12)),
                           'bias_norm': float(bias.norm())}
        choices.append({'keep': keep, 'arms': arm})
        masks.append({'layer': i, 'before': width, 'after': keep_n, 'sha256': tensor_digest(keep)})
        diagnostics.append({'layer': i, 'lambda_absolute': reg, 'arms': stats})
        # Drop activation references once layer diagnostics and fits are complete.
        mask_act[i] = None
        fit_act[i] = None
        hold_act[i] = None
        print(json.dumps({'fit_layer': i, 'keep': keep_n, 'lambda_absolute': reg}), flush=True)

    def linear_slice(source, rows=None, cols=None, delta=None, bias_override=None):
        weight = source.weight.detach()
        if rows is not None:
            weight = weight[rows]
        if cols is not None:
            weight = weight[:, cols]
        if delta is not None:
            weight = weight.float() + delta * cfg.get('gain', 1.0)
        bias = source.bias.detach() if source.bias is not None else None
        if rows is not None and bias is not None:
            bias = bias[rows]
        if bias_override is not None:
            bias = bias_override * cfg.get('gain', 1.0) + (bias if bias is not None else 0)
        result = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None, device=device, dtype=dtype)
        with torch.no_grad():
            result.weight.copy_(weight)
            if bias is not None:
                result.bias.copy_(bias)
        return result

    def install(name):
        for i, layer in enumerate(layers):
            gate, up, down = originals[i]
            if name == 'original':
                layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj = originals[i]
                continue
            keep = choices[i]['keep']
            delta, bias = choices[i]['arms'][name]
            layer.mlp.gate_proj = linear_slice(gate, rows=keep)
            layer.mlp.up_proj = linear_slice(up, rows=keep)
            layer.mlp.down_proj = linear_slice(down, cols=keep, delta=delta, bias_override=bias)

    def nll_record(ids, target_start=1):
        xx = torch.tensor(ids, device=device)[None]
        with torch.inference_mode():
            logits = model(input_ids=xx, use_cache=False).logits[0, :-1].float()
            losses = F.cross_entropy(logits, xx[0,1:], reduction='none')
            begin = max(1, target_start)-1
            values = losses[begin:]
            correct = (logits[begin:].argmax(-1) == xx[0, begin+1:]).all().item()
        return float(values.sum()), values.numel(), bool(correct)

    # Freeze evaluation IDs once. No labels or heldout scores affect fits.
    evals = {}
    for split in ('validation', 'test'):
        count = cfg.get('wiki_'+split+'_windows', 0)
        if count:
            path = Path(cfg['wiki_'+split])
            n = count*seq
            ids = tokenizer(path.read_text(), add_special_tokens=False, truncation=True,
                            max_length=cfg['wiki_offset']+n).input_ids
            ids = ids[cfg['wiki_offset']:cfg['wiki_offset']+n]
            if len(ids) != n:
                raise RuntimeError('insufficient '+split+' tokens')
            evals['wiki_'+split] = [(str(i), ids[i*seq:(i+1)*seq], 1) for i in range(count)]
    data_dir = Path(cfg['data_dir'])
    if cfg.get('lambada_n', 0):
        rows = [json.loads(x) for x in (data_dir/'lambada.jsonl').read_text().splitlines()]
        selected = rows[cfg['lambada_offset']:cfg['lambada_offset']+cfg['lambada_n']]
        if len(selected) != cfg['lambada_n']:
            raise RuntimeError('insufficient lambada docs')
        data = []
        for row in selected:
            text = row['text'].rstrip()
            boundary = text.rfind(' ') + 1
            encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            target = next(i for i, (s,e) in enumerate(encoded.offset_mapping) if e > boundary)
            ids = encoded.input_ids
            trim = max(0, len(ids)-seq)
            if target <= trim:
                raise RuntimeError('target lacks context after truncation')
            data.append((str(row['id']), ids[trim:], target-trim))
        evals['lambada'] = data
    arc_rows = []
    if cfg.get('arc_n', 0):
        rows = [json.loads(x) for x in (data_dir/('arc_'+cfg.get('arc_split','test')+'.jsonl')).read_text().splitlines()]
        arc_rows = rows[cfg['arc_offset']:cfg['arc_offset']+cfg['arc_n']]
        if len(arc_rows) != cfg['arc_n']:
            raise RuntimeError('insufficient ARC docs')

    def evaluate():
        result = {}
        for name, rows in evals.items():
            records = []
            for rid, ids, target in rows:
                total, count, _ = nll_record(ids)
                record = {'id': rid, 'sum_nll': total, 'tokens': count}
                if name == 'lambada':
                    word_nll, nt, correct = nll_record(ids, target)
                    record.update({'target_sum_nll': word_nll, 'target_tokens': nt, 'last_word_correct': correct})
                records.append(record)
            loss = sum(x['sum_nll'] for x in records)/sum(x['tokens'] for x in records)
            result[name] = {'nll': loss, 'perplexity': math.exp(loss), 'records': records}
            if name == 'lambada':
                result[name]['last_word_accuracy'] = sum(x['last_word_correct'] for x in records)/len(records)
        if arc_rows:
            records = []
            for row in arc_rows:
                prefix = 'Question: '+row['question']+'\nAnswer:'
                prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
                sums, means = [], []
                for answer in row['choices']:
                    ids = tokenizer(prefix+' '+answer, add_special_tokens=False).input_ids
                    # Back up over a tokenization boundary merge, as likelihood harnesses do.
                    common = 0
                    for x,y in zip(prefix_ids, ids):
                        if x != y:
                            break
                        common += 1
                    trim = max(0, len(ids)-seq)
                    if common <= trim:
                        raise RuntimeError('ARC continuation has no prefix')
                    total, count, _ = nll_record(ids[trim:], common-trim)
                    sums.append(total)
                    means.append(total/count)
                pred_raw = min(range(len(sums)), key=sums.__getitem__)
                pred_norm = min(range(len(means)), key=means.__getitem__)
                records.append({'id': str(row['id']), 'correct_raw': pred_raw==row['gold'],
                                'correct_norm': pred_norm==row['gold'], 'choice_nll': sums, 'choice_mean_nll': means})
            result['arc_easy'] = {'accuracy_raw': sum(x['correct_raw'] for x in records)/len(records),
                                  'accuracy_norm': sum(x['correct_norm'] for x in records)/len(records),
                                  'records': records}
        return result

    results = {}
    for mode in cfg.get('methods', ['original','drop','mean','ridge','affine']):
        install(mode)
        torch.cuda.synchronize()
        t = time.monotonic()
        result = evaluate()
        torch.cuda.synchronize()
        results[mode] = {'parameters': sum(p.numel() for p in model.parameters()),
                         'seconds': time.monotonic()-t, 'scores': result}
        if cfg.get('benchmark', False):
            latency = {}
            with torch.inference_mode():
                for length in (256, 512, 1024):
                    ids = torch.tensor(train_ids[:length], device=device)[None]
                    for _ in range(3):
                        warmup = model(input_ids=ids, use_cache=False)
                        del warmup
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    for _ in range(10):
                        measured = model(input_ids=ids, use_cache=False)
                        del measured
                    end_event.record()
                    torch.cuda.synchronize()
                    latency['prefill_'+str(length)+'_ms'] = start_event.elapsed_time(end_event)/10
            results[mode]['microbenchmark'] = {
                'batch_size': 1, 'warmup_iterations': 3, 'measured_iterations': 10,
                'cuda_event_times': latency,
                'resident_memory_includes_reference_modules': True,
                'protocol': 'eager forward with full logits and use_cache=False'}
        # Preserve completed arms if a later arm reaches the campaign deadline.
        partial = output.with_suffix('.partial.json')
        temporary = output.with_suffix('.partial.tmp')
        temporary.write_text(json.dumps({'status': 'partial', 'spec': cfg,
                                         'results': results, 'mask': masks}))
        os.replace(temporary, partial)
        print(json.dumps({'method': mode, 'scores': {k:{x:y for x,y in v.items() if x!='records'} for k,v in result.items()}}), flush=True)
    pruned = [v['parameters'] for k,v in results.items() if k!='original']
    if len(set(pruned)) > 1 or (pruned and pruned[0]>=dense_parameters):
        raise RuntimeError('invalid matched parameter budget')
    result = {'spec': cfg, 'spec_sha256': digest(a.spec), 'source_sha256': digest(__file__),
              'mask': masks, 'mask_sha256': hashlib.sha256(json.dumps(masks, sort_keys=True).encode()).hexdigest(),
              'diagnostics': diagnostics, 'results': results, 'dense_parameters': dense_parameters,
              'runtime': {'torch': torch.__version__, 'transformers': transformers.__version__,
                          'gpu': torch.cuda.get_device_name(0), 'dtype': str(dtype),
                          'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'tf32': False},
              'protocol': {'gradient_training': False, 'script_invokes_llm': False,
                           'arc_prompt': 'Question: {question}\\nAnswer: {choice}',
                           'arc_scores': 'custom zero-shot answer likelihood; not standard lm-eval reproduction',
                           'lambada_score': 'teacher-forced exact greedy target-token match with left truncation',
                           'bias_slots_matched_in_all_pruned_arms': True,
                           'timings_are_serving_benchmarks': False},
              'inputs': {'train_sha256': digest(train_path), 'dataset_manifest_sha256': digest(data_dir/'manifest.json')},
              'wall_seconds': time.monotonic()-started}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({'status': 'completed', 'output': str(output), 'seconds': result['wall_seconds']}), flush=True)

if __name__ == '__main__':
    main()
