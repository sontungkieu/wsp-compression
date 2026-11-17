"""Compare sparse projection and output-fitted edits on natural Qwen3.5 loops."""
import argparse
import json
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import time

import six_gpu_campaign as queue


PERIODS = (4, 8, 12, 16, 24, 32, 48, 64)


def periodic_steps(tokens):
    for end in range(2 * min(PERIODS), len(tokens)):
        for period in PERIODS:
            if end >= 2 * period and tokens[end - 2 * period:end - period] == tokens[end - period:end]:
                if tokens[end] == tokens[end - period]:
                    yield end
                    break


def number(text):
    match = re.fullmatch(r'\s*([-+]?\d+(?:\.\d+)?)\s*', text.replace(',', ''))
    return float(match[1]) if match else None


def score(tokenizer, rows, outputs, limit):
    results = []
    for row, tokens in zip(rows, outputs):
        completion = tokenizer.decode(tokens, skip_special_tokens=False)
        boxes = re.findall(r'\\boxed\{([^{}]+)\}', completion)
        gold = number(row['answer'].rsplit('####', 1)[-1])
        guess = number(boxes[-1]) if boxes else None
        results.append({'id': row['id'], 'correct': gold is not None and guess == gold,
                        'boxed_answer': guess, 'ended_thinking': '</think>' in completion,
                        'hit_cap': len(tokens) == limit, 'length': len(tokens),
                        'repeat_steps': sum(1 for _ in periodic_steps(tokens))})
    return results


def run(args):
    import pyarrow.parquet as parquet
    import torch
    import transformers
    from transformers import AutoModelForImageTextToText, AutoTokenizer, LogitsProcessor
    from wsp_functional import Patch, compile_patch, margins

    assets = args.assets
    manifest = json.loads((assets / 'manifest.json').read_text())
    for name, receipt in manifest['artifacts'].items():
        if queue.sha(assets / name) != receipt['sha256']:
            raise SystemExit('asset digest mismatch: ' + name)
    if torch.cuda.device_count() != 1:
        raise SystemExit('one assigned CUDA device required')
    model = AutoModelForImageTextToText.from_pretrained(
        assets / 'model', local_files_only=True, dtype=torch.bfloat16).to('cuda:0').eval()
    tokenizer = AutoTokenizer.from_pretrained(assets / 'model', local_files_only=True)
    tokenizer.padding_side = 'left'
    text_model = model.model.language_model
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    stop_ids = {eos} if isinstance(eos, int) else set(eos)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else min(stop_ids)

    class PresencePenalty(LogitsProcessor):
        def __init__(self, prompt_width):
            self.prompt_width = prompt_width

        def __call__(self, input_ids, scores):
            seen = torch.zeros_like(scores, dtype=torch.bool)
            seen.scatter_(1, input_ids[:, self.prompt_width:], True)
            return scores - 1.5 * seen

    def sample(split, count, split_seed):
        rows = parquet.read_table(assets / f'{split}-00000-of-00001.parquet').to_pylist()
        indices = list(range(len(rows)))
        random.Random(split_seed).shuffle(indices)
        start = args.shard * count
        return [dict(rows[i], id=i) for i in indices[start:start + count]]

    def prompts(rows):
        instruction = '\nPlease reason step by step, and put your final answer within \\boxed{}.'
        messages = [[{'role': 'user', 'content': row['question'] + instruction}] for row in rows]
        return [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                              enable_thinking=True) for m in messages]

    @torch.inference_mode()
    def generate(rows, limit):
        records = []
        for start in range(0, len(rows), args.batch):
            if time.time() >= args.deadline - 120:
                raise TimeoutError('no time for the next generation batch')
            torch.manual_seed(args.seed + start)
            batch = tokenizer(prompts(rows[start:start + args.batch]), return_tensors='pt',
                              add_special_tokens=False, padding=True).to('cuda:0')
            width = batch.input_ids.shape[1]
            outputs = model.generate(**batch, do_sample=True, temperature=.6, top_p=.95,
                                     top_k=20, max_new_tokens=limit, eos_token_id=list(stop_ids),
                                     pad_token_id=pad, logits_processor=[PresencePenalty(width)])
            for prefix, mask, answer in zip(batch.input_ids, batch.attention_mask, outputs[:, width:]):
                tokens = answer.tolist()
                end = next((i + 1 for i, token in enumerate(tokens) if token in stop_ids), len(tokens))
                records.append((prefix[mask.bool()].tolist(), tokens[:end]))
            print(json.dumps({'generated': len(records), 'limit': limit}), flush=True)
        return records

    @torch.inference_mode()
    def measure(rows):
        outputs = []
        # Single-item forwards avoid padding-dependent state for the hybrid text model.
        for row in rows:
            ids = torch.tensor([row], device='cuda')
            hidden = text_model(input_ids=ids, use_cache=False).last_hidden_state[:, -1]
            outputs.append(model.lm_head(hidden).float().cpu())
        return torch.cat(outputs)

    train = sample('train', args.train, 411)
    heldout = sample('test', args.test, 412)
    sequences = generate(train, args.tokens)
    positives, negatives, target = [], [], []
    train_rows = []
    for row, (prompt, tokens) in zip(train, sequences):
        bad = next((n for n in periodic_steps(tokens) if n >= 96), None)
        if bad is None:
            continue
        earlier = max(24, bad - min(128, bad // 2))
        positives.append(prompt + tokens[:bad])
        negatives.append(prompt + tokens[:earlier])
        target.append(tokens[bad])
        train_rows.append(row['id'])
    root = args.root
    queue.save(root / 'calibration.json', {'train_sample': [row['id'] for row in train],
                                           'loop_pair_ids': train_rows, 'token_cap': args.tokens,
                                           'model_revision': manifest['model_revision'],
                                           'train_outcomes': score(tokenizer, train, [t for _, t in sequences],
                                                                   args.tokens)})
    if len(positives) < 8:
        raise SystemExit('fewer than eight natural loop pairs at this token budget')

    indexes = (15, 19)
    capture = {index: [] for index in indexes}

    def hook(layer_index):
        def remember(_layer, inputs, output):
            capture[layer_index].append((inputs[0][0, -1].detach().float(),
                                         output[0, -1].detach().float()))
        return remember

    layers = [text_model.layers[i].mlp.down_proj for i in indexes]
    handles = [layer.register_forward_hook(hook(i)) for i, layer in zip(indexes, layers)]
    start_e = measure(positives)
    start_p = measure(negatives)
    for handle in handles:
        handle.remove()
    changes, selected_columns = [], []
    for i, layer in zip(indexes, layers):
        activation, output = (torch.stack(items) for items in zip(*capture[i]))
        n = len(positives)
        direction = output[:n].mean(0) - output[n:].mean(0)
        direction /= direction.norm().clamp_min(1e-12)
        writer_projection = direction @ layer.weight.detach().float()
        contrast = activation[:n].mean(0) - activation[n:].mean(0)
        selected = (contrast * writer_projection).topk(64).indices
        delta = torch.zeros_like(layer.weight, dtype=torch.float32)
        delta[:, selected] = -direction[:, None] * writer_projection[selected]
        changes.append([(layer.weight, delta)])
        selected_columns.append(selected.tolist())
    patch = Patch(changes)
    blocked = torch.tensor(target)[:, None]
    top = start_e.topk(2, dim=1).indices
    alternatives = torch.where(top[:, :1] == blocked, top[:, 1:], top[:, :1])
    reference_tokens = start_p.topk(9, dim=1).indices

    def measurement():
        return (margins(measure(positives), blocked, alternatives),
                margins(measure(negatives), reference_tokens[:, 1:], reference_tokens[:, :1]))

    desired = margins(start_e, blocked, alternatives) - 2
    preserved = margins(start_p, reference_tokens[:, 1:], reference_tokens[:, :1])
    coefficients, fit_info = compile_patch(patch, measurement, desired, preserved,
                                          float(patch.norms.norm()), 1.)
    queue.save(root / 'fit.json', {'diagnostic': fit_info, 'columns': selected_columns,
                                   'model_sha256': manifest['artifacts'],
                                   'torch': torch.__version__, 'transformers': transformers.__version__})
    for name, values in [('original', torch.zeros_like(coefficients)),
                         ('sparse_projection', patch.norms),
                         ('function_fit', coefficients)]:
        patch.apply(values)
        completions = [output for _, output in generate(heldout, args.tokens)]
        per_problem = score(tokenizer, heldout, completions, args.tokens)
        queue.save(root / (name + '.json'), {'weight_frobenius': float(values.norm()),
                                            'per_problem': per_problem})
    print(json.dumps({'status': 'completed', 'loop_pairs': len(positives), 'test': len(heldout)}),
          flush=True)


def supervise(args):
    while time.time() < args.deadline - 180:
        prior = args.after / 'plan.json'
        progress_file = args.after / f'lane{args.lane}' / 'progress.json'
        if prior.is_file() and progress_file.is_file():
            plan = json.loads(prior.read_text())
            expected = plan['devices'][args.lane]
            if expected['gpu'] != args.gpu or expected['uuid'] != args.uuid:
                raise SystemExit('predecessor GPU scope differs')
            if plan['deadline_epoch'] != args.deadline:
                raise SystemExit('predecessor deadline differs')
            status = json.loads(progress_file.read_text())['status']
            if status == 'cancelled':
                raise SystemExit('predecessor was cancelled')
            if status in ('completed_all_planned', 'failed', 'budget_exhausted'):
                try:
                    queue.device(args.gpu, args.uuid)
                except RuntimeError:
                    time.sleep(15)
                    continue
                if (args.assets / 'manifest.json').is_file():
                    break
        time.sleep(15)
    else:
        raise SystemExit('no GPU or validated assets before deadline')
    env = {k: v for k, v in os.environ.items() if k.lower() not in
           {'http_proxy','https_proxy','all_proxy','hf_token','hugging_face_hub_token','wandb_api_key'}}
    env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(args.gpu), HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
    with (args.root / 'payload.log').open('x') as log:
        arguments = [arg for arg in sys.argv[1:] if arg != '--supervise']
        child = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), '--worker',
                                  *arguments], env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            code = child.wait(timeout=max(1, args.deadline - time.time() - 20))
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            code = child.returncode
    queue.save(args.root / 'status.json', {'status': 'completed' if code == 0 else 'failed',
                                           'exit_code': code, 'deadline_epoch': args.deadline})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--supervise', action='store_true')
    group.add_argument('--worker', action='store_true')
    group.add_argument('--probe', action='store_true')
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--gpu', type=int, choices=(0, 1), required=True)
    parser.add_argument('--lane', type=int, choices=(0, 1), required=True)
    parser.add_argument('--uuid', required=True)
    parser.add_argument('--deadline', type=int, required=True)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--shard', type=int, choices=(0, 1, 2), default=0)
    parser.add_argument('--tokens', type=int, default=4096)
    parser.add_argument('--train', type=int, default=24)
    parser.add_argument('--test', type=int, default=48)
    parser.add_argument('--batch', type=int, default=4)
    args = parser.parse_args()
    if args.probe:
        import pyarrow.parquet as parquet
        from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer
        model_path = args.assets / 'model'
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        assert type(config) in AutoModelForImageTextToText._model_mapping
        assert config.model_type == 'qwen3_5'
        assert tokenizer.chat_template and tokenizer.eos_token_id is not None
        print(json.dumps({'model_type': config.model_type,
                          'train_rows': parquet.ParquetFile(args.assets / 'train-00000-of-00001.parquet').metadata.num_rows,
                          'test_rows': parquet.ParquetFile(args.assets / 'test-00000-of-00001.parquet').metadata.num_rows}))
    elif args.worker:
        run(args)
    elif args.supervise:
        supervise(args)
    else:
        if not 120 < args.deadline - time.time() <= 6 * 3600:
            raise SystemExit('deadline outside six-hour allocation')
        args.root.mkdir()
        with (args.root / 'supervisor.log').open('x') as log:
            child = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()),
                                      '--supervise', *sys.argv[1:]], start_new_session=True,
                                     stdout=log, stderr=subprocess.STDOUT)
        queue.save(args.root / 'schedule.json', {'pid': child.pid, 'gpu': args.gpu,
                                                   'uuid': args.uuid, 'deadline': args.deadline,
                                                   'after': str(args.after), 'seed': args.seed,
                                                   'shard': args.shard})
        print(json.dumps({'pid': child.pid, 'status': 'scheduled', 'root': str(args.root)}))


if __name__ == '__main__':
    main()
