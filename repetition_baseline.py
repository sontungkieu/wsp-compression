"""Measure natural repetition and intentional copy on unmodified checkpoints."""
import argparse
from collections import Counter, deque
import hashlib
import json
from pathlib import Path
import random
import re
import statistics
import time

CAPS = (512, 1024, 2048)
POLICIES = ('greedy', 'sample')
VERIFIER_VERSION = 1
DECIMAL = re.compile(r'[-+]?\d+(?:,\d{3})*(?:\.\d+)?')
INTEGER_LIST = re.compile(r'\d+(?:\s*,\s*\d+)*')
FINAL = re.compile(r'(?im)^\s*final(?: answer)?\s*:\s*([^\n]+)')


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def first_period(tokens, copies=4, max_period=48):
    # Count consecutive matches at each lag. At lag p, (k-1)p matches prove k copies.
    earliest = None
    for period in range(1, min(max_period, len(tokens) // copies) + 1):
        run = 0
        for i in range(period, earliest[0] if earliest else len(tokens)):
            run = run + 1 if tokens[i] == tokens[i - period] else 0
            if run == (copies - 1) * period:
                earliest = (i + 1, period)
                break
    return earliest


def first_recurrent(tokens, ngram=16):
    last, recent = {}, deque()
    for end in range(ngram, len(tokens) + 1):
        key = tuple(tokens[end - ngram:end])
        previous = last.get(key)
        last[key] = end
        if previous is not None and end - previous <= 2048:
            while recent and recent[0] <= end - 128:
                recent.popleft()
            recent.append(end)
            if len(recent) == 8:
                return end
    return None


def panel(data, count, shard):
    import pyarrow.parquet as pq

    source = pq.read_table(data).to_pylist()
    indices = list(range(len(source)))
    random.Random(655231).shuffle(indices)
    start = shard * 32
    if count > 32 or start + count > len(indices):
        raise ValueError('GSM8K panel outside held-out shard')
    rows = []
    for index in indices[start:start + count]:
        row = source[index]
        rows.append({'id': f'gsm8k:{index}', 'task': 'math',
                     'prompt': row['question'] + '\nSolve the problem. End with a line Final: followed by the numerical answer.',
                     'gold': row['answer'].rsplit('####', 1)[-1].strip().replace(',', '')})
    rng = random.Random(923311 + shard)
    for i in range(min(count, 24)):
        start_n, divisor = rng.randrange(11, 57), rng.randrange(2, 6)
        span = (18, 35, 60)[i % 3]
        stop = start_n + span
        gold = [n for n in range(start_n, stop + 1) if n % divisor == 0]
        rows.append({'id': f'enumeration:{shard}:{i}', 'task': 'enumeration', 'gold': gold,
                     'prompt': f'List all multiples of {divisor} from {start_n} to {stop} inclusive, in ascending order. '
                               'Give a single line containing only the numbers separated by commas.'})
        motif = ''.join(rng.choices('bcdfghjklmnprstvz', k=6))
        copies = (1, 3, 5, 9)[i % 4]
        gold = ' | '.join([motif] * copies)
        rows.append({'id': f'copy:{shard}:{i}', 'task': 'copy', 'gold': gold,
                     'prompt': f'Write the exact string {motif} exactly {copies} times. '
                               'Separate occurrences with a space, a vertical bar, and a space. '
                               'For one occurrence, output only the string. Output nothing else.'})
    return rows


def assess(row, text, tokens, eos, cap, pending_thought=False):
    # The closing tag separates scratchpad from the answer. Open thinking is incomplete.
    _, closed, post = text.rpartition('</think>')
    answer = post.strip() if closed else text.strip()
    if '<think>' in answer or (pending_thought and not closed):
        answer = ''
    task = row['task']
    duplicate_items = None
    state_loop = None
    if task == 'math':
        matches = FINAL.findall(answer)
        boxed = re.findall(r'\\boxed\{([^{}]+)\}', answer)
        answer = (matches[-1] if matches else boxed[-1] if boxed else answer).strip(' .')
        answer_found = bool(DECIMAL.fullmatch(answer))
        correct = answer_found and float(answer.replace(',', '')) == float(row['gold'])
    elif task == 'enumeration':
        # A closed-form list is scored only when the answer is parseable as a list.
        answer = answer.rstrip('.')
        answer_found = bool(INTEGER_LIST.fullmatch(answer))
        items = [int(x.strip()) for x in answer.split(',')] if answer_found else []
        correct = answer_found and items == row['gold']
        if answer_found:
            allowed = set(row['gold'])
            seen = set()
            stall = 0
            state_loop = False
            duplicate_items = len(items) - len(set(items))
            for item in items:
                if item in allowed and item not in seen:
                    seen.add(item)
                    stall = 0
                else:
                    stall += 1
                    if stall >= 3 and item in seen:
                        state_loop = True
    else:
        answer_found = bool(answer)
        correct = answer == row['gold']
    periodic = first_period(tokens)
    recurrent = first_recurrent(tokens)
    return {'tokens': len(tokens), 'cap': len(tokens) == cap and not eos, 'eos': eos,
            'answer_found': answer_found, 'correct': correct, 'think_closed': bool(closed),
            'periodic': periodic, 'recurrent16': recurrent,
            'duplicate_items': duplicate_items, 'enumeration_stall_proxy': state_loop,
            'intended_repeat': task == 'copy' and correct and '|' in row['gold']}


def render(tokenizer, row, mode):
    if mode == 'base':
        return 'Problem: ' + row['prompt'] + '\nResponse:'
    kwargs = {'enable_thinking': mode == 'think'} if mode in ('think', 'no-think') else {}
    return tokenizer.apply_chat_template([{'role': 'user', 'content': row['prompt']}],
                                         tokenize=False, add_generation_prompt=True, **kwargs)


def score_rollout(row, text, tokens, eos, cap, tokenizer, pending_thought):
    caps = sorted({c for c in CAPS if c <= cap} | {cap})
    outcomes = {}
    for limit in caps:
        prefix = tokens[:limit]
        visible = text if len(tokens) <= limit else tokenizer.decode(prefix, skip_special_tokens=False)
        outcomes[str(limit)] = assess(row, visible, prefix, eos and len(tokens) <= limit,
                                      limit, pending_thought)
    return outcomes


def summarize(path):
    groups = {}
    for raw in path.read_text().splitlines():
        row = json.loads(raw)
        if row['record'] != 'rollout':
            continue
        for cap, score in row['outcomes'].items():
            key = row['mode'], row['decoder'], row['task'], cap
            counts, lengths = groups.setdefault(key, (Counter(), []))
            counts['n'] += 1
            for field in ('correct', 'answer_found', 'eos', 'cap', 'intended_repeat', 'think_closed'):
                counts[field] += int(score[field])
            for field in ('periodic', 'recurrent16', 'enumeration_stall_proxy'):
                counts[field] += int(bool(score[field]))
                counts[field + '_and_incorrect'] += int(bool(score[field]) and not score['correct'])
                counts[field + '_and_intended_repeat'] += int(bool(score[field]) and score['intended_repeat'])
            counts['enumeration_parseable'] += int(score['duplicate_items'] is not None)
            lengths.append(score['tokens'])
    return [{'mode': mode, 'decoder': decoder, 'task': task, 'cap': int(cap),
             'p50_tokens': statistics.median(lengths), **dict(counts)}
            for (mode, decoder, task, cap), (counts, lengths) in sorted(groups.items())]


def run(args):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.device_count() != 1:
        raise RuntimeError('exactly one assigned CUDA device required')
    manifest = json.loads((args.data.parent / 'manifest.json').read_text())
    data_sha = sha(args.data)
    if data_sha != manifest['artifacts'][args.data.name]['sha256']:
        raise ValueError('GSM8K digest mismatch')
    start = time.monotonic()
    rows = panel(args.data, args.count, args.shard)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16).to('cuda:0').eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = 'left'
    eos = model.generation_config.eos_token_id
    eos = eos if eos is not None else tokenizer.eos_token_id
    stop_ids = {eos} if isinstance(eos, int) else set(eos or ())
    if not stop_ids:
        raise ValueError('could not determine native EOS')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = min(stop_ids)
    artifacts = {}
    for p in sorted(args.model.iterdir()):
        if p.is_file() and (p.suffix in ('.json', '.jinja', '.txt', '.model') or
                            p.suffix == '.safetensors'):
            stat = p.stat()
            artifacts[p.name] = {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
            if p.suffix != '.safetensors':
                artifacts[p.name]['sha256'] = sha(p)
    app = {'record': 'manifest', 'artifact_root': str(args.model.resolve()),
           'artifacts': artifacts, 'source_sha': sha(Path(__file__)), 'seed': args.seed,
           'data_revision': manifest['gsm8k_revision'], 'data_sha': data_sha,
           'models_are_unmodified': True, 'torch': str(torch.__version__),
           'transformers': transformers.__version__, 'verifier': VERIFIER_VERSION,
           'panel': rows, 'eos_ids': sorted(stop_ids), 'batch': args.batch,
           'modes': args.mode, 'decoders': args.decoder, 'cap': args.cap,
           'data': str(args.data), 'shard': args.shard, 'sample_seeds': args.sample_seeds,
           'sample_temperature': .7, 'sample_top_p': .9, 'use_model_defaults': False}
    with args.output.open('x', buffering=1) as log, torch.inference_mode():
        print(json.dumps(app), file=log)
        print('loaded', args.model.name, args.mode, len(rows), 'max_tokens', args.cap, flush=True)
        for mode in args.mode:
            for decoder in args.decoder:
                do_sample = decoder == 'sample'
                config = transformers.GenerationConfig(
                    max_new_tokens=args.cap, do_sample=do_sample, eos_token_id=sorted(stop_ids),
                    pad_token_id=tokenizer.pad_token_id, temperature=.7 if do_sample else 1.,
                    top_p=.9 if do_sample else 1., top_k=0 if do_sample else 50,
                    repetition_penalty=1., use_cache=True)
                for draw in range(args.sample_seeds if do_sample else 1):
                    for begin in range(0, len(rows), args.batch):
                        if time.time() >= args.deadline - 120:
                            raise TimeoutError('approaching absolute deadline')
                        torch.manual_seed(args.seed + 10000 * draw + begin)
                        batch_rows = rows[begin:begin + args.batch]
                        prompts = [render(tokenizer, row, mode) for row in batch_rows]
                        inp = tokenizer(prompts, add_special_tokens=False, padding=True,
                                        return_tensors='pt').to('cuda:0')
                        snippets = model.generate(**inp, generation_config=config, use_model_defaults=False)
                        for row, prompt, output in zip(batch_rows, prompts,
                                                       snippets[:, inp.input_ids.shape[1]:], strict=True):
                            tokens = output.tolist()
                            stop = next((i for i, token in enumerate(tokens) if token in stop_ids), None)
                            tokens = tokens if stop is None else tokens[:stop + 1]
                            text = tokenizer.decode(tokens, skip_special_tokens=False)
                            # Keep the thinking delimiter for strict final scoring and strip native stop tokens.
                            if stop is not None:
                                eos_text = tokenizer.decode(tokens[-1:], skip_special_tokens=False)
                                if eos_text:
                                    text = text.removesuffix(eos_text)
                            pending = mode == 'think' or prompt.rfind('<think>') > prompt.rfind('</think>')
                            results = score_rollout(row, text, tokens, stop is not None,
                                                    args.cap, tokenizer, pending)
                            print(json.dumps({'record': 'rollout', 'id': row['id'], 'task': row['task'],
                                              'mode': mode, 'decoder': decoder, 'draw': draw, 'text': text,
                                              'token_ids': tokens, 'outcomes': results}, ensure_ascii=False), file=log)
                        print(mode, decoder, draw, min(begin + args.batch, len(rows)), '/', len(rows),
                              'elapsed_seconds', round(time.monotonic() - start), flush=True)
        print(json.dumps({'record': 'completed', 'elapsed_seconds': time.monotonic() - start}), file=log)


def self_test():
    assert first_period([1, 2, 1, 2, 1, 2, 1, 2]) == (8, 2)
    assert first_period([0] * 8) == (4, 1)
    assert first_period(list(range(90))) is None
    assert first_recurrent(list(range(32)) * 3) is not None
    assert first_recurrent(list(range(96))) is None
    enum = {'task': 'enumeration', 'gold': [2, 4, 6]}
    correct = assess(enum, '2, 4, 6', [2, 4, 6], True, 5)
    assert correct['correct'] and not correct['enumeration_stall_proxy']
    stalled = assess(enum, '2, 4, 4, 4, 4', [2, 4, 4, 4, 4], False, 5)
    assert stalled['enumeration_stall_proxy'] and stalled['cap'] and not stalled['correct']
    assert not assess(enum, '2, 2, 4, 4, 6', [2, 2, 4, 4, 6], True, 5)['enumeration_stall_proxy']
    assert assess(enum, 'missing', [2], True, 5)['enumeration_stall_proxy'] is None
    copy = {'task': 'copy', 'gold': 'xyz | xyz'}
    assert assess(copy, 'xyz | xyz', [7, 8, 7], True, 9)['intended_repeat']
    assert not assess(copy, 'xyz | xyz | xyz', [7, 8, 7, 8, 7], True, 9)['correct']
    math = {'task': 'math', 'gold': '12'}
    assert assess(math, '<think> Final: 4\n</think>\nFinal: 12', [1], True, 5)['correct']
    assert not assess(math, '<think>\nFinal: 12', [1], False, 5)['correct']
    assert not assess(math, 'Final: 12', [1], False, 5, pending_thought=True)['correct']
    assert assess(math, '4</think>Final: 12', [1], True, 5, pending_thought=True)['correct']
    assert not assess(math, 'Final: 11', [1], True, 5)['correct']
    print('baseline detector and verifier checks passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--summarize', type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--deadline', type=int)
    parser.add_argument('--mode', choices=('base', 'chat', 'think', 'no-think'), action='append')
    parser.add_argument('--decoder', choices=POLICIES, action='append')
    parser.add_argument('--seed', type=int, default=1944)
    parser.add_argument('--sample-seeds', type=int, default=1)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--count', type=int, default=32)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--cap', type=int, default=2048)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.summarize:
        print(json.dumps(summarize(args.summarize)))
    else:
        if not all((args.model, args.data, args.output, args.deadline, args.mode)):
            parser.error('model, data, output, deadline and mode are required')
        if min(args.count, args.batch, args.cap, args.sample_seeds) <= 0 or args.shard < 0:
            parser.error('invalid panel or generation settings')
        args.decoder = args.decoder or list(POLICIES)
        run(args)


if __name__ == '__main__':
    main()
