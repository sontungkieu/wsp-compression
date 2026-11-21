"""Measure natural repetition and intentional copy on unmodified checkpoints."""
import argparse
from collections import Counter, deque
from decimal import Decimal
import hashlib
import inspect
from itertools import product
import json
from pathlib import Path
import random
import re
import statistics
import time
import unicodedata

CAPS = (512, 1024, 2048, 4096, 8192, 16384)
POLICIES = {
    'greedy': (False, 1., 1., 50, 0.),
    'sample': (True, .7, .9, 0, 0.),
    'astra': (True, .6, .95, 20, 1.5),
}
ASTRA_MATH = r"Please reason step by step, and put your final answer within \boxed{}."
VERIFIER_VERSION = 1
DECIMAL = re.compile(r'[-+]?\d+(?:,\d{3})*(?:\.\d+)?')
INTEGER_LIST = re.compile(r'\d+(?:\s*,\s*\d+)*')
FINAL = re.compile(r'(?im)^\s*final(?: answer)?\s*:\s*([^\n]+)')
BOXED = re.compile(r'\\boxed\{([^{}]+)\}')


def astra_instruction(arm):
    if arm == 'canonical':
        return ASTRA_MATH
    if arm == 'legacy':
        return ASTRA_MATH.replace(r'\boxed{}', chr(8) + 'oxed{}')
    raise ValueError('unknown Astra prompt arm')


def validate_astra_prompt(prompt, arm):
    controls = [ch for ch in prompt if unicodedata.category(ch) == 'Cc' and ch not in '\n\t']
    expected = [] if arm == 'canonical' else [chr(8)]
    if controls != expected or prompt.count(astra_instruction(arm)) != 1:
        raise ValueError(f'unexpected control character or instruction for {arm} Astra prompt')
    if arm == 'canonical' and (prompt.count(r'\boxed{}') != 1 or r'\\boxed{}' in prompt):
        raise ValueError('canonical Astra prompt requires one literal boxed directive')


def math_answer(row, text, pending_thought=True):
    """Read a numeric boxed answer, an explicit final-answer line, or the whole answer as a number."""
    _, closed, post = text.rpartition('</think>')
    answer = post.strip() if closed else text.strip()
    if '<think>' in answer or (pending_thought and not closed):
        answer = ''
    exact = BOXED.search(answer)
    exact_boxed = any(DECIMAL.fullmatch(m.group(1).strip()) and not answer[m.end():].strip(' .\n\t')
                      for m in BOXED.finditer(answer)) if exact else False
    body = answer.replace(chr(8) + 'oxed{', r'\boxed{')
    boxes = BOXED.findall(body)
    finals = FINAL.findall(body)
    explicit = boxes[-1] if boxes else finals[-1] if finals else body
    number = explicit.strip().removesuffix('.')
    parsed = number.replace(',', '') if DECIMAL.fullmatch(number) else None
    return {'exact_format': exact_boxed, 'answer_found': parsed is not None,
            'numeric_correct': parsed is not None and Decimal(parsed) == Decimal(row['gold']),
            'parsed_answer': parsed, 'think_closed': bool(closed)}


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


def panel(data, count, shard, astra_math=False, astra_arm='canonical', *, explicit_ids=None):
    import pyarrow.parquet as pq

    source = pq.read_table(data).to_pylist()
    if explicit_ids is None:
        indices = list(range(len(source)))
        random.Random(655231).shuffle(indices)
        start = shard * 32
        if count > 32 or start + count > len(indices):
            raise ValueError('GSM8K panel outside held-out shard')
        indices = indices[start:start + count]
    else:
        lookup = {f'gsm8k:{index}': index for index in range(len(source))}
        if (not astra_math or not 1 <= count <= 32 or len(set(explicit_ids)) != count
                or len(explicit_ids) != count or not set(explicit_ids).issubset(lookup)):
            raise ValueError('invalid explicit canonical GSM8K IDs')
        indices = [lookup[x] for x in explicit_ids]
    rows = []
    instruction = astra_instruction(astra_arm) if astra_math else 'Solve the problem. End with a line Final: followed by the numerical answer.'
    for index in indices:
        row = source[index]
        rows.append({'id': f'gsm8k:{index}', 'task': 'math',
                     'prompt': row['question'] + '\n' + instruction,
                     'boxed_answer_required': astra_math,
                     'gold': row['answer'].rsplit('####', 1)[-1].strip().replace(',', '')})
        if astra_math:
            rows[-1]['astra_prompt_arm'] = astra_arm
    if astra_math:
        return rows
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
        if row.get('boxed_answer_required'):
            answer = boxed[-1] if boxed else ''
        else:
            answer = matches[-1] if matches else boxed[-1] if boxed else answer
        answer = answer.strip(' .')
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
        prompt = 'Problem: ' + row['prompt'] + '\nResponse:'
    else:
        kwargs = {'enable_thinking': mode == 'think'} if mode in ('think', 'no-think') else {}
        prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': row['prompt']}],
                                               tokenize=False, add_generation_prompt=True, **kwargs)
    if arm := row.get('astra_prompt_arm'):
        validate_astra_prompt(prompt, arm)
    return prompt


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
            for field in ('correct', 'answer_found', 'eos', 'intended_repeat', 'think_closed'):
                counts[field] += int(score[field])
            counts['hit_cap'] += int(score['cap'])
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
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, LogitsProcessor
    from repetition_replay import terminal_cycle

    if torch.cuda.device_count() != 1:
        raise RuntimeError('exactly one assigned CUDA device required')
    manifest = json.loads((args.data.parent / 'manifest.json').read_text())
    data_sha = sha(args.data)
    if data_sha != manifest['artifacts'][args.data.name]['sha256']:
        raise ValueError('GSM8K digest mismatch')
    start = time.monotonic()
    explicit_ids = json.loads(args.panel_manifest.read_text())['ids'] if args.panel_manifest is not None else None
    rows = panel(args.data, args.count, args.shard, args.astra_math, explicit_ids=explicit_ids)
    panels = {'canonical' if args.astra_math else 'standard': rows}
    if args.astra_legacy_audit:
        panels['legacy'] = panel(args.data, args.count, args.shard, args.astra_math, astra_arm='legacy')
    model_type = json.loads((args.model / 'config.json').read_text())['model_type']
    pinned = {}
    loader = AutoModelForCausalLM
    if model_type == 'qwen3_5':
        loader = AutoModelForImageTextToText
        for name, receipt in manifest['artifacts'].items():
            relative = Path(name)
            if relative.parts[0] == 'model':
                actual = sha(args.model / relative.name)
                if actual != receipt['sha256']:
                    raise ValueError('pinned model digest mismatch: ' + relative.name)
                pinned[relative.name] = actual
        if not any(name.endswith('.safetensors') for name in pinned):
            raise ValueError('no pinned checkpoint weights')
    model = loader.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16).to('cuda:0').eval()
    defaults_arg = {'use_model_defaults': False} if 'use_model_defaults' in inspect.signature(model.generate).parameters else {}

    class PresencePenalty(LogitsProcessor):
        def __init__(self, input_width, amount):
            self.input_width = input_width
            self.amount = amount

        def __call__(self, input_ids, scores):
            seen = torch.zeros_like(scores, dtype=torch.bool)
            seen.scatter_(1, input_ids[:, self.input_width:], True)
            return scores - self.amount * seen
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
            if p.name in pinned or p.suffix != '.safetensors':
                artifacts[p.name]['sha256'] = pinned[p.name] if p.name in pinned else sha(p)
    app = {'record': 'manifest', 'artifact_root': str(args.model.resolve()),
           'artifacts': artifacts, 'source_sha': sha(Path(__file__)), 'seed': args.seed,
           'data_revision': manifest['gsm8k_revision'], 'data_sha': data_sha,
           'models_are_unmodified': True, 'torch': str(torch.__version__),
           'transformers': transformers.__version__, 'verifier': VERIFIER_VERSION,
           'panel': rows, 'eos_ids': sorted(stop_ids), 'batch': args.batch,
           'modes': args.mode, 'decoders': args.decoder, 'cap': args.cap,
           'data': str(args.data), 'shard': args.shard, 'sample_seeds': args.sample_seeds,
           'sample_temperature': .7, 'sample_top_p': .9, 'use_model_defaults': False,
           'model_revision': manifest['model_revision'] if pinned else None,
           'astra_math_prompt': args.astra_math, 'prompt_arms': list(panels),
           'decoder_settings': {name: dict(zip(('sample', 'temperature', 'top_p', 'top_k', 'presence_penalty'),
                                               POLICIES[name], strict=True)) for name in args.decoder}}
    if args.panel_manifest is not None:
        app['panel_manifest_sha256'] = sha(args.panel_manifest)
    with args.output.open('x', buffering=1) as log, torch.inference_mode():
        print(json.dumps(app), file=log)
        print('loaded', args.model.name, args.mode, len(rows), 'max_tokens', args.cap, flush=True)
        for mode, prompt_arm in product(args.mode, panels):
            rows = panels[prompt_arm]
            for decoder in args.decoder:
                do_sample, temperature, top_p, top_k, presence = POLICIES[decoder]
                config = transformers.GenerationConfig(
                    max_new_tokens=args.cap, do_sample=do_sample, eos_token_id=sorted(stop_ids),
                    pad_token_id=tokenizer.pad_token_id, temperature=temperature,
                    top_p=top_p, top_k=top_k,
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
                        processors = [PresencePenalty(inp.input_ids.shape[1], presence)] if presence else None
                        snippets = model.generate(**inp, generation_config=config,
                                                  logits_processor=processors, **defaults_arg)
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
                            extra = {}
                            if args.astra_math:
                                continuation = tokens if stop is None else tokens[:-1]
                                extra = {'prompt_arm': prompt_arm,
                                         'render_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                                         'terminal_cycle': terminal_cycle(continuation) is not None,
                                         **math_answer(row, text, pending)}
                            print(json.dumps({'record': 'rollout', 'id': row['id'], 'task': row['task'],
                                              'mode': mode, 'decoder': decoder, 'draw': draw, 'text': text,
                                              'token_ids': tokens, 'outcomes': results, **extra}, ensure_ascii=False), file=log)
                        print(mode, decoder, prompt_arm, draw, min(begin + args.batch, len(rows)), '/', len(rows),
                              'elapsed_seconds', round(time.monotonic() - start), flush=True)
        print(json.dumps({'record': 'completed', 'elapsed_seconds': time.monotonic() - start}), file=log)


def self_test():
    canonical = astra_instruction('canonical')
    legacy = astra_instruction('legacy')
    assert canonical.encode().count(bytes([92]) + b'boxed{}') == 1 and chr(8) not in canonical
    assert legacy.count(chr(8)) == 1 and chr(8) + 'oxed{}' in legacy

    class TestTemplate:
        def apply_chat_template(self, messages, **kwargs):
            return 'system\n' + messages[0]['content'] + '\n<think>'

    template = TestTemplate()
    for arm, instruction in (('canonical', canonical), ('legacy', legacy)):
        row = {'prompt': 'How many?\n' + instruction, 'astra_prompt_arm': arm}
        checked = render(template, row, 'think')
        assert checked.count(instruction) == 1
        try:
            render(template, {**row, 'prompt': row['prompt'] + chr(7)}, 'think')
        except ValueError:
            pass
        else:
            raise AssertionError('invalid rendered control character accepted')
    g = {'gold': '42'}
    good = math_answer(g, '</think>\n' + r'\boxed{42}')
    old = math_answer(g, '</think>\n' + chr(8) + 'oxed{42}')
    assert good['exact_format'] and good['numeric_correct']
    assert not old['exact_format'] and old['numeric_correct']
    assert math_answer(g, '</think>\nFinal answer: 42.')['numeric_correct']
    assert not math_answer(g, '</think>\n6 times 7 is 42')['answer_found']
    assert not math_answer(g, r'\boxed{42}')['answer_found']
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
    assert assess({**math, 'boxed_answer_required': True}, r'</think>\boxed{12}', [1], True, 5)['correct']
    assert not assess({**math, 'boxed_answer_required': True}, 'Final: 12', [1], True, 5)['correct']
    assert not assess(math, 'Final: 11', [1], True, 5)['correct']
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:
        path = Path(directory) / 'two_caps.jsonl'
        row = {'record': 'rollout', 'mode': 'base', 'decoder': 'greedy', 'task': 'math',
               'outcomes': {'512': {**correct, 'cap': True}, '1024': correct}}
        path.write_text(json.dumps(row))
        counts = {r['cap']: r['hit_cap'] for r in summarize(path)}
        assert counts == {512: 1, 1024: 0}
    print('baseline detector, verifier and summary checks passed')


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
    parser.add_argument('--astra-math', action='store_true', help='use the released boxed-answer prompt on GSM8K only')
    parser.add_argument('--astra-legacy-audit', action='store_true', help='pair canonical with the historical malformed prompt')
    parser.add_argument('--seed', type=int, default=1944)
    parser.add_argument('--sample-seeds', type=int, default=1)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--panel-manifest', type=Path)
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
        args.decoder = args.decoder or ['greedy', 'sample']
        if args.astra_legacy_audit and (not args.astra_math or args.mode != ['think'] or args.decoder != ['astra']):
            parser.error('legacy audit requires Astra math, thinking mode, and the Astra decoder')
        if args.panel_manifest and (not args.astra_math or args.astra_legacy_audit):
            parser.error('an explicit panel is supported for canonical Astra math')
        run(args)


if __name__ == '__main__':
    main()
