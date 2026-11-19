"""Replay audited Llama GSM8K stalls with paired decoder and information treatments."""
import argparse
from collections import defaultdict
from decimal import Decimal
import json
from pathlib import Path
import random
import re
import statistics
import time

from repetition_baseline import assess, render, sha

PHASES = ('cross', 'replicate_cross')
MINIMUM = 30
FACT = re.compile(r'<<([0-9.()+*/ \-]+)=([-+]?\d+(?:\.\d+)?)>>')


def terminal_cycle(tokens):
    """Find a sustained cycle in the exact tail, returning period and replay cutoff."""
    n = len(tokens)
    for period in range(1, min(128, n // 5) + 1):
        start = n - 1
        while start >= period and tokens[start] == tokens[start - period]:
            start -= 1
        start = start + 1 - period
        width = n - start
        if width >= max(128, 5 * period):
            return period, start + 5 * period
    return None


def select_information(solution, gold):
    for match in FACT.finditer(solution.split('####')[0]):
        equation, intermediate = match.groups()
        if Decimal(intermediate) != Decimal(gold):
            return f'{equation.strip()} = {intermediate}'
    return None


def cases(root, data):
    import pyarrow.parquet as pq

    source = pq.read_table(data, columns=['answer']).to_pylist()
    status = json.loads((root / 'status.json').read_text())
    jobs = {j['name']: j for j in status['jobs'] if j['status'] == 'completed'}
    selected = {}
    verified = set()
    model = None
    for phase in PHASES:
        if phase not in jobs:
            continue
        path = root / (phase + '.jsonl')
        if sha(path) != jobs[phase]['sha256']:
            raise RuntimeError('baseline result digest mismatch')
        with path.open() as stream:
            manifest = json.loads(next(stream))
            if (manifest['data_sha'] != sha(data) or manifest.get('use_model_defaults') is not False
                    or Path(manifest['artifact_root']).name != 'Llama-3.2-3B-Base'):
                raise RuntimeError('baseline checkpoint, decoder or data mismatch')
            identity = (manifest['artifact_root'], manifest['artifacts'], manifest['eos_ids'])
            if model is not None and model != identity:
                raise RuntimeError('matched panels use different model or tokenizer artifacts')
            model = identity
            gold = {p['id']: p for p in manifest['panel']}
            for line in stream:
                row = json.loads(line)
                if (row['record'] != 'rollout' or row['task'] != 'math' or row['mode'] != 'base'
                        or row['decoder'] != 'greedy' or not row['outcomes']['2048']['cap']):
                    continue
                cycle = terminal_cycle(row['token_ids'])
                if cycle is None:
                    continue
                key = row['id']
                verified.add(key)
                p = gold[key]
                fact = select_information(source[int(key.split(':')[1])]['answer'], p['gold'])
                if fact is not None:
                    period, cutoff = cycle
                    selected[key] = {'row': p, 'completion_prefix': row['token_ids'][:cutoff], 'period': period,
                                     'phase': phase, 'fact': fact}
    return model, list(selected.values()), len(verified)


def with_notes(tokenizer, prefix, fact):
    start = tokenizer.encode('\n\n[Additional context]\n', add_special_tokens=False)
    end = tokenizer.encode('\n[/Additional context]\nContinue solving. End with Final: followed by the numerical answer.\n',
                           add_special_tokens=False)
    fact_ids = tokenizer.encode('The calculation ' + fact + ' is correct.', add_special_tokens=False)
    placebo_ids = tokenizer.encode('Consider the original problem carefully.', add_special_tokens=False)
    filler = tokenizer.encode(' ', add_special_tokens=False)
    if len(filler) != 1:
        raise ValueError('expected one neutral padding token for the matched control')
    width = max(len(fact_ids), len(placebo_ids))
    def attach(ids):
        return prefix + start + ids + filler * (width - len(ids)) + end
    return {'plain': prefix, 'fact': attach(fact_ids), 'placebo': attach(placebo_ids)}


def run(args, identity, selected, verified):
    import torch
    import transformers

    if torch.cuda.device_count() != 1:
        raise RuntimeError('exactly one assigned CUDA device required')
    path, artifacts, eos = identity
    path = Path(path)
    for name, metadata in artifacts.items():
        file = path / name
        st = file.stat()
        if st.st_size != metadata['bytes'] or st.st_mtime_ns != metadata['mtime_ns']:
            raise RuntimeError('checkpoint artifact changed: ' + name)
        if 'sha256' in metadata and sha(file) != metadata['sha256']:
            raise RuntimeError('checkpoint artifact digest mismatch: ' + name)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, dtype=torch.bfloat16).to('cuda:0').eval()
    tok = transformers.AutoTokenizer.from_pretrained(path, local_files_only=True)
    pad = tok.pad_token_id if tok.pad_token_id is not None else min(eos)
    items = []
    for item in selected:
        prompt = tok.encode(render(tok, item['row'], 'base'), add_special_tokens=False)
        prior = item['completion_prefix']
        prefix = prompt + prior
        seen = tok.decode(prior, skip_special_tokens=False)
        items.append({**item, 'prefixes': with_notes(tok, prefix, item['fact']),
                      'prior_final': assess(item['row'], seen, prior, False, 2048)['correct'],
                      'fact_verbatim_in_original_prefix': item['fact'] in tok.decode(prefix)})
    header = {'kind': 'manifest', 'verified_unique': verified, 'eligible_unique': len(items),
              'model': str(path), 'source_sha': sha(Path(__file__)), 'data_sha': sha(args.data),
              'cap': args.cap, 'draws': args.draws, 'min_cases': args.min_cases, 'eos_ids': eos,
              'verified_cases': [{'id': i['row']['id'], 'phase': i['phase'], 'period': i['period'],
                                  'prior_final': i['prior_final'], 'hint': i['fact'],
                                  'fact_verbatim_in_original_prefix': i['fact_verbatim_in_original_prefix'],
                                  'prefix_len': {k: len(v) for k, v in i['prefixes'].items()}} for i in items]}
    start = time.monotonic()
    with args.output.open('x', buffering=1) as out, torch.inference_mode():
        print(json.dumps(header), file=out)
        print('start replay', len(items), 'of', verified, 'verified prompts', flush=True)
        for sampled in (False, True):
            config = transformers.GenerationConfig(
                max_new_tokens=args.cap, do_sample=sampled, eos_token_id=eos, pad_token_id=pad,
                temperature=.7 if sampled else 1., top_p=.9 if sampled else 1.,
                top_k=0 if sampled else 50, repetition_penalty=1., use_cache=True)
            for draw in range(args.draws if sampled else 1):
                for begin in range(0, len(items), args.batch):
                    group = items[begin:begin + args.batch]
                    for context in ('plain', 'placebo', 'fact'):
                        if time.time() >= args.deadline - 120:
                            raise TimeoutError('approaching the original absolute deadline')
                        torch.manual_seed(28937 + 10000 * draw + begin)
                        parts = [i['prefixes'][context] for i in group]
                        width = max(map(len, parts))
                        ids = torch.tensor([[pad] * (width - len(p)) + p for p in parts], device='cuda')
                        mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in parts], device='cuda')
                        generated = model.generate(input_ids=ids, attention_mask=mask, generation_config=config,
                                                   use_model_defaults=False)
                        for item, row in zip(group, generated[:, width:], strict=True):
                            tokens = row.tolist()
                            stop = next((i for i, token in enumerate(tokens) if token in eos), None)
                            tokens = tokens if stop is None else tokens[:stop]
                            text = tok.decode(tokens, skip_special_tokens=False)
                            score = assess(item['row'], text, tokens, stop is not None, args.cap)
                            cycle = terminal_cycle(tokens)
                            print(json.dumps({'kind': 'rollout', 'id': item['row']['id'],
                                              'origin': item['phase'], 'draw': draw,
                                              'decoder': 'sample' if sampled else 'greedy', 'context': context,
                                              'eos': stop is not None, 'correct_final': score['correct'],
                                              'terminal_cycle': cycle, 'good_termination': score['correct'] and stop is not None,
                                              'text': text, 'continuation_ids': tokens}), file=out)
                    print('sample' if sampled else 'greedy', draw, min(begin + args.batch, len(items)),
                          'of', len(items), 'seconds', round(time.monotonic() - start), flush=True)
        print(json.dumps({'kind': 'complete', 'seconds': time.monotonic() - start}), file=out)


def summary(path):
    paired = defaultdict(list)
    finished = False
    with path.open() as stream:
        head = json.loads(next(stream))
        for line in stream:
            row = json.loads(line)
            if row['kind'] == 'complete':
                finished = True
            if row['kind'] == 'rollout':
                key = row['decoder'], row['context'], row['id']
                paired[key].append(row)
    results = {}
    for decoder in ('greedy', 'sample'):
        for context in ('plain', 'placebo', 'fact'):
            rows = [values for (d, c, _), values in paired.items() if (d, c) == (decoder, context)]
            if rows:
                results[f'{decoder}:{context}'] = {'prompts': len(rows), 'rollouts': sum(map(len, rows))}
                for field in ('correct_final', 'good_termination', 'eos', 'terminal_cycle'):
                    results[f'{decoder}:{context}'][field] = statistics.fmean(
                        statistics.fmean(bool(r[field]) for r in values) for values in rows)
    return {'complete': finished, 'eligible': head['eligible_unique'], 'verified': head['verified_unique'], 'cells': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summarize', type=Path)
    parser.add_argument('--baseline-lane', type=Path)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--deadline', type=int)
    parser.add_argument('--cap', type=int, default=512)
    parser.add_argument('--draws', type=int, default=8)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--min-cases', type=int, default=MINIMUM)
    parser.add_argument('--inventory', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        assert terminal_cycle(list(range(7)) * 20) == (7, 35)
        assert terminal_cycle(list(range(32)) * 4) is None
        assert terminal_cycle([99] + list(range(32)) * 5) == (32, 161)
        assert select_information('We found <<12*3=36>> and <<36-10=26>>. #### 26', '26') == '12*3 = 36'
        assert select_information('We found <<36-10=26>>. #### 26', '26') is None
        print('exact-cycle and intermediate-information checks passed')
        return
    if args.summarize:
        print(json.dumps(summary(args.summarize)))
        return
    if not args.baseline_lane or not args.data:
        parser.error('baseline lane and pinned dataset required')
    identity, selected, verified = cases(args.baseline_lane, args.data)
    print(json.dumps({'exact_tail_verified': verified, 'valid_cue': len(selected), 'gate': args.min_cases}), flush=True)
    if args.inventory:
        return
    if not args.output or not args.deadline or min(args.cap, args.batch, args.draws, args.min_cases) < 1:
        parser.error('invalid output, deadline or sampling contract')
    if len(selected) < args.min_cases:
        raise SystemExit('too few independently verified prompts with an intermediate cue')
    run(args, identity, selected, verified)


if __name__ == '__main__':
    main()
