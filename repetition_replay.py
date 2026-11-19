"""Replay audited GSM8K stalls with paired decoder and information treatments."""
import argparse
from collections import defaultdict
from decimal import Decimal
import hashlib
import inspect
from itertools import combinations, product
import json
from pathlib import Path
import random
import re
import statistics
import time

from repetition_baseline import assess, render, sha

ASTRA_REVISION = '2fc06364715b967f1860aea9cf38778875588b17'
LLAMA_PHASES = ('cross', 'replicate_cross')
ASTRA_PHASES = ('primary-a', 'primary-b')
MINIMUM = 30
FACT = re.compile(r'<<([0-9.()+*/ \-]+)=([-+]?\d+(?:\.\d+)?)>>')
NOTE_TEXT = {
    'neutral': '',
    'continue': 'Keep solving the problem from the current step.',
    'finalize': 'Give your final answer to the problem now.',
}
CONTEXT_OPEN = '\n\n[Additional context]\n'
CONTEXT_CLOSE = '\n[/Additional context]\n'
FRAME_FILLERS = {'spaces': ' ', 'newlines': '\n', 'periods': '.'}
FRAME_ARMS = {f'{frame}_{name}': {'frame': frame == 'framed', 'filler': filler}
              for frame in ('bare', 'framed') for name, filler in FRAME_FILLERS.items()}
FRAME_COMPONENTS = ('words', 'brackets', 'edges')
PUNCTUATION_COMPONENTS = ('opener', 'closer', 'slash')
CLOSING_MARKS = {'none': '', 'left_square': '[', 'right_square': ']', 'square_pair': '[]', 'round_pair': '()'}
REPLAY_SEED = 28937


def factorial_arms(parts):
    return {'_'.join(name for name, present in zip(parts, flags) if present) or 'none':
            dict(zip(parts, flags)) for flags in product((False, True), repeat=len(parts))}


COMPONENT_ARMS = factorial_arms(FRAME_COMPONENTS)
PUNCTUATION_ARMS = factorial_arms(PUNCTUATION_COMPONENTS)


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


def cases(root, data, astra=False):
    import pyarrow.parquet as pq

    source = pq.read_table(data, columns=['answer']).to_pylist()
    phases = ASTRA_PHASES if astra else LLAMA_PHASES
    cap, mode, decoder = (8192, 'think', 'astra') if astra else (2048, 'base', 'greedy')
    data_sha = sha(data)
    status = json.loads((root / 'status.json').read_text())
    jobs = {j['name']: j for j in status['jobs'] if j['status'] == 'completed'}
    selected = {}
    verified = set()
    model = None
    for phase in phases:
        if phase not in jobs:
            continue
        path = root / (phase + '.jsonl')
        if sha(path) != jobs[phase]['sha256']:
            raise RuntimeError('baseline result digest mismatch')
        with path.open() as stream:
            manifest = json.loads(next(stream))
            if astra:
                checkpoint_matches = (manifest.get('model_revision') == ASTRA_REVISION
                                      and manifest.get('astra_math_prompt') is True
                                      and manifest['decoder_settings']['astra']['presence_penalty'] == 1.5)
            else:
                checkpoint_matches = Path(manifest['artifact_root']).name == 'Llama-3.2-3B-Base'
            if (manifest['data_sha'] != data_sha or manifest.get('use_model_defaults') is not False
                    or not checkpoint_matches or manifest['cap'] != cap):
                raise RuntimeError('baseline checkpoint, decoder or data mismatch')
            identity = (manifest['artifact_root'], manifest['artifacts'], manifest['eos_ids'])
            if model is not None and model != identity:
                raise RuntimeError('matched panels use different model or tokenizer artifacts')
            model = identity
            gold = {p['id']: p for p in manifest['panel']}
            for line in stream:
                row = json.loads(line)
                if (row['record'] != 'rollout' or row['task'] != 'math' or row['mode'] != mode
                        or row['decoder'] != decoder or not row['outcomes'][str(cap)]['cap']):
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


def with_notes(tokenizer, prefix, fact, astra=False):
    start = tokenizer.encode('\n\n[Additional context]\n', add_special_tokens=False)
    instructions = ('Continue solving. Give the final numerical answer in a box.'
                    if astra else 'Continue solving. End with Final: followed by the numerical answer.')
    end = tokenizer.encode('\n[/Additional context]\n' + instructions + '\n', add_special_tokens=False)
    fact_ids = tokenizer.encode('The calculation ' + fact + ' is correct.', add_special_tokens=False)
    placebo_ids = tokenizer.encode('Consider the original problem carefully.', add_special_tokens=False)
    filler = tokenizer.encode(' ', add_special_tokens=False)
    if len(filler) != 1:
        raise ValueError('expected one neutral padding token for the matched control')
    width = max(len(fact_ids), len(placebo_ids))
    def attach(ids):
        return prefix + start + ids + filler * (width - len(ids)) + end
    return {'plain': prefix, 'fact': attach(fact_ids), 'placebo': attach(placebo_ids)}


def prior_replay(path):
    status = json.loads((path / 'status.json').read_text())
    result = path / 'replay.jsonl'
    if status['stage'] != 'completed' or sha(result) != status['sha256']:
        raise ValueError('expected a completed, checksum-verified replay')
    with result.open() as stream:
        manifest = json.loads(next(stream))
    if (manifest['mode'] != 'think' or manifest['sampling'] != {
            'temperature': .6, 'top_p': .95, 'top_k': 20, 'presence_penalty': 1.5}):
        raise ValueError('parent replay used different decoding')
    return manifest, status['sha256'], status['deadline']


def with_matched_notes(tokenizer, prefix, inserted_tokens):
    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    start = encode(CONTEXT_OPEN)
    end = encode(CONTEXT_CLOSE)
    space = encode(' ')
    if len(space) != 1:
        raise ValueError('expected one space token for the neutral condition')
    body = inserted_tokens - len(start) - len(end)
    messages = {condition: encode(content) if content else [] for condition, content in NOTE_TEXT.items()}
    if len(messages['continue']) != len(messages['finalize']):
        raise ValueError('the two instructions must use the same number of tokens')
    result = {}
    for condition, tokens in messages.items():
        if len(tokens) > body:
            raise ValueError('parent insertion is too short for the treatment text')
        result[condition] = prefix + start + tokens + space * (body - len(tokens)) + end
    return result


def with_frame_ablation(tokenizer, prefix, inserted_tokens):
    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    start, end = encode(CONTEXT_OPEN), encode(CONTEXT_CLOSE)
    body = inserted_tokens - len(start) - len(end)
    tokens = {text: encode(text) for text in FRAME_FILLERS.values()}
    if body < 1 or any(len(ids) != 1 for ids in tokens.values()):
        raise ValueError('frame ablation requires room for one-token fillers')
    result = {}
    for name, arm in FRAME_ARMS.items():
        filler = tokens[arm['filler']]
        suffix = start + filler * body + end if arm['frame'] else filler * inserted_tokens
        result[name] = prefix + suffix
    return result


def newline_boundaries(opening, closing):
    return '\n\n' + opening + '\n', '\n' + closing + '\n'


def component_tags(*, words, brackets, edges):
    label = 'Additional context' if words else ''
    opening, closing = (f'[{label}]', f'[/{label}]') if brackets else (label, label)
    if edges:
        return newline_boundaries(opening, closing)
    return opening, closing


def punctuation_tags(*, opener, closer, slash):
    opening = '[]' if opener else ''
    mark = '/' if slash else ''
    closing = f'[{mark}]' if closer else mark
    return newline_boundaries(opening, closing)


def with_newline_tags(tokenizer, prefix, inserted_tokens, tags):
    newline = tokenizer.encode('\n', add_special_tokens=False)
    if len(newline) != 1:
        raise ValueError('tag ablation requires a single newline token')
    result = {}
    for arm, (opening, closing) in tags.items():
        start = tokenizer.encode(opening, add_special_tokens=False) if opening else []
        end = tokenizer.encode(closing, add_special_tokens=False) if closing else []
        body = inserted_tokens - len(start) - len(end)
        if body < 1:
            raise ValueError('component tags exceed the parent insertion budget')
        result[arm] = prefix + start + newline * body + end
    return result


def with_component_ablation(tokenizer, prefix, inserted_tokens):
    tags = {arm: component_tags(**flags) for arm, flags in COMPONENT_ARMS.items()}
    return with_newline_tags(tokenizer, prefix, inserted_tokens, tags)


def with_punctuation_ablation(tokenizer, prefix, inserted_tokens):
    tags = {arm: punctuation_tags(**flags) for arm, flags in PUNCTUATION_ARMS.items()}
    return with_newline_tags(tokenizer, prefix, inserted_tokens, tags)


def with_closing_marks_ablation(tokenizer, prefix, inserted_tokens):
    tags = {arm: newline_boundaries('', mark) for arm, mark in CLOSING_MARKS.items()}
    return with_newline_tags(tokenizer, prefix, inserted_tokens, tags)


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
    loader = transformers.AutoModelForImageTextToText if args.astra else transformers.AutoModelForCausalLM
    model = loader.from_pretrained(path, local_files_only=True, dtype=torch.bfloat16).to('cuda:0').eval()
    defaults_arg = ({'use_model_defaults': False}
                    if 'use_model_defaults' in inspect.signature(model.generate).parameters else {})
    class PresencePenalty(transformers.LogitsProcessor):
        def __init__(self, previous, width):
            self.previous = previous
            self.width = width

        def __call__(self, input_ids, scores):
            seen = torch.zeros_like(scores, dtype=torch.bool)
            for i, ids in enumerate(self.previous):
                seen[i, ids] = True
            seen.scatter_(1, input_ids[:, self.width:], True)
            return scores - 1.5 * seen
    tok = transformers.AutoTokenizer.from_pretrained(path, local_files_only=True)
    pad = tok.pad_token_id if tok.pad_token_id is not None else min(eos)
    mode = 'think' if args.astra else 'base'
    temperature, top_p, top_k = (.6, .95, 20) if args.astra else (.7, .9, 0)
    ablation = args.ablate_note is not None
    arms, attach = {
        'instructions': (NOTE_TEXT, with_matched_notes),
        'frame': (FRAME_ARMS, with_frame_ablation),
        'components': (COMPONENT_ARMS, with_component_ablation),
        'punctuation': (PUNCTUATION_ARMS, with_punctuation_ablation),
        'closing-marks': (CLOSING_MARKS, with_closing_marks_ablation),
    }[args.ablation_design]
    parent_cases = {}
    if ablation:
        source, source_sha, parent_deadline = prior_replay(args.ablate_note)
        parent_cases = {case['id']: case for case in source['verified_cases']}
        if (set(parent_cases) != {item['row']['id'] for item in selected}
                or len(parent_cases) != source['eligible_unique']
                or source['cap'] != args.cap or args.draw_start + args.draws > source['draws']
                or source['data_sha'] != sha(args.data) or Path(source['model']) != path or source['eos_ids'] != eos
                or args.deadline != parent_deadline):
            raise ValueError('ablation must match the parent replay cohort, model, data, cap, draws and deadline')
    items = []
    for item in selected:
        prompt = tok.encode(render(tok, item['row'], mode), add_special_tokens=False)
        prior = item['completion_prefix']
        prefix = prompt + prior
        seen = tok.decode(prior, skip_special_tokens=False)
        pending = args.astra and '</think>' not in seen
        had_answer = assess(item['row'], seen, prior, False, 8192 if args.astra else 2048, args.astra)['correct']
        details = {}
        if ablation:
            source_case = parent_cases[item['row']['id']]
            lengths = source_case['prefix_len']
            if (len(prefix) != lengths['plain'] or lengths['placebo'] != lengths['fact']
                    or item['phase'] != source_case['phase'] or item['period'] != source_case['period']
                    or had_answer != source_case['prior_final']):
                raise ValueError('prefix or insertion does not match the parent replay')
            inserted = lengths['placebo'] - lengths['plain']
            prefixes = attach(tok, prefix, inserted)
            if any(len(ids) != len(prefix) + inserted or ids[:len(prefix)] != prefix for ids in prefixes.values()):
                raise RuntimeError('ablation arms changed the original prefix or insertion length')
            details = {'inserted_tokens': inserted,
                       'original_prefix_sha256': hashlib.sha256(json.dumps(prefix).encode()).hexdigest()}
        else:
            prefixes = with_notes(tok, prefix, item['fact'], args.astra)
        items.append({**item, 'prefixes': prefixes, 'pending_thought': pending, 'prior_final': had_answer,
                      'ablation_details': details,
                      'fact_verbatim_in_original_prefix': item['fact'] in tok.decode(prefix)})
    contexts = tuple(arms) if ablation else ('plain', 'placebo', 'fact')
    header = {'kind': 'manifest', 'verified_unique': verified, 'eligible_unique': len(items),
              'model': str(path), 'source_sha': sha(Path(__file__)), 'data_sha': sha(args.data),
              'cap': args.cap, 'draws': args.draws, 'draw_start': args.draw_start,
              'seed_base': REPLAY_SEED, 'min_cases': args.min_cases, 'eos_ids': eos, 'mode': mode,
              'sampling': {'temperature': temperature, 'top_p': top_p, 'top_k': top_k,
                           'presence_penalty': 1.5 if args.astra else 0.},
              'verified_cases': [{'id': i['row']['id'], 'phase': i['phase'], 'period': i['period'],
                                  'prior_final': i['prior_final'], 'hint': i['fact'],
                                  'fact_verbatim_in_original_prefix': i['fact_verbatim_in_original_prefix'],
                                  **i['ablation_details'],
                                  'prefix_len': {k: len(v) for k, v in i['prefixes'].items()}} for i in items]}
    if ablation:
        header.update(note_ablation=arms, ablation_design=args.ablation_design, parent_replay_sha256=source_sha, runtime={
            'torch': str(torch.__version__), 'transformers': transformers.__version__, 'cuda': torch.version.cuda})
    start = time.monotonic()
    with args.output.open('x', buffering=1) as out, torch.inference_mode():
        print(json.dumps(header), file=out)
        print('start replay', len(items), 'of', verified, 'verified prompts', flush=True)
        for sampled in ((True,) if ablation else (False, True)):
            config = transformers.GenerationConfig(
                max_new_tokens=args.cap, do_sample=sampled, eos_token_id=eos, pad_token_id=pad,
                temperature=temperature if sampled else 1., top_p=top_p if sampled else 1.,
                top_k=top_k if sampled else 50, repetition_penalty=1., use_cache=True)
            draws = range(args.draw_start, args.draw_start + args.draws) if sampled else range(1)
            for draw in draws:
                for begin in range(0, len(items), args.batch):
                    group = items[begin:begin + args.batch]
                    for context in contexts:
                        if time.time() >= args.deadline - 120:
                            raise TimeoutError('approaching the original absolute deadline')
                        torch.manual_seed(REPLAY_SEED + 10000 * draw + begin)
                        parts = [i['prefixes'][context] for i in group]
                        width = max(map(len, parts))
                        ids = torch.tensor([[pad] * (width - len(p)) + p for p in parts], device='cuda')
                        mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in parts], device='cuda')
                        processor = None
                        if args.astra and sampled:
                            previous = [sorted(set(i['completion_prefix'])) for i in group]
                            processor = [PresencePenalty(previous, width)]
                        generated = model.generate(input_ids=ids, attention_mask=mask, generation_config=config,
                                                   logits_processor=processor, **defaults_arg)
                        for item, row in zip(group, generated[:, width:], strict=True):
                            tokens = row.tolist()
                            stop = next((i for i, token in enumerate(tokens) if token in eos), None)
                            tokens = tokens if stop is None else tokens[:stop]
                            text = tok.decode(tokens, skip_special_tokens=False)
                            score = assess(item['row'], text, tokens, stop is not None,
                                           args.cap, item['pending_thought'])
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


def ablation_contrasts(design):
    if design == 'instructions':
        return {f'{a}_minus_{b}': {a: 1, b: -1} for a, b in (
            ('continue', 'neutral'), ('finalize', 'neutral'), ('finalize', 'continue'))}

    if design == 'closing-marks':
        pairs = [('square_pair', control) for control in ('none', 'left_square', 'right_square', 'round_pair')]
        pairs += [('right_square', 'left_square'), ('left_square', 'none'),
                  ('right_square', 'none'), ('round_pair', 'none')]
        return {**{f'{a}_minus_{b}': {a: 1, b: -1} for a, b in pairs},
                'square_pair_interaction': {'square_pair': 1, 'left_square': -1, 'right_square': -1, 'none': 1}}

    if design in ('components', 'punctuation'):
        parts, design_arms = ((FRAME_COMPONENTS, COMPONENT_ARMS) if design == 'components'
                              else (PUNCTUATION_COMPONENTS, PUNCTUATION_ARMS))
        factorial = {}
        for order in range(1, len(parts) + 1):
            denominator = 2 ** (len(parts) - order)
            for terms in combinations(parts, order):
                factorial['_by_'.join(terms)] = {
                    arm: (-1) ** sum(not flags[t] for t in terms) / denominator
                    for arm, flags in design_arms.items()}
        full = '_'.join(parts)
        drops = {f'full_minus_no_{part}': {full: 1, '_'.join(n for n in parts if n != part): -1}
                 for part in parts}
        if design == 'punctuation':
            drops['slash_only_minus_none'] = {'slash': 1, 'none': -1}
        return {**factorial, **drops}

    frame_effects = {f'frame_effect_{f}': {f'framed_{f}': 1, f'bare_{f}': -1} for f in FRAME_FILLERS}
    return {
        **frame_effects,
        'frame_effect_average': {arm: (1 if spec['frame'] else -1) / len(FRAME_FILLERS)
                                 for arm, spec in FRAME_ARMS.items()},
        **{f'{f}_minus_periods_{frame}': {f'{frame}_{f}': 1, f'{frame}_periods': -1}
           for f in ('spaces', 'newlines') for frame in ('bare', 'framed')},
        'frame_by_spaces_vs_periods': {'framed_spaces': 1, 'bare_spaces': -1,
                                     'framed_periods': -1, 'bare_periods': 1},
    }


def summarize_ablation(paths):
    cells = defaultdict(dict)
    protocol = None
    draws = set()
    for path in paths:
        with path.open() as stream:
            head = json.loads(next(stream))
            fingerprint = {key: head[key] for key in (
                'parent_replay_sha256', 'note_ablation', 'verified_cases', 'sampling', 'model', 'cap',
                'seed_base', 'runtime', 'eos_ids', 'source_sha', 'data_sha')}
            fingerprint['ablation_design'] = head.get('ablation_design', 'instructions')
            if protocol is not None and protocol != fingerprint:
                raise ValueError('ablation outputs have different protocols')
            protocol = fingerprint
            shard_draws = set(range(head['draw_start'], head['draw_start'] + head['draws']))
            if draws & shard_draws:
                raise ValueError('ablation shards repeated a draw')
            draws.update(shard_draws)
            complete = False
            for line in stream:
                row = json.loads(line)
                if row['kind'] == 'complete':
                    complete = True
                    continue
                key = row['id'], row['draw']
                values = cells[row['context']]
                if row['decoder'] != 'sample' or row['draw'] not in shard_draws or key in values:
                    raise ValueError('unexpected or duplicate ablation rollout')
                values[key] = row
            if not complete:
                raise ValueError('ablation shard is missing its completion footer')

    cases_by_id = {c['id']: c for c in protocol['verified_cases']}
    expected = {(case, draw) for case in cases_by_id for draw in draws}
    if set(cells) != set(protocol['note_ablation']) or any(set(rows) != expected for rows in cells.values()):
        raise ValueError('incomplete ablation matrix')
    metrics = ('terminal_cycle', 'correct_final', 'eos', 'correct_and_no_terminal_cycle', 'new_correct_and_eos')
    by = {}
    results = {}
    for arm, rows in cells.items():
        outcomes = defaultdict(list)
        for (case, _), row in rows.items():
            loop = bool(row['terminal_cycle'])
            outcomes[case].append((loop, row['correct_final'], row['eos'], row['correct_final'] and not loop,
                                   row['good_termination'] and not cases_by_id[case]['prior_final']))
        by[arm] = {case: [statistics.fmean(flags) for flags in zip(*values, strict=True)]
                   for case, values in outcomes.items()}
        results[arm] = {name: sum(v[i] for values in outcomes.values() for v in values)
                        for i, name in enumerate(metrics)}
        results[arm]['prompts_new_correct_and_eos'] = sum(x[-1] > 0 for x in by[arm].values())

    rng = random.Random(11738)
    contrasts = {}
    for name, weights in ablation_contrasts(protocol['ablation_design']).items():
        comparison = {}
        for metric in ('terminal_cycle', 'new_correct_and_eos'):
            i = metrics.index(metric)
            paired = [sum(coef * by[arm][case][i] for arm, coef in weights.items()) for case in sorted(cases_by_id)]
            boot = sorted(statistics.fmean(rng.choices(paired, k=len(paired))) for _ in range(3000))
            comparison[metric] = {'delta_pp': round(100 * statistics.fmean(paired), 2),
                                  'ci95_pp': [round(100 * boot[x], 2) for x in (75, 2924)]}
        contrasts[name] = comparison
    return {'prompts': len(cases_by_id), 'draw_ids': sorted(draws), 'rollouts_per_arm': len(expected),
            'inserted_token_counts': sorted({item['inserted_tokens'] for item in cases_by_id.values()}),
            'cap': protocol['cap'], 'sampling': protocol['sampling'], 'runtime': protocol['runtime'],
            'ablation_design': protocol['ablation_design'], 'notes': protocol['note_ablation'],
            'arms': results, 'paired_prompt_bootstrap': contrasts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summarize', type=Path)
    parser.add_argument('--summarize-ablation', type=Path, nargs='+')
    parser.add_argument('--baseline-lane', type=Path)
    parser.add_argument('--astra', action='store_true', help='use pinned Qwen3.5 thinking-mode baseline shards')
    parser.add_argument('--data', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--deadline', type=int)
    parser.add_argument('--cap', type=int, default=512)
    parser.add_argument('--draws', type=int, default=8)
    parser.add_argument('--draw-start', type=int, default=0)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--min-cases', type=int, default=MINIMUM)
    parser.add_argument('--inventory', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--ablate-note', type=Path, help='match the cohort and inserted token count of a completed Astra replay directory')
    parser.add_argument('--ablation-design', choices=('instructions', 'frame', 'components', 'punctuation', 'closing-marks'),
                        default='instructions')
    args = parser.parse_args()
    if args.self_test:
        assert terminal_cycle(list(range(7)) * 20) == (7, 35)
        assert terminal_cycle(list(range(32)) * 4) is None
        assert terminal_cycle([99] + list(range(32)) * 5) == (32, 161)
        assert select_information('We found <<12*3=36>> and <<36-10=26>>. #### 26', '26') == '12*3 = 36'
        assert select_information('We found <<36-10=26>>. #### 26', '26') is None
        assert component_tags(words=True, brackets=True, edges=True) == (CONTEXT_OPEN, CONTEXT_CLOSE)
        assert punctuation_tags(opener=True, closer=True, slash=True) == component_tags(words=False, brackets=True, edges=True)
        for arm, bracketed in (('none', False), ('square_pair', True)):
            assert newline_boundaries('', CLOSING_MARKS[arm]) == punctuation_tags(
                opener=False, closer=bracketed, slash=False)
        print('exact-cycle and intermediate-information checks passed')
        return
    if args.summarize:
        print(json.dumps(summary(args.summarize)))
        return
    if args.summarize_ablation:
        print(json.dumps(summarize_ablation(args.summarize_ablation)))
        return
    if not args.baseline_lane or not args.data:
        parser.error('baseline lane and pinned dataset required')
    identity, selected, verified = cases(args.baseline_lane, args.data, args.astra)
    print(json.dumps({'exact_tail_verified': verified, 'valid_cue': len(selected), 'gate': args.min_cases}), flush=True)
    if args.inventory:
        return
    if not args.output or not args.deadline or min(args.cap, args.batch, args.draws, args.min_cases) < 1:
        parser.error('invalid output, deadline or sampling contract')
    if args.draw_start < 0 or (args.draw_start and not args.ablate_note):
        parser.error('nonzero draw offsets are for ablation shards')
    if args.ablate_note and not args.astra:
        parser.error('note ablation requires --astra')
    if args.ablation_design != 'instructions' and not args.ablate_note:
        parser.error('the selected ablation design requires --ablate-note')
    if len(selected) < args.min_cases:
        raise SystemExit('too few independently verified prompts with an intermediate cue')
    run(args, identity, selected, verified)


if __name__ == '__main__':
    main()
