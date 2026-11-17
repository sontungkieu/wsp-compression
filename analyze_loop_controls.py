"""Analyze paired recurrence on the matched genuine and shuffled cells."""
import argparse
import json
from pathlib import Path
import random
import statistics

from six_gpu_campaign import sha


def collect(root):
    progress = json.loads((root / 'lane0' / 'progress.json').read_text())
    matches = {}
    for job in progress['jobs']:
        if job['status'] != 'completed':
            continue
        path = root / 'lane0' / job['id'] / 'result.json'
        if sha(path) != job['sha256']:
            raise ValueError('invalid result hash: ' + job['id'])
        result = json.loads(path.read_text())
        spec = result['spec']
        program = spec['functional']
        key = Path(spec['model']).name, program['start']
        pair = matches.setdefault(key, {})
        arm = 'shuffled' if program['shuffled_basis'] else 'genuine'
        if arm in pair:
            raise ValueError('duplicate control cell')
        pair[arm] = result['results']
    return matches


def recurrence(arm):
    g = arm['greedy_rollout']
    return list(zip(g['per_example_recurrence'], g['per_example_length'], strict=True))


def ratio(rows):
    return sum(r for r, _ in rows) / sum(n for _, n in rows)


def confidence(groups, seed=332, repetitions=4000):
    def difference(sample):
        return ratio([x for a, b in sample for x in a]) - ratio([x for a, b in sample for x in b])
    rng = random.Random(seed)
    bootstrap = []
    for _ in range(repetitions):
        sample = []
        for a, b in groups:
            indices = rng.choices(range(len(a)), k=len(a))
            sample.append(([a[i] for i in indices], [b[i] for i in indices]))
        bootstrap.append(difference(sample))
    bootstrap.sort()
    return {'difference': difference(groups), 'ci95_conditioned_on_offsets': [
        bootstrap[int(repetitions * .025)], bootstrap[int(repetitions * .975)]]}


def summarize(matches):
    models = {}
    for (model, offset), pair in sorted(matches.items()):
        if set(pair) != {'genuine', 'shuffled'}:
            continue
        a, b = pair['genuine'], pair['shuffled']
        if recurrence(a['original']) != recurrence(b['original']):
            raise ValueError('mismatched original rollouts')
        modes = {name: recurrence(arm) for name, arm in {
            'base': a['original'], 'local': a['local'], 'fit': a['function_fit'],
            'shuffled_fit': b['function_fit']}.items()}
        if len({len(rows) for rows in modes.values()}) != 1:
            raise ValueError('unpaired generation counts')
        models.setdefault(model, []).append((offset, modes, a, b))

    summaries = []
    for model, conditions in models.items():
        comparisons = {}
        for baseline in ('base', 'local', 'shuffled_fit'):
            comparisons[baseline] = confidence([(modes['fit'], modes[baseline])
                                                for _, modes, _, _ in conditions])
        summaries.append({
            'model': model, 'offsets': [x[0] for x in conditions],
            'prompts': sum(len(x[1]['base']) for x in conditions),
            'recurrence_by_offset': [
                {k: ratio(rows) for k, rows in modes.items()} for _, modes, _, _ in conditions],
            'fit_clean_kl_median': statistics.median(a['function_fit']['preservation_kl']
                                                    for _, _, a, _ in conditions),
            'fit_mean_generated_tokens_by_offset': [
                statistics.mean(n for _, n in modes['fit']) for _, modes, _, _ in conditions],
            'fit_minus': comparisons,
        })
    return summaries


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('controls_root', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(collect(args.controls_root))))
