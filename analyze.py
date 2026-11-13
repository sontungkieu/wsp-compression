#!/usr/bin/env python3
"""Paired document bootstrap for the reserved holdout shards of one lane."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('lane_root')
args = parser.parse_args()
root = Path(args.lane_root)
progress = json.loads((root / 'progress.json').read_text())
records = {}
for job in progress['jobs']:
    if job['status'] != 'completed' or job['stage'] != 'holdout':
        continue
    raw = Path(job['output']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != job['sha256']:
        raise RuntimeError('receipt hash mismatch')
    result = json.loads(raw)
    for arm, value in result['results'].items():
        for corpus, metric in value['scores'].items():
            if corpus == 'wiki_test':
                continue
            rows = records.setdefault((corpus, arm), {})
            for row in metric['records']:
                if row['id'] in rows and rows[row['id']] != row:
                    raise RuntimeError('conflicting scores for a repeated test document')
                rows[row['id']] = row

output = {'completed_at': progress.get('finished_at'), 'replicates': 2000, 'seed': 42,
          'protocol': 'paired bootstrap over documents; candidate minus reference',
          'comparisons': []}
generator = np.random.default_rng(42)
for corpus in ('lambada', 'arc_easy'):
    candidate_rows = records.get((corpus, 'candidate'), {})
    if not candidate_rows:
        continue
    identifiers = sorted(candidate_rows)
    candidate = [candidate_rows[key] for key in identifiers]
    n = len(identifiers)
    samples = generator.integers(0, n, size=(2000, n))
    for reference in ('drop', 'mean', 'ridge', 'affine'):
        reference_rows = records.get((corpus, reference), {})
        if set(reference_rows) != set(identifiers):
            raise RuntimeError('paired example IDs differ')
        baseline = [reference_rows[key] for key in identifiers]
        metrics = ('accuracy_raw', 'accuracy_norm') if corpus == 'arc_easy' else ('nll', 'last_word_accuracy')
        for metric in metrics:
            if metric == 'nll':
                left = np.array([row['sum_nll'] for row in candidate])
                right = np.array([row['sum_nll'] for row in baseline])
                counts = np.array([row['tokens'] for row in candidate])
                difference = (left.sum() - right.sum()) / counts.sum()
                distribution = (left[samples] - right[samples]).sum(1) / counts[samples].sum(1)
            else:
                field = {'accuracy_raw': 'correct_raw', 'accuracy_norm': 'correct_norm',
                         'last_word_accuracy': 'last_word_correct'}[metric]
                delta = np.array([float(row[field]) - float(other[field])
                                  for row, other in zip(candidate, baseline)])
                difference = delta.mean()
                distribution = delta[samples].mean(1)
            lower, upper = np.quantile(distribution, [.025, .975])
            output['comparisons'].append({'corpus': corpus, 'metric': metric, 'reference': reference,
                                           'documents': n, 'difference': float(difference),
                                           'ci95': [float(lower), float(upper)]})
temporary = root / 'paired_analysis.temporary'
temporary.write_text(json.dumps(output, indent=2) + '\n')
temporary.replace(root / 'paired_analysis.json')
print(json.dumps(output))
