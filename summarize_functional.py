"""Audit and summarize complete functional-edit receipts without loading models."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

from six_gpu_campaign import now, save, sha


def load(path):
    return json.loads(path.read_text())


def paired_scores(result):
    record = result['functional']
    summary = {
        'calibration_mse_before': record['fit_mse_before'],
        'calibration_mse_fit': record['fit_mse_actual'],
        'linearization_rmse': record['linearization_rmse'],
        'strict_unreachable': record['strict_zero_drift']['relative_unreachable'],
        'unprotected_unreachable': record['unprotected']['relative_unreachable'],
    }
    arms = result['results']
    if result['spec']['functional']['task'] == 'compression':
        summary['teacher_kl_fit_minus_ridge'] = (record['function_fit_heldout_teacher_kl'] -
                                                  record['ridge_heldout_teacher_kl'])
        for corpus, metric in (('wiki_test', 'nll'), ('lambada', 'last_word_accuracy'),
                               ('arc_easy', 'accuracy_norm')):
            left = arms['function_fit']['scores'][corpus][metric]
            right = arms['ridge']['scores'][corpus][metric]
            summary[corpus + '_fit_minus_ridge'] = left - right
    else:
        original, local, fit = (arms[name] for name in ('original', 'local', 'function_fit'))
        summary.update({
            'fit_target_shift': fit['target_shift'],
            'fit_clean_target_shift': fit['false_target_shift'],
            'fit_unseen_target_shift': fit['target_shift_unseen_token'],
            'fit_clean_kl': fit['preservation_kl'],
            'local_clean_kl': local['preservation_kl'],
            'local_target_shift': local['target_shift'],
            'fit_eos_probability': fit['eos_probability'],
            'original_recurrence': original['greedy_rollout']['recurrence_per_generated_token'],
            'local_recurrence': local['greedy_rollout']['recurrence_per_generated_token'],
            'fit_recurrence': fit['greedy_rollout']['recurrence_per_generated_token'],
            'oracle_recurrence': result['oracle_rollout']['recurrence_per_generated_token'],
            'fit_generated_length': statistics.mean(fit['greedy_rollout']['per_example_length']),
            'original_generated_length': statistics.mean(original['greedy_rollout']['per_example_length']),
        })
    return summary


def collect(roots, cache=None):
    if cache is None:
        cache = {}
    data = {'observed_at': now(), 'roots': {}}
    for root in roots:
        plan = load(root / 'plan.json')
        evidence = {'planned': len(plan['jobs']), 'deadline_utc': plan['deadline_utc'],
                    'workers': [], 'completed': [], 'failures': []}
        data['roots'][root.name] = evidence
        for lane in range(len(plan['devices'])):
            lane_root = root / f'lane{lane}'
            progress_path = lane_root / 'progress.json'
            if not progress_path.is_file():
                evidence['workers'].append({'lane': lane, 'status': 'unobserved'})
                continue
            progress = load(progress_path)
            evidence['workers'].append({k: progress[k] for k in ('lane', 'gpu', 'status')})
            if progress.get('error'):
                evidence['failures'].append({'lane': lane, 'error': progress['error']})
            for job in progress['jobs']:
                if job['status'] != 'completed':
                    continue
                path = lane_root / job['id'] / 'result.json'
                key = (path, job['sha256'])
                if key in cache:
                    evidence['completed'].append(cache[key])
                    continue
                if sha(path) != job['sha256']:
                    raise ValueError(f'job output differs from receipt: {job["id"]}')
                result = load(path)
                spec = result['spec']
                report = {
                    'id': job['id'], 'model': Path(spec['model']).name,
                    'program': spec['functional'], 'fraction': spec['remove_fraction'],
                    'scores': paired_scores(result),
                }
                cache[key] = report
                evidence['completed'].append(report)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots', nargs='+', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--compact', action='store_true')
    args = parser.parse_args()
    if (args.watch or args.detach) and args.output is None:
        parser.error('--watch and --detach require --output')
    if args.output and args.output.exists():
        raise SystemExit('fresh summary output required')
    deadline = max(load(root / 'plan.json')['deadline_epoch'] for root in args.roots)
    if args.detach:
        log_path = args.output.with_suffix('.log')
        with log_path.open('x') as log:
            child = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()),
                                      '--watch', '--output', str(args.output),
                                      *map(str, args.roots)], start_new_session=True,
                                     stdout=log, stderr=subprocess.STDOUT)
        print(json.dumps({'pid': child.pid, 'output': str(args.output)}))
        return
    cache = {}
    while True:
        evidence = collect(args.roots, cache)
        if args.output:
            save(args.output, evidence)
        elif args.compact:
            for name, report in evidence['roots'].items():
                completed = report['completed']
                print(json.dumps({'root': name, 'planned': report['planned'],
                                  'workers': report['workers'], 'failures': report['failures'],
                                  'completed': len(completed), 'examples': completed[:2]}))
        else:
            print(json.dumps(evidence))
        if not args.watch or time.time() >= deadline + 60 or all(
            w['status'] not in ('running', 'unobserved')
            for root in evidence['roots'].values() for w in root['workers']
        ):
            break
        time.sleep(120)


if __name__ == '__main__':
    main()
