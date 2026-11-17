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
            'fit_familiar_motif_shift': fit.get('familiar_motif_shift'),
            'fit_familiar_clean_shift': fit.get('familiar_clean_shift'),
            'fit_copy_target_shift': fit.get('copy_target_shift'),
            'fit_copy_kl': fit.get('copy_kl'),
            'basis_trigger_fraction': record.get('basis_trigger_fraction'),
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
        if not (root / 'plan.json').is_file():
            data['roots'][root.name] = {'planned': None, 'workers': [{'status': 'unobserved'}],
                                        'completed': [], 'failures': []}
            continue
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


def collect_natural(roots):
    result = {}
    for root in roots:
        record = {'status': 'unobserved', 'arm_scores': {}}
        status_path = root / 'status.json'
        if status_path.is_file():
            record.update(load(status_path))
        calibration_path = root / 'calibration.json'
        if calibration_path.is_file():
            calibration = load(calibration_path)
            record['loop_pairs'] = len(calibration['loop_pair_ids'])
            record['token_cap'] = calibration['token_cap']
        for arm in ('original', 'sparse_projection', 'function_fit'):
            path = root / (arm + '.json')
            if path.is_file():
                scores = load(path)['per_problem']
                fields = ('correct', 'ended_thinking', 'hit_cap', 'length', 'repeat_steps')
                record['arm_scores'][arm] = {'count': len(scores), **{
                    field: statistics.mean(row[field] for row in scores) for field in fields}}
        result[root.name] = record
    return result


def cohort_summary(completed):
    groups = {}
    for job in completed:
        program = job['program']
        if program['task'] == 'compression':
            cohort = ('compression', job['model'], program.get('anchor', 'drop'),
                      program.get('readout', 'margins'), program.get('column_partitions', 1))
        else:
            cohort = ('repeat', job['model'], program.get('basis_rows', 0),
                      bool(program.get('paired_basis')), bool(program.get('shuffled_basis')),
                      program.get('basis_trajectory') or 'fixed')
        groups.setdefault(cohort, []).append(job['scores'])
    reports = []
    for cohort, scores in sorted(groups.items()):
        keys = ('teacher_kl_fit_minus_ridge', 'wiki_test_fit_minus_ridge') if cohort[0] == 'compression' else (
            'fit_target_shift', 'fit_clean_target_shift', 'fit_unseen_target_shift',
            'fit_familiar_motif_shift', 'fit_clean_kl', 'original_recurrence', 'fit_recurrence',
            'oracle_recurrence', 'fit_copy_target_shift', 'fit_copy_kl', 'basis_trigger_fraction')
        medians = {}
        for key in ('strict_unreachable', 'unprotected_unreachable', 'linearization_rmse', *keys):
            values = [point[key] for point in scores if point.get(key) is not None]
            if values:
                medians[key] = statistics.median(values)
        report = {'cohort': cohort, 'conditions': len(scores), 'medians': medians}
        if cohort[0] == 'compression':
            report['teacher_kl_improvements'] = sum(point['teacher_kl_fit_minus_ridge'] < 0 for point in scores)
        else:
            report['recurrence_improvements'] = sum(point['fit_recurrence'] < point['original_recurrence']
                                                    for point in scores)
        reports.append(report)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots', nargs='+', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--compact', action='store_true')
    parser.add_argument('--deadline', type=int)
    parser.add_argument('--natural', type=Path, action='append', default=[])
    args = parser.parse_args()
    if (args.watch or args.detach) and args.output is None:
        parser.error('--watch and --detach require --output')
    if args.output and args.output.exists():
        raise SystemExit('fresh summary output required')
    deadline = args.deadline if args.deadline is not None else max(
        load(root / 'plan.json')['deadline_epoch'] for root in args.roots)
    if args.detach:
        log_path = args.output.with_suffix('.log')
        with log_path.open('x') as log:
            natural_args = [arg for path in args.natural for arg in ('--natural', str(path))]
            child = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()),
                                      '--watch', '--output', str(args.output), '--deadline', str(deadline),
                                      *natural_args, *map(str, args.roots)], start_new_session=True,
                                     stdout=log, stderr=subprocess.STDOUT)
        print(json.dumps({'pid': child.pid, 'output': str(args.output)}))
        return
    cache = {}
    while True:
        evidence = collect(args.roots, cache)
        evidence['natural'] = collect_natural(args.natural)
        if args.output:
            save(args.output, evidence)
        elif args.compact:
            for name, report in evidence['roots'].items():
                completed = report['completed']
                print(json.dumps({'root': name, 'planned': report['planned'],
                                  'workers': report['workers'], 'failures': report['failures'],
                                  'completed': len(completed), 'cohorts': cohort_summary(completed)}))
            if evidence['natural']:
                print(json.dumps({'natural': evidence['natural']}))
        else:
            print(json.dumps(evidence))
        planned_complete = all(
            w['status'] not in ('running', 'unobserved')
            for root in evidence['roots'].values() for w in root['workers']
        )
        natural_complete = all(row['status'] in ('completed', 'failed')
                               for row in evidence['natural'].values())
        if not args.watch or time.time() >= deadline + 60 or (planned_complete and natural_complete):
            break
        time.sleep(120)


if __name__ == '__main__':
    main()
