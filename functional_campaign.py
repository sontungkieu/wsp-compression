"""Launch the same research protocol with independent queues on three GPU hosts."""
import argparse
import datetime
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

import six_gpu_campaign as queue


def condition(task, model, *, start, directions=8, rows=16, weight=1., fraction=.5,
              shuffled=False, label=''):
    functional = {'task': task, 'start': start, 'directions': directions, 'rows': rows,
                  'length': 64, 'preserve_weight': weight}
    if task == 'repeat':
        functional.update(penalty=2., radius_scale=2., shuffled_basis=shuffled)
        return {'model': model, 'functional': functional}, f'repeat_{label}'
    return {'model': model, 'functional': functional, 'remove_fraction': fraction,
            'mask_windows': 8, 'fit_windows': 8,
            'methods': ['original', 'drop', 'ridge', 'candidate', 'function_fit'],
            'wiki_test_windows': 16, 'lambada_offset': 4352, 'lambada_n': 96,
            'arc_offset': 1664, 'arc_n': 96}, f'compression_{label}'


def build_jobs(role, models, candidate):
    result = []

    def append(task, model, **kwargs):
        outputs, label = condition(task, models[model], **kwargs)
        outputs['candidate'] = candidate
        result.append({'id': f'{len(result):03d}_{model}_{label}',
                       'group': task, 'overrides': outputs})

    if role == 'compression':
        for start in (24576, 36864, 49152):
            for directions, fraction, weight in (
                (8, .5, 1.), (8, .25, 1.), (8, .75, 1.),
                (4, .5, 1.), (16, .5, 1.), (8, .5, 0.), (8, .5, 10.),
            ):
                for model in ('q05', 'q15'):
                    append(role, model, start=start, directions=directions,
                           fraction=fraction, weight=weight,
                           label=f's{start}_k{directions}_p{fraction}_w{weight}')
    elif role == 'repeat':
        for start in (24576, 36864, 49152):
            for directions, rows, weight in (
                (8, 16, 1.), (4, 16, 1.), (16, 16, 1.),
                (8, 16, 0.), (8, 16, 10.), (8, 32, 1.), (16, 32, 1.),
            ):
                for model in ('q05', 'q15'):
                    append(role, model, start=start, directions=directions,
                           rows=rows, weight=weight,
                           label=f's{start}_k{directions}_n{rows}_w{weight}')
    elif role == 'controls':
        for start in (24576, 36864, 49152):
            for model in ('q05', 'q15'):
                for shuffled in (False, True):
                    append('repeat', model, start=start, shuffled=shuffled,
                           label=f's{start}_shuffle{int(shuffled)}')
                append('compression', model, start=start, label=f's{start}_replication')
    return result


def validate_inputs(parent, used_models):
    for path, expected in parent['data_sha256'].items():
        if queue.sha(path) != expected:
            raise SystemExit('input data hash mismatch: ' + path)
    config = parent['config']
    for model in used_models:
        for name, expected in config['model_sha256'][model].items():
            if queue.sha(Path(model) / name) != expected:
                raise SystemExit('model hash mismatch: ' + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', required=True, choices=('compression', 'repeat', 'controls'))
    parser.add_argument('--uuid', required=True, nargs='+')
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--parent-plan', required=True, type=Path)
    parser.add_argument('--deadline', required=True, type=int)
    args = parser.parse_args()
    if len(args.uuid) != (1 if args.role == 'controls' else 2):
        raise SystemExit('incorrect number of GPU UUIDs for this role')
    if args.root.exists():
        raise SystemExit('fresh output root required')
    if not 120 < args.deadline - time.time() <= 6 * 3600:
        raise SystemExit('deadline must fall within the six-hour allocation')

    parent = json.loads(args.parent_plan.read_text())
    config = parent['config']
    source_dir = Path(__file__).resolve().parent
    config['source'] = str(source_dir / 'recovery.py')
    config['base_spec'].update(wiki_validation_windows=0)
    jobs = build_jobs(args.role, config['models'], parent['selection']['candidate'])
    validate_inputs(parent, {job['overrides']['model'] for job in jobs})
    devices = [queue.device(gpu, uuid) for gpu, uuid in enumerate(args.uuid)]
    worker_path = Path(queue.__file__).resolve()
    functional = source_dir / 'wsp_functional.py'
    plan = {
        'role': args.role, 'host': socket.gethostname(), 'started_at': queue.now(),
        'parent_plan_sha256': queue.sha(args.parent_plan),
        'config': config, 'jobs': jobs, 'devices': devices,
        'data_sha256': parent['data_sha256'],
        'aux_sha256': {str(functional): queue.sha(functional)},
        'source_sha256': queue.sha(config['source']),
        'queue_sha256': queue.sha(worker_path),
        'launcher_sha256': queue.sha(__file__),
        'deadline_epoch': args.deadline,
        'deadline_utc': datetime.datetime.fromtimestamp(args.deadline, datetime.timezone.utc).isoformat(),
    }
    root = args.root.resolve()
    root.mkdir()
    queue.save(root / 'plan.json', plan)
    workers = []
    for index, gpu in enumerate(devices):
        lane = root / f'lane{index}'
        lane.mkdir()
        with (lane / 'supervisor.log').open('x') as log:
            child = subprocess.Popen([sys.executable, '-B', str(worker_path), '--root',
                                      str(root), '--worker', str(index)],
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        workers.append({'lane': index, 'gpu': gpu['gpu'], 'pid': child.pid})
    receipt = {'role': args.role, 'host': plan['host'], 'jobs': len(jobs),
               'children': workers, 'deadline_utc': plan['deadline_utc']}
    queue.save(root / 'launch.json', receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
