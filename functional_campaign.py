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


def build_followup(role, models, candidate):
    jobs = []

    def add(task, model, start, label, program, **kwargs):
        overrides, _ = condition(task, models[model], start=start, label=label, **kwargs)
        overrides['functional'].update(program)
        overrides['candidate'] = candidate
        jobs.append({'id': f'{len(jobs):03d}_{model}_{label}_s{start}', 'group': task,
                     'overrides': overrides})

    for start in (24576, 36864, 49152):
        if role == 'compression':
            settings = [(anchor, readout, 32, .5) for anchor in ('drop', 'ridge')
                        for readout in ('margins', 'fisher')]
            settings += [('ridge', 'fisher', 64, fraction) for fraction in (.25, .5, .75)]
            for anchor, readout, rows, fraction in settings:
                for model in ('q05', 'q15'):
                    add(role, model, start, f'{anchor}_{readout}_n{rows}_p{fraction}',
                        {'anchor': anchor, 'readout': readout}, rows=rows, fraction=fraction)
        elif role in ('repeat', 'controls'):
            settings = [(0, False, False), (64, False, False), (256, False, False),
                        (512, False, False), (256, True, False), (512, True, False)]
            if role == 'controls':
                settings = [(256, True, False), (256, True, True)]
            for basis_rows, paired, shuffled in settings:
                for model in ('q05', 'q15'):
                    add('repeat', model, start, f'crossfit_n{basis_rows}_pair{int(paired)}_shuf{int(shuffled)}',
                        {'basis_rows': basis_rows, 'basis_start': 62500, 'paired_basis': paired,
                         'shuffled_basis': shuffled, 'motif_transfer': True}, rows=32)
    return jobs


def wait_for_queue(root, deadline, uuids, role):
    plan_path = root / 'plan.json'
    while not plan_path.is_file():
        if time.time() >= deadline - 120:
            raise SystemExit('preceding queue never became ready within the allocation')
        time.sleep(15)
    parent = json.loads(plan_path.read_text())
    if parent['role'] != role or parent['host'] != socket.gethostname():
        raise SystemExit('preceding queue role or host differs')
    if parent['deadline_epoch'] != deadline or [d['uuid'] for d in parent['devices']] != uuids:
        raise SystemExit('preceding queue has different deadline or assigned devices')
    states = ('completed_all_planned', 'budget_exhausted', 'failed', 'cancelled')
    while time.time() < deadline - 120:
        paths = [root / f'lane{i}' / 'progress.json' for i in range(len(uuids))]
        if all(p.is_file() and json.loads(p.read_text())['status'] in states for p in paths):
            if any(json.loads(p.read_text())['status'] == 'cancelled' for p in paths):
                raise SystemExit('preceding queue was cancelled')
            return
        time.sleep(15)
    raise SystemExit('original budget exhausted while waiting')


def build_basis_study(role, models, candidate):
    if role != 'compression':
        raise ValueError('the basis study requires a compression queue')
    jobs = []
    for start in (24576, 36864, 49152):
        for fraction in (.25, .5, .75):
            for directions, parts in ((8, 1), (16, 1), (16, 2), (32, 4)):
                for model in ('q15', 'q7i'):
                    label = f's{start}_p{fraction}_k{directions}_part{parts}'
                    overrides, _ = condition(role, models[model], start=start, label=label,
                                             directions=directions, fraction=fraction, rows=64)
                    overrides['functional'].update(anchor='ridge', readout='fisher',
                                                   column_partitions=parts)
                    overrides['candidate'] = candidate
                    jobs.append({'id': f'{len(jobs):03d}_{model}_{label}', 'group': role,
                                 'overrides': overrides})
    return jobs


def build_trajectory_study(role, models, candidate):
    if role != 'controls':
        raise ValueError('the trajectory study uses the controls GPU')
    jobs = []
    for start in (24576, 36864, 49152):
        for source in (None, 'base', 'oracle'):
            for model in ('q05', 'q15'):
                label = f's{start}_source{source or "fixed"}'
                overrides, _ = condition('repeat', models[model], start=start, label=label, rows=32)
                overrides['functional'].update(
                    basis_rows=256, basis_start=62500, paired_basis=True, protect_copy=True,
                    motif_transfer=True, basis_trajectory=source, generation_steps=64,
                )
                overrides['candidate'] = candidate
                jobs.append({'id': f'{len(jobs):03d}_{model}_{label}', 'group': 'repeat',
                             'overrides': overrides})
    return jobs


def wait_for_natural(root, deadline, uuids):
    schedule = json.loads((root / 'schedule.json').read_text())
    if schedule['deadline'] != deadline or schedule['gpu'] != 0 or [schedule['uuid']] != uuids:
        raise SystemExit('natural trial has different deadline or assigned device')
    while time.time() < deadline - 120:
        status = root / 'status.json'
        if status.is_file() and json.loads(status.read_text())['status'] in ('completed', 'failed'):
            return
        time.sleep(15)
    raise SystemExit('natural trial did not release its device before the deadline')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', required=True, choices=('compression', 'repeat', 'controls'))
    parser.add_argument('--uuid', required=True, nargs='+')
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--parent-plan', required=True, type=Path)
    parser.add_argument('--deadline', required=True, type=int)
    parser.add_argument('--stage', type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument('--after', type=Path)
    parser.add_argument('--schedule', type=Path, help='fresh log file for waiting launcher')
    args = parser.parse_args()
    if len(args.uuid) != (1 if args.role == 'controls' else 2):
        raise SystemExit('incorrect number of GPU UUIDs for this role')
    if args.root.exists():
        raise SystemExit('fresh output root required')
    if not 120 < args.deadline - time.time() <= 6 * 3600:
        raise SystemExit('deadline must fall within the six-hour allocation')

    if args.schedule:
        if not args.after:
            parser.error('--schedule requires --after')
        with args.schedule.open('x') as log:
            child = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()),
                                      '--role', args.role, '--uuid', *args.uuid,
                                      '--root', str(args.root), '--parent-plan', str(args.parent_plan),
                                      '--deadline', str(args.deadline), '--stage', str(args.stage),
                                      '--after', str(args.after)], stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        print(json.dumps({'status': 'waiting', 'pid': child.pid, 'after': str(args.after),
                          'root': str(args.root)}))
        return
    if args.after:
        if args.stage == 4:
            wait_for_natural(args.after, args.deadline, args.uuid)
        else:
            wait_for_queue(args.after, args.deadline, args.uuid, args.role)

    parent = json.loads(args.parent_plan.read_text())
    config = parent['config']
    source_dir = Path(__file__).resolve().parent
    config['source'] = str(source_dir / 'recovery.py')
    config['base_spec'].update(wiki_validation_windows=0)
    builder = {1: build_jobs, 2: build_followup, 3: build_basis_study,
               4: build_trajectory_study}[args.stage]
    jobs = builder(args.role, config['models'], parent['selection']['candidate'])
    validate_inputs(parent, {job['overrides']['model'] for job in jobs})
    devices = [queue.device(gpu, uuid) for gpu, uuid in enumerate(args.uuid)]
    worker_path = Path(queue.__file__).resolve()
    functional = source_dir / 'wsp_functional.py'
    plan = {
        'role': args.role, 'stage': args.stage, 'host': socket.gethostname(), 'started_at': queue.now(),
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
