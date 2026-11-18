"""Run independent, deadline-bounded loop-gradient queues on assigned GPUs."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


def save(path, value):
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2)
    tmp.replace(path)


def digest(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def smi(fields, apps=False):
    option = '--query-compute-apps=' if apps else '--query-gpu='
    output = subprocess.run(['nvidia-smi', option + fields, '--format=csv,noheader,nounits'],
                            check=True, capture_output=True, text=True, timeout=15).stdout
    return [tuple(x.strip() for x in row.split(',')) for row in output.splitlines() if row.strip()]


def idle_devices(assignments):
    busy = {row[0] for row in smi('gpu_uuid,pid', apps=True)}
    cards = {int(i): (u, int(mem), int(util)) for i, u, mem, util
             in smi('index,uuid,memory.used,utilization.gpu')}
    return {index for index, uuid, _ in assignments if cards.get(index, ('', 0, 0))[0] == uuid
            and cards[index][1] <= 32 and cards[index][2] == 0 and uuid not in busy}


def ready(predecessor):
    if not predecessor:
        return True
    path = Path(predecessor)
    return path.is_file() and json.loads(path.read_text()).get('status') in {
        'completed', 'completed_all_planned', 'failed', 'budget_exhausted'}


def foreign_process(uuid, expected_group):
    for gpu, pid in smi('gpu_uuid,pid', apps=True):
        if gpu == uuid:
            try:
                if os.getpgid(int(pid)) != expected_group:
                    return True
            except ProcessLookupError:
                continue
            except PermissionError:
                return True
    return False


def await_idle(assignment, deadline):
    stop = min(deadline - 35, time.time() + 60)
    while time.time() < stop:
        if assignment[0] in idle_devices([assignment]):
            return True
        time.sleep(3)
    return False


def jobs(slot, offload):
    models = ['q05', 'q15', 'q7i']
    if slot % 3 == 1:
        models = models[1:] + models[:1]
    elif slot % 3 == 2:
        models = models[2:] + models[:2]
    yield dict(model=models[0] if models[0] != 'q7i' else 'q05', period=8, horizon=16,
               objective='aligned', rows=8, test_rows=12, draws=3, columns=32, strength=.5)
    recipes = [(8, 32, 'aligned'), (8, 32, 'recurrence'), (4, 16, 'aligned'),
               (12, 40, 'recurrence'), (12, 24, 'aligned'), (4, 24, 'recurrence')]
    recipes = recipes[slot % len(recipes):] + recipes[:slot % len(recipes)]
    for round_no, (_, _, (rows, cols, strength, draws)) in enumerate(itertools.product(
            models, recipes, [(24, 32, .5, 4), (40, 128, .35, 6), (48, 128, .7, 8),
                              (64, 512, .5, 8), (64, 512, .75, 12), (80, 128, .5, 12)])):
        # Mix models and motifs among early jobs instead of exhausting one grid axis first.
        period, horizon, objective = recipes[(round_no // 3 + round_no % 3) % len(recipes)]
        model = models[round_no % len(models)]
        yield dict(model=model, period=period, horizon=horizon, objective=objective,
                   rows=rows, test_rows=32 if offload else 40, draws=draws,
                   columns=cols, strength=strength)


def deletion_jobs(slot):
    """Run the gate comparison first, then repeat it in Astra's matrix family."""
    models = ('q05', 'q15') if slot % 2 == 0 else ('q15', 'q05')
    yield dict(model=models[0], period=8, columns=8, rows=8, test_rows=12,
               steps=6, random_draws=2, eval_draws=1, mask_module='gate_proj')
    for module, period, columns, model in itertools.product(
            ('gate_proj', 'down_proj'), (8, 4, 12), (16, 64), models):
        yield dict(model=model, period=period, columns=columns, rows=64, test_rows=40,
                   steps=48, mask_module=module)


DELETION_SUPPORT = ('loop_gradient.py', 'six_gpu_campaign.py', 'wsp_functional.py')


def worker(args, assignment):
    index, uuid, _ = assignment
    root = args.root / f'gpu{index}'
    root.mkdir()
    state = {'host': args.host, 'gpu': index, 'uuid': uuid, 'pid': os.getpid(),
             'status': 'running', 'deadline': args.until, 'jobs': []}
    progress = root / 'status.json'
    save(progress, state)
    parent = json.loads(args.parent.read_text())
    config = parent['config']
    data = {Path(p).name: p for p in config['input_files'] if Path(p).name in {'train.txt', 'test.txt'}}
    for path in data.values():
        if digest(path) != parent['data_sha256'][path]:
            raise RuntimeError('data hash differs from parent')
    deletion = args.experiment == 'deletion'
    payload = Path(__file__).with_name('loop_deletion.py' if deletion else 'loop_gradient.py')
    sources = {payload: args.payload_sha,
               **{payload.with_name(name): expected for name, expected in args.support_sha.items()}}
    verified_models, failures = set(), {}
    env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(index),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
               PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4')
    active = None

    def stop_child(signum=signal.SIGTERM):
        if active is not None and active.poll() is None:
            os.killpg(active.pid, signum)

    def on_stop(_signal, _frame):
        stop_child()
        raise InterruptedError('worker stopped')

    signal.signal(signal.SIGTERM, on_stop)
    signal.signal(signal.SIGINT, on_stop)
    try:
        batch = deletion_jobs(args.slot + index) if deletion else jobs(args.slot + index, args.offload)
        for number, job in enumerate(batch):
            if time.time() >= args.until - 120:
                state['status'] = 'deadline'
                break
            if not await_idle(assignment, args.until):
                state['status'] = 'device_became_busy'
                break
            for path, expected in sources.items():
                if digest(path) != expected:
                    raise RuntimeError(f'source hash changed: {path.name}')
            model = config['models'][job['model']]
            if failures.get(model, 0) >= 2:
                continue
            if model not in verified_models:
                for name, expected in config['model_sha256'][model].items():
                    if digest(Path(model) / name) != expected:
                        raise RuntimeError('model hash differs from parent')
                verified_models.add(model)
            if not await_idle(assignment, args.until):
                state['status'] = 'device_became_busy'
                break
            label = job['mask_module'] if deletion else job['objective']
            folder = root / f'{number:03d}-{job["model"]}-p{job["period"]}-{label}'
            folder.mkdir()
            output = folder / 'result.json'
            # The two 12-condition deletion panels share calibration and holdout draws.
            case = (number - 1) % 12 + 1 if deletion and number else number
            settings = {**job, 'model': model, 'train': data['train.txt'], 'test': data['test.txt'],
                        'seed': 13201 + args.slot * 10000 + index * 1200 + case,
                        'offset': 4096 + ((args.slot * 3071 + index * 8191 + case * 5381) % 80000),
                        'test_offset': (args.slot * 313 + index * 2048 + case * 4201) % 36000,
                        'eval_horizon': 48, 'output': str(output),
                        'batch': 4 if job['model'] == 'q7i' else 8}
            if not deletion:
                settings['grad_batch'] = 2 if job['model'] == 'q7i' else 4
            argv = [sys.executable, '-B', str(payload)]
            for k, v in settings.items():
                argv += ['--' + k.replace('_', '-'), str(v)]
            entry = {'folder': str(folder), 'settings': settings, 'status': 'running',
                     'started': time.time(), 'canary': deletion and number == 0}
            state['jobs'].append(entry)
            with (folder / 'stdout.log').open('x') as log:
                active = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT,
                                          start_new_session=True)
                entry['payload_pid'] = active.pid
                save(progress, state)
                job_until = min(time.time() + 1400, args.until - 20)
                while active.poll() is None and time.time() < job_until:
                    if args.offload and foreign_process(uuid, active.pid):
                        entry['yielded_to_occupancy'] = True
                        break
                    try:
                        active.wait(timeout=max(1, min(15 if args.offload else 1400, job_until - time.time())))
                    except subprocess.TimeoutExpired:
                        continue
                if active.poll() is None:
                    stop_child()
                    try:
                        code = active.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        stop_child(signal.SIGKILL)
                        code = active.wait()
                    entry['timed_out'] = 'yielded_to_occupancy' not in entry
                else:
                    code = active.returncode
                active = None
            entry.update(exit_code=code, finished=time.time())
            if code == 0 and output.is_file():
                result = json.loads(output.read_text())
                eligible = result['eligible'] if deletion else result['centered_risk_sd'] > 0.
                entry.update(status='completed', sha256=digest(output), eligible=eligible)
            else:
                entry['status'] = 'failed'
                failures[model] = failures.get(model, 0) + 1
            save(progress, state)
            if entry.get('yielded_to_occupancy'):
                state['status'] = 'device_became_busy'
                break
        else:
            state['status'] = 'exhausted_grid'
    except (InterruptedError, KeyboardInterrupt):
        state['status'] = 'stopped'
        if active is not None and active.poll() is None:
            try:
                active.wait(timeout=8)
            except subprocess.TimeoutExpired:
                stop_child(signal.SIGKILL)
                active.wait()
    except Exception as exc:
        state.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
        stop_child()
        raise
    finally:
        save(progress, state)


def watch(args):
    plan = {'host': args.host, 'pid': os.getpid(), 'available_from': args.start, 'until': args.until,
            'assignments': args.assign, 'offload': args.offload, 'experiment': args.experiment,
            'payload_sha256': args.payload_sha, 'support_sha256': args.support_sha,
            'launcher_sha256': digest(__file__), 'parent_sha256': digest(args.parent),
            'poll_seconds': 1800 if args.offload else 20, 'launched': {}, 'polls': []}
    progress = args.root / 'watcher.json'
    save(progress, plan)
    time.sleep(max(0., args.start - time.time()))
    while time.time() < args.until - 180:
        tick = time.time()
        eligible = [a for a in args.assign if str(a[0]) not in plan['launched'] and ready(a[2])]
        snapshot = {'at': tick, 'eligible': [a[0] for a in eligible], 'idle': []}
        try:
            idle = idle_devices(eligible)
            if args.offload and idle:
                time.sleep(15)
                idle &= idle_devices(eligible)
            snapshot['idle'] = sorted(idle)
            for assignment in eligible:
                index = assignment[0]
                if index not in idle:
                    continue
                until = min(args.until, int(time.time()) + 6900) if args.offload else args.until
                argv = [sys.executable, '-B', str(Path(__file__).resolve()), *sys.argv[1:],
                        '--worker', str(index), '--worker-until', str(until)]
                with (args.root / f'gpu{index}.log').open('x') as log:
                    child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                plan['launched'][str(index)] = {'pid': child.pid, 'until': until, 'started': time.time()}
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            snapshot['error'] = type(exc).__name__ + ': ' + str(exc)
        plan['polls'].append(snapshot)
        plan['polls'] = plan['polls'][-32:]
        save(progress, plan)
        if len(plan['launched']) == len(args.assign):
            plan['status'] = 'all_admitted'
            break
        time.sleep(max(0., min(tick + plan['poll_seconds'], args.until) - time.time()))
    else:
        plan['status'] = 'window_closed'
    save(progress, plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--payload-sha', required=True)
    parser.add_argument('--experiment', choices=('gradient', 'deletion'), default='gradient')
    parser.add_argument('--support-sha', action='append', default=[], metavar='FILE=SHA256')
    parser.add_argument('--assign', action='append', required=True, help='index,uuid,predecessor status path')
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--until', type=int, required=True)
    parser.add_argument('--slot', type=int, required=True)
    parser.add_argument('--offload', action='store_true')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--worker', type=int)
    parser.add_argument('--worker-until', type=int)
    args = parser.parse_args()
    if any('=' not in entry for entry in args.support_sha):
        parser.error('--support-sha expects FILE=SHA256')
    supports = dict(entry.split('=', 1) for entry in args.support_sha)
    required = set(DELETION_SUPPORT if args.experiment == 'deletion' else ())
    if set(supports) != required or len(supports) != len(args.support_sha):
        parser.error(f'expected support hashes for {sorted(required)}')
    args.support_sha = supports
    args.assign = [(int(i), u, p) for i, u, p in (a.split(',', 2) for a in args.assign)]
    if socket.gethostname() != args.host or len({x[0] for x in args.assign}) != len(args.assign):
        parser.error('unexpected host or duplicate GPU assignment')
    if args.start >= args.until or (not args.watch and args.worker is None and args.until <= time.time() + 180):
        parser.error('a live allocation with at least three minutes remaining is required')
    if args.offload and args.until - args.start > 14400:
        parser.error('offload window exceeds four hours')
    if args.worker is not None:
        args.until = min(args.until, args.worker_until)
        worker(args, next(a for a in args.assign if a[0] == args.worker))
    elif args.watch:
        watch(args)
    else:
        if args.experiment == 'deletion':
            payload = Path(__file__).with_name('loop_deletion.py')
            expected_sources = {payload: args.payload_sha,
                                **{payload.with_name(name): expected for name, expected in supports.items()}}
            if any(not path.is_file() or digest(path) != expected for path, expected in expected_sources.items()):
                parser.error('deletion source snapshot is incomplete or differs from the pinned hashes')
        args.root.mkdir()
        with (args.root / 'watcher.log').open('x') as log:
            child = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), *sys.argv[1:], '--watch'],
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'pid': child.pid, 'root': str(args.root), 'first_poll': args.start, 'until': args.until}))


if __name__ == '__main__':
    main()
