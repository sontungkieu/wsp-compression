#!/usr/bin/env python3
"""Two independent, deadline-bounded GPU lanes with matched search budgets."""
import argparse
import datetime as datetime
import hashlib
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.temporary')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def device_status(gpu, expected_uuid):
    process = subprocess.run(['nvidia-smi', '-i', str(gpu),
                              '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, check=True, timeout=15)
    actual_uuid, free = [part.strip() for part in process.stdout.strip().split(',')]
    if actual_uuid != expected_uuid or int(free) < 70000:
        raise RuntimeError('assigned device identity or free memory changed')
    applications = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid',
                                    '--format=csv,noheader,nounits'],
                                   capture_output=True, text=True, check=True, timeout=15)
    if applications.stdout.strip():
        raise RuntimeError('assigned GPU already has a compute process')
    return {'uuid': actual_uuid, 'free_mib': int(free)}


def random_candidates(count=20):
    generator = random.Random(20261114)
    choices = []
    seen = set()
    while len(choices) < count:
        candidate = {'basis': generator.choice(['ridge', 'affine']),
                     'gain': generator.choice([.5, .75, 1., 1.25, 1.5]),
                     'mean_mix': generator.choice([0., .25, .5, .75, 1.]),
                     'profile': generator.choice(['flat', 'front', 'back', 'middle',
                                                   'protect_late', 'protect_early']),
                     'row_norm_cap': generator.choice([None, .05, .1, .2, .4])}
        key = json.dumps(candidate, sort_keys=True)
        if key not in seen:
            seen.add(key)
            choices.append(candidate)
    return choices


def run_lane(root, lane):
    plan = json.loads((root / 'plan.json').read_text())
    config = plan['config']
    settings = config['lanes'][lane]
    gpu = settings['gpu']
    lane_root = root / ('lane' + str(lane))
    source = Path(config['source'])
    progress = {'status': 'running', 'started_at': utc_now(), 'pid': os.getpid(),
                'lane': lane, 'gpu': gpu, 'deadline_utc': plan['deadline_utc'], 'jobs': []}
    active_process = None
    cancelled = False
    records = {}
    verified_models = set()

    def cancel_handler(_signal, _frame):
        nonlocal cancelled
        cancelled = True
        if active_process is not None and active_process.poll() is None:
            os.killpg(active_process.pid, signal.SIGTERM)

    signal.signal(signal.SIGTERM, cancel_handler)
    signal.signal(signal.SIGINT, cancel_handler)
    environment = {key: value for key, value in os.environ.items()
                   if key.lower() not in {'http_proxy', 'https_proxy', 'all_proxy', 'hf_token',
                                          'hugging_face_hub_token', 'wandb_api_key'}}
    environment.update({'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'CUDA_VISIBLE_DEVICES': str(gpu),
                        'OMP_NUM_THREADS': '4', 'HF_HUB_OFFLINE': '1',
                        'TRANSFORMERS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false',
                        'PYTHONDONTWRITEBYTECODE': '1'})

    def persist():
        write_json(lane_root / 'progress.json', progress)
        summary = {'status': progress['status'], 'lane': lane,
                   'completed': sum(job['status'] == 'completed' for job in progress['jobs']),
                   'failed': [job['id'] for job in progress['jobs'] if job['status'] == 'failed'],
                   'jobs': []}
        for label, result in records.items():
            summary['jobs'].append({'id': label, 'spec': result['spec'],
                                    'mask_sha256': result['mask_sha256'],
                                    'scores': {name: {corpus: {key: value for key, value in metrics.items()
                                                              if key != 'records'}
                                                     for corpus, metrics in arm['scores'].items()}
                                               for name, arm in result['results'].items()}})
        write_json(lane_root / 'summary.json', summary)

    def make_spec(**overrides):
        result = dict(config['base_spec'])
        result['gpu'] = gpu
        result.update(overrides)
        return result

    def run_job(label, stage, specification, timeout=1800):
        nonlocal active_process
        if cancelled:
            raise InterruptedError('user cancellation')
        remaining = plan['deadline_epoch'] - time.time()
        if remaining < 120:
            raise TimeoutError('eight-hour campaign budget exhausted')
        if file_hash(source) != plan['source_sha256']:
            raise RuntimeError('active experiment source changed')
        if file_hash(__file__) != plan['queue_sha256']:
            raise RuntimeError('active queue source changed')
        for path, expected in plan['data_sha256'].items():
            if file_hash(path) != expected:
                raise RuntimeError('input data changed during the campaign')
        model = specification['model']
        if model not in verified_models:
            for filename, expected in config['model_sha256'][model].items():
                if file_hash(Path(model) / filename) != expected:
                    raise RuntimeError('model snapshot hash mismatch')
            verified_models.add(model)
        device = device_status(gpu, settings['uuid'])
        remaining = plan['deadline_epoch'] - time.time()
        if remaining < 120:
            raise TimeoutError('eight-hour campaign budget exhausted before job admission')
        folder = lane_root / label
        folder.mkdir(exist_ok=False)
        write_json(folder / 'spec.json', specification)
        entry = {'id': label, 'stage': stage, 'status': 'running', 'started_at': utc_now(),
                 'output': str(folder / 'result.json'), 'log': str(folder / 'stdout.log'),
                 'device': device}
        progress['jobs'].append(entry)
        started = time.monotonic()
        with (folder / 'stdout.log').open('x') as log:
            active_process = subprocess.Popen([sys.executable, '-B', str(source), '--spec',
                                                str(folder / 'spec.json'), '--output', entry['output']],
                                               env=environment, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True)
            entry['pid'] = active_process.pid
            persist()
            try:
                code = active_process.wait(timeout=min(timeout, max(1, remaining - 20)))
            except subprocess.TimeoutExpired:
                os.killpg(active_process.pid, signal.SIGTERM)
                try:
                    active_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(active_process.pid, signal.SIGKILL)
                    active_process.wait()
                code = active_process.returncode
                entry['error'] = 'job timeout'
            active_process = None
        entry.update({'exit_code': code, 'seconds': time.monotonic() - started,
                      'finished_at': utc_now()})
        if code != 0 or not Path(entry['output']).is_file():
            entry['status'] = 'failed'
            persist()
            if cancelled:
                raise InterruptedError('user cancellation')
            failures = sum(job['status'] == 'failed' for job in progress['jobs'])
            if failures >= 3:
                raise RuntimeError('three failed jobs require diagnosis')
            return None
        result = json.loads(Path(entry['output']).read_text())
        if result['spec'] != specification or result['source_sha256'] != plan['source_sha256']:
            raise RuntimeError('result contract mismatch')
        entry['status'] = 'completed'
        entry['sha256'] = file_hash(entry['output'])
        records[label] = result
        persist()
        return result

    def validation(label, candidate):
        return run_job(label, 'validation_search', make_spec(candidate=candidate,
                       methods=['candidate'], wiki_validation_windows=64, wiki_test_windows=0,
                       lambada_n=0, arc_split='validation', arc_offset=0, arc_n=128))

    try:
        baseline = validation('baseline', {'basis': 'ridge', 'gain': 1., 'profile': 'flat',
                                          'mean_mix': 0., 'row_norm_cap': None})
        if baseline is None:
            raise RuntimeError('reference validation did not complete')
        reference_nll = baseline['results']['candidate']['scores']['wiki_validation']['nll']
        if lane == 0:
            proposals = json.loads((root / 'agent_round1.json').read_text())
            for index, proposal in enumerate(proposals):
                validation('search_' + str(index).zfill(2), proposal['candidate'])
            feedback = {'completed_at': utc_now(), 'jobs': [
                {'id': label, 'candidate': result['spec']['candidate'],
                 'scores': {name: {key: value for key, value in metrics.items() if key != 'records'}
                            for name, metrics in result['results']['candidate']['scores'].items()}}
                for label, result in records.items() if label.startswith('search_')]}
            write_json(lane_root / 'round1_feedback.json', feedback)
            # A fallback is pre-registered so the detached queue remains runnable
            # after browser access ends. The exact consumed proposal file is saved.
            round2_path = root / 'agent_round2.json'
            waiting_since = time.monotonic()
            while not round2_path.exists() and time.monotonic() - waiting_since < 180:
                if cancelled or time.time() >= plan['deadline_epoch'] - 120:
                    raise InterruptedError('campaign stopped while awaiting the next proposal batch')
                time.sleep(2)
            if not round2_path.exists():
                round2_path = root / 'agent_round2_fallback.json'
            round2 = json.loads(round2_path.read_text())
            write_json(lane_root / 'round2_consumed.json', {'at': utc_now(), 'file': round2_path.name,
                                                         'waiting_seconds': time.monotonic() - waiting_since,
                                                         'sha256': file_hash(round2_path), 'proposals': round2})
            for index, proposal in enumerate(round2, start=len(proposals)):
                validation('search_' + str(index).zfill(2), proposal['candidate'])
        else:
            for index, candidate in enumerate(random_candidates()):
                validation('search_' + str(index).zfill(2), candidate)
        search_results = [(label, result) for label, result in records.items() if label.startswith('search_')]
        feasible = []
        for label, result in search_results:
            scores = result['results']['candidate']['scores']
            nll = scores['wiki_validation']['nll']
            accuracy = scores['arc_easy']['accuracy_norm']
            if math.isfinite(nll) and nll <= reference_nll + .02:
                feasible.append((-accuracy, nll, label, result))
        if feasible:
            _, _, winner, winning_result = min(feasible, key=lambda item: item[:3])
        else:
            winner, winning_result = 'baseline', baseline
        candidate = winning_result['spec']['candidate']
        selection = {'frozen_at': utc_now(), 'winner': winner, 'candidate': candidate,
                     'reference_nll': reference_nll, 'search_completed': len(search_results),
                     'rule': 'maximize ARC validation length-normalized accuracy with Wiki NLL <= reference + 0.02'}
        write_json(lane_root / 'selection.json', selection)
        # All holdout scoring starts only after the winner has been frozen.
        for offset, count in ((512, 256), (768, 256), (4352, 256), (4608, 256), (4864, 256)):
            run_job(f'holdout_lambada_{offset}', 'holdout', make_spec(candidate=candidate,
                    lambada_offset=offset, lambada_n=count, arc_n=0))
        for offset in range(1664, 2376, 128):
            run_job(f'holdout_arc_{offset}', 'holdout', make_spec(candidate=candidate,
                    lambada_n=0, arc_offset=offset, arc_n=min(128, 2376-offset)))
        subprocess.run([sys.executable, '-B', config['analysis_source'], str(lane_root)],
                       env=environment, check=True, timeout=120, capture_output=True)
        # Fixed transfer and robustness grid. Each job evaluates all paired arms.
        for model_name, model_path in config['models'].items():
            for fraction in (.25, .5):
                run_job(f'transfer_{model_name}_{fraction}', 'transfer',
                        make_spec(candidate=candidate, model=model_path, remove_fraction=fraction,
                                  lambada_offset=4352, lambada_n=256, arc_offset=1664, arc_n=256))
        for model_name, model_path in config['models'].items():
            for fraction in (.25, .5):
                run_job(f'latency_{model_name}_{fraction}', 'latency',
                        make_spec(candidate=candidate, model=model_path, remove_fraction=fraction,
                                  benchmark=True, lambada_n=0, arc_n=0, wiki_test_windows=16))
        for fit_tokens in (256, 1024, 4096, 8192):
            for fit_offset in (0, 8192, 16384):
                for fraction in (.25, .5):
                    run_job(f'fit_{fit_tokens}_{fit_offset}_{fraction}', 'fit_robustness',
                            make_spec(candidate=candidate, fit_windows=fit_tokens//256,
                                      fit_offset=fit_offset, remove_fraction=fraction,
                                      lambada_offset=4352, lambada_n=256, arc_offset=1664, arc_n=128))
        for model_name, model_path in config['models'].items():
            for fraction in (.1, .25, .5, .75):
                for context in (256, 1024, 2048):
                    run_job(f'context_{model_name}_{fraction}_{context}', 'context',
                            make_spec(candidate=candidate, model=model_path, remove_fraction=fraction,
                                      context=context, wiki_test_windows=32768//context,
                                      lambada_offset=4352, lambada_n=256, arc_offset=1664, arc_n=128))
        # Longer, paired heldout estimates dominate the later budget.
        for model_name, model_path in config['models'].items():
            for fraction in (.25, .5):
                for fit_offset in (0, 4096, 8192):
                    run_job(f'extended_{model_name}_{fraction}_{fit_offset}', 'extended',
                            make_spec(candidate=candidate, model=model_path, remove_fraction=fraction,
                                      fit_offset=fit_offset, lambada_offset=4352, lambada_n=801,
                                      arc_offset=1664, arc_n=712), timeout=2400)
        # Spare budget measures transfer sensitivity to fitting regularization.
        # The previously selected candidate remains frozen across this grid.
        for model_name, model_path in config['models'].items():
            for fraction in (.25, .5):
                for regularizer in (.01, .1, 10.):
                    run_job(f'regularization_{model_name}_{fraction}_{regularizer}', 'regularization',
                            make_spec(candidate=candidate, model=model_path, remove_fraction=fraction,
                                      ridge_lambda=regularizer, lambada_offset=4352, lambada_n=801,
                                      arc_offset=1664, arc_n=712), timeout=2400)
        progress['status'] = 'completed_all_planned'
    except TimeoutError as error:
        progress.update({'status': 'budget_exhausted', 'error': str(error)})
    except InterruptedError as error:
        progress.update({'status': 'cancelled', 'error': str(error)})
    except Exception as error:
        progress.update({'status': 'failed', 'error': type(error).__name__ + ': ' + str(error)})
    progress['finished_at'] = utc_now()
    persist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--root', required=True)
    parser.add_argument('--mode', choices=['start', 'worker'], required=True)
    parser.add_argument('--lane', type=int)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if args.mode == 'worker':
        run_lane(root, args.lane)
        return
    config = json.loads(Path(args.config).read_text())
    if root.exists():
        raise SystemExit('fresh campaign root required')
    if len(config['lanes']) != 2 or len({lane['gpu'] for lane in config['lanes']}) != 2:
        raise SystemExit('two distinct GPU assignments required')
    for lane in config['lanes']:
        device_status(lane['gpu'], lane['uuid'])
    root.mkdir()
    deadline = time.time() + 8 * 3600
    plan = {'created_at': utc_now(), 'deadline_epoch': deadline,
            'deadline_utc': datetime.datetime.fromtimestamp(deadline, datetime.timezone.utc).isoformat(),
            'config': config, 'source_sha256': file_hash(config['source']),
            'queue_sha256': file_hash(__file__), 'data_sha256': {path: file_hash(path) for path in config['input_files']}}
    write_json(root / 'plan.json', plan)
    for filename in ('agent_round1.json', 'agent_round2_fallback.json'):
        write_json(root / filename, json.loads((Path(args.config).parent / filename).read_text()))
    children = []
    for lane in range(2):
        lane_root = root / ('lane' + str(lane))
        lane_root.mkdir()
        with (lane_root / 'supervisor.log').open('x') as log:
            child = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), '--root',
                                      str(root), '--mode', 'worker', '--lane', str(lane)],
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        children.append({'lane': lane, 'gpu': config['lanes'][lane]['gpu'], 'pid': child.pid})
    write_json(root / 'launch.json', {'started_at': utc_now(), 'children': children})
    print(json.dumps({'status': 'launched', 'root': str(root), 'children': children,
                      'deadline_utc': plan['deadline_utc']}))


if __name__ == '__main__':
    main()
