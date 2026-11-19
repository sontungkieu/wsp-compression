"""Run matched, original-weight panels in four independent GPU lanes."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from repetition_baseline import sha, summarize

FIRST = {
    4: ('Qwen2.5-1.5B', ('base',)),
    5: ('Qwen2.5-1.5B-Instruct', ('chat',)),
    6: ('Qwen3-0.6B', ('think', 'no-think')),
    7: ('Llama-3.2-3B-Instruct', ('chat',)),
}
SECOND = {
    4: ('Qwen3-0.6B-Base', ('base',)),
    5: ('Qwen2.5-3B-Instruct', ('chat',)),
    6: ('DeepSeek-R1-Distill-Qwen-1.5B', ('chat',)),
    7: ('Llama-3.2-3B-Base', ('base',)),
}
PILOT_SCRIPT = 'insight_campaign.py'


def save(path, value):
    pending = path.with_suffix('.tmp')
    pending.write_text(json.dumps(value, indent=2))
    pending.replace(path)


def smi(option):
    raw = subprocess.check_output(['nvidia-smi', option, '--format=csv,noheader,nounits'],
                                  text=True, timeout=10)
    return [tuple(value.strip() for value in row.split(',')) for row in raw.splitlines() if row.strip()]


def idle(uuid, index):
    inventory = smi('--query-gpu=index,uuid,memory.used,utilization.gpu')
    entry = next((row for row in inventory if row[0] == str(index)), None)
    if not entry or entry[1] != uuid:
        raise RuntimeError('GPU identity changed')
    apps = smi('--query-compute-apps=gpu_uuid,pid')
    return int(entry[2]) <= 64 and int(entry[3]) == 0 and all(row[0] != uuid for row in apps)


def wait_idle(uuid, index, until):
    while time.time() < until:
        if idle(uuid, index):
            time.sleep(2)
            if idle(uuid, index):
                return True
        time.sleep(5)
    return False


def retired_pilot(prior, index, pid, deadline):
    """Retire only the predecessor worker recorded for this exact own pilot lane."""
    status = prior / f'gpu{index}' / 'status.json'
    if time.time() >= deadline - 180:
        return False
    current = json.loads(status.read_text())
    if current['status'] != 'running':
        return True
    jobs = current['jobs']
    if not jobs or not any(j['kind'] == 'natural' and j['status'] == 'failed' for j in jobs):
        raise RuntimeError('expected earlier natural pilot failure was absent')
    proc = Path('/proc') / str(pid)
    try:
        if proc.stat().st_uid != os.getuid():
            raise RuntimeError('predecessor belongs to a different owner')
        argv = [x.decode() for x in (proc / 'cmdline').read_bytes().split(b'\0') if x]
    except FileNotFoundError:
        return True
    if (len(argv) != 7 or Path(argv[2]).name != PILOT_SCRIPT or
            argv[3:] != ['--root', str(prior), '--worker', str(index)]):
        raise RuntimeError('predecessor PID identity mismatch')
    if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
        raise RuntimeError('current runtime lacks pidfd; no predecessor signal sent')
    try:
        fd = os.pidfd_open(pid)
    except ProcessLookupError:
        return True
    try:
        try:
            if (proc / 'cmdline').read_bytes().split(b'\0')[:-1] != [x.encode() for x in argv]:
                raise RuntimeError('predecessor identity changed after pidfd open')
            signal.pidfd_send_signal(fd, signal.SIGTERM)
        except (FileNotFoundError, ProcessLookupError):
            return True
    finally:
        os.close(fd)
    return True


def run_lane(root, index):
    plan = json.loads((root / 'plan.json').read_text())
    if plan['host'] != socket.gethostname():
        raise RuntimeError('wrong host')
    deadline, uuid = plan['deadline'], plan['uuids'][str(index)]
    source = Path(__file__).with_name('repetition_baseline.py').resolve()
    if sha(source) != plan['source_sha']:
        raise RuntimeError('baseline source changed')
    folder = root / f'gpu{index}'
    folder.mkdir()
    status = folder / 'status.json'
    state = {'stage': 'waiting', 'uuid': uuid, 'jobs': []}
    save(status, state)
    if plan['retire_pilot']:
        retired_pilot(Path(plan['prior']), index, plan['prior_pids'][str(index)], deadline)
    if plan['after_baseline']:
        predecessor = Path(plan['after_baseline']) / f'gpu{index}' / 'status.json'
        while time.time() < deadline - 180:
            if predecessor.exists() and json.loads(predecessor.read_text())['stage'] not in ('running', 'waiting'):
                break
            time.sleep(15)
    if not wait_idle(uuid, index, deadline - 180):
        state['stage'] = 'deadline_waiting'
        save(status, state)
        return
    state['stage'] = 'running'
    env = {k: v for k, v in os.environ.items() if k.lower() not in
           {'http_proxy', 'https_proxy', 'all_proxy', 'hf_token', 'hugging_face_hub_token', 'wandb_api_key'}}
    env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=uuid,
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='2',
               TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1')
    first = FIRST[index]
    second = SECOND[index]
    # Ensure every lane measures both decoders before the longer checks on additional models.
    cases = [('canary', first, 4, 256, 0, 1), ('core', first, 32, 2048, 0, 1),
             ('cross', second, 32, 2048, 0, 1), ('replicate', first, 32, 2048, 1, 3),
             ('large_cap', first, 12, 8192, 0, 1), ('replicate_cross', second, 32, 2048, 1, 3)]
    for name, (model, modes), count, cap, shard, draws in cases:
        if not wait_idle(uuid, index, min(deadline - 180, time.time() + 50)):
            state['stage'] = 'deadline' if time.time() >= deadline - 180 else 'device_became_busy'
            break
        output = folder / (name + '.jsonl')
        cmd = [plan['worker_python'], '-B', str(source), '--model', str(Path(plan['models']) / model),
               '--data', plan['data'], '--output', str(output), '--deadline', str(deadline),
               '--count', str(count), '--cap', str(cap), '--shard', str(shard),
               '--sample-seeds', str(draws)]
        for mode in modes:
            cmd.extend(('--mode', mode))
        job = {'name': name, 'model': model, 'modes': modes, 'status': 'running',
               'cap': cap, 'started': time.time()}
        state['jobs'].append(job)
        with (folder / (name + '.log')).open('x') as log:
            child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
            job['pid'] = child.pid
            save(status, state)
            while child.poll() is None and time.time() < deadline - 15:
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if any(gpu == uuid and int(pid) != child.pid
                           for gpu, pid in smi('--query-compute-apps=gpu_uuid,pid')):
                        job['foreign_gpu_process'] = True
                        break
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=6)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        job.update(status='completed' if child.returncode == 0 else 'failed',
                   exit_code=child.returncode, finished=time.time())
        if output.is_file() and child.returncode == 0:
            job['sha256'] = sha(output)
            save(folder / (name + '.summary.json'), summarize(output))
        save(status, state)
        if job.get('foreign_gpu_process') or (name == 'canary' and child.returncode != 0):
            state['stage'] = 'device_became_busy' if job.get('foreign_gpu_process') else 'failed_canary'
            break
    else:
        state['stage'] = 'exhausted_grid'
    save(status, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prior', type=Path)
    parser.add_argument('--models', type=Path)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--host')
    parser.add_argument('--worker-python', type=Path, default=Path(sys.executable))
    parser.add_argument('--retire-pilot', action='store_true')
    parser.add_argument('--after-baseline', type=Path)
    parser.add_argument('--gpu', type=int, choices=tuple(FIRST), action='append')
    parser.add_argument('--lane', type=int, choices=tuple(FIRST))
    args = parser.parse_args()
    if args.lane is not None:
        try:
            run_lane(args.root, args.lane)
        except Exception as exc:
            status = args.root / f'gpu{args.lane}' / 'status.json'
            if status.exists():
                state = json.loads(status.read_text())
                save(status, {**state, 'stage': 'failed', 'error': str(exc)})
            raise
        return
    if not all((args.prior, args.models, args.data, args.host)):
        parser.error('prior allocation, model and dataset paths and host are required')
    prior = args.prior.resolve()
    launch = json.loads((prior / 'launch.json').read_text())
    earlier = json.loads((prior / 'plan.json').read_text())
    if (args.host != socket.gethostname() or launch['host'] != args.host or earlier['host'] != args.host or
            launch['deadline'] != earlier['deadline'] or not 180 < launch['deadline'] - time.time() <= 36000):
        raise RuntimeError('allocation identity or deadline mismatch')
    indexes = sorted(set(args.gpu or FIRST))
    uuids = {str(i): earlier['uuids'][i] for i in indexes}
    if args.after_baseline:
        before = json.loads((args.after_baseline / 'plan.json').read_text())
        if (args.retire_pilot or before['host'] != args.host or before['deadline'] != launch['deadline']
                or any(before['uuids'].get(i) != u for i, u in uuids.items())):
            raise RuntimeError('earlier baseline allocation does not match')
    inventory = {index: uuid for index, uuid in smi('--query-gpu=index,uuid')}
    if any(inventory.get(index) != uuid for index, uuid in uuids.items()):
        raise RuntimeError('GPU UUIDs differ from the original allocation')
    if not args.data.is_file() or not all((args.models / m / 'config.json').is_file()
                                        for m, _ in tuple(FIRST.values()) + tuple(SECOND.values())):
        raise RuntimeError('missing staged data or model configuration')
    if not args.worker_python.is_absolute() or not os.access(args.worker_python, os.X_OK):
        raise RuntimeError('worker Python must be an absolute, executable runtime path')
    root = args.root.resolve()
    root.mkdir()
    script = Path(__file__).resolve()
    save(root / 'plan.json', {'prior': str(prior), 'prior_pids': launch['worker_pids'], 'uuids': uuids,
                             'deadline': launch['deadline'], 'host': args.host, 'retire_pilot': args.retire_pilot,
                             'models': str(args.models.resolve()), 'data': str(args.data.resolve()),
                             'worker_python': str(args.worker_python),
                             'after_baseline': str(args.after_baseline.resolve()) if args.after_baseline else None,
                             'first': FIRST, 'second': SECOND, 'launcher_sha': sha(script),
                             'source_sha': sha(script.with_name('repetition_baseline.py'))})
    pids = {}
    for index in indexes:
        with (root / f'gpu{index}.log').open('x') as log:
            proc = subprocess.Popen([sys.executable, '-B', str(script), '--root', str(root), '--lane', str(index)],
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        pids[index] = proc.pid
    receipt = {'pids': pids, 'host': args.host, 'deadline': launch['deadline'], 'root': str(root)}
    save(root / 'launch.json', receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
