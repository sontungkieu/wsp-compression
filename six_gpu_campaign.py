#!/usr/bin/env python3
"""Bounded, independent GPU workers for paired pruning-recovery robustness jobs."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

OLD = '/workspace/storage-shared/nlp/tungks/'
NEW = '/workspace/storage-shared/nlp/users/tungks/'

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def save(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def moved(path):
    return NEW + path[len(OLD):] if path.startswith(OLD) else path

def device(index, expected=None):
    info = subprocess.run(['nvidia-smi', '-i', str(index), '--query-gpu=uuid,memory.free',
        '--format=csv,noheader,nounits'], check=True, text=True, capture_output=True, timeout=15).stdout.strip()
    uuid, free = [x.strip() for x in info.split(',')]
    if expected and uuid != expected:
        raise RuntimeError('GPU UUID changed')
    apps = subprocess.run(['nvidia-smi', '-i', str(index), '--query-compute-apps=pid',
        '--format=csv,noheader,nounits'], check=True, text=True, capture_output=True, timeout=15).stdout.strip()
    if apps or int(free) < 70000:
        raise RuntimeError('assigned GPU is busy or has insufficient free VRAM')
    return {'gpu': index, 'uuid': uuid, 'free_mib': int(free)}

def build_jobs(config, candidate):
    jobs = []
    def add(group, model, fraction, mask, fit, fit_windows=16, **extra):
        overrides = {'model': config['models'][model], 'remove_fraction': fraction,
            'mask_offset': mask, 'fit_offset': fit, 'fit_windows': fit_windows,
            'candidate': candidate, 'lambada_offset': 4352, 'lambada_n': 256,
            'arc_offset': 1664, 'arc_n': 128, 'wiki_test_windows': 32}
        overrides.update(extra)
        jobs.append({'id': f'{len(jobs):04d}_{group}_{model}_p{fraction}_m{mask}_f{fit}_w{fit_windows}',
                     'group': group, 'overrides': overrides})
    # Balanced initial wave across all models and pruning fractions.
    for mask in (0, 8192, 16384):
        for model in ('q05', 'q15', 'q7i'):
            for fraction in (.25, .5):
                add('mask', model, fraction, mask, 0)
    # Fitting corpus and sample size while keeping masks fixed.
    for fit_windows in (1, 4, 16, 32):
        for fit in (0, 8192, 16384):
            for model in ('q05', 'q15', 'q7i'):
                for fraction in (.25, .5):
                    if fit == 0 and fit_windows == 16:
                        continue
                    add('fit', model, fraction, 0, fit, fit_windows)
    # Examine paired intervention across contexts and harsher sparsity.
    for context in (256, 1024, 2048):
        for fraction in (.1, .25, .5, .75):
            for model in ('q05', 'q15', 'q7i'):
                add('context', model, fraction, 0, 0, context=context,
                    wiki_test_windows=32768 // context)
    return jobs

def worker(root, lane):
    plan = json.loads((root / 'plan.json').read_text())
    cfg = plan['config']; settings = plan['devices'][lane]; gpu = settings['gpu']
    folder = root / f'lane{lane}'; progress = {'lane': lane, 'gpu': gpu, 'status': 'running',
        'started_at': now(), 'deadline_utc': plan['deadline_utc'], 'jobs': []}
    active = None; cancelled = False
    def stop(_sig, _frame):
        nonlocal cancelled
        cancelled = True
        if active is not None and active.poll() is None:
            os.killpg(active.pid, signal.SIGTERM)
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    env = {k: v for k, v in os.environ.items() if k.lower() not in
           {'http_proxy','https_proxy','all_proxy','hf_token','hugging_face_hub_token','wandb_api_key'}}
    env.update({'CUDA_DEVICE_ORDER':'PCI_BUS_ID','CUDA_VISIBLE_DEVICES':str(gpu),
        'OMP_NUM_THREADS':'4','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
        'TOKENIZERS_PARALLELISM':'false','PYTHONDONTWRITEBYTECODE':'1'})
    try:
        for job in plan['jobs'][lane::6]:
            if cancelled: raise InterruptedError('cancelled')
            remaining = plan['deadline_epoch'] - time.time()
            if remaining < 120: raise TimeoutError('campaign deadline reached')
            if sha(__file__) != plan['queue_sha256'] or sha(cfg['source']) != plan['source_sha256']:
                raise RuntimeError('code hash changed')
            for path, expected in plan['data_sha256'].items():
                if sha(path) != expected: raise RuntimeError('data hash changed')
            d = device(gpu, settings['uuid'])
            spec = dict(cfg['base_spec']); spec.update(job['overrides']); spec['gpu'] = gpu
            jobroot = folder / job['id']; jobroot.mkdir()
            save(jobroot / 'spec.json', spec)
            entry = {'id': job['id'], 'group': job['group'], 'status': 'running', 'started_at': now(),
                     'device': d, 'output': str(jobroot/'result.json')}
            progress['jobs'].append(entry); save(folder/'progress.json', progress)
            with (jobroot/'stdout.log').open('x') as log:
                active = subprocess.Popen([sys.executable, '-B', cfg['source'], '--spec', str(jobroot/'spec.json'),
                    '--output', entry['output']], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                entry['pid'] = active.pid; save(folder/'progress.json', progress)
                try:
                    code = active.wait(timeout=min(2400, max(1, plan['deadline_epoch']-time.time()-20)))
                except subprocess.TimeoutExpired:
                    os.killpg(active.pid, signal.SIGTERM)
                    try: active.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(active.pid, signal.SIGKILL); active.wait()
                    code = active.returncode; entry['error'] = 'timeout'
                active = None
            entry.update({'exit_code': code, 'finished_at': now()})
            if code == 0 and Path(entry['output']).is_file():
                result = json.loads(Path(entry['output']).read_text())
                if result['spec'] != spec or result['source_sha256'] != plan['source_sha256']:
                    raise RuntimeError('result contract mismatch')
                entry['status'] = 'completed'; entry['sha256'] = sha(entry['output'])
            else:
                entry['status'] = 'failed'; save(folder/'progress.json', progress)
                raise RuntimeError('job failed; lane stopped for diagnosis')
            save(folder/'progress.json', progress)
        progress['status'] = 'completed_all_planned'
    except TimeoutError as e: progress.update(status='budget_exhausted', error=str(e))
    except InterruptedError as e: progress.update(status='cancelled', error=str(e))
    except Exception as e: progress.update(status='failed', error=type(e).__name__+': '+str(e))
    progress['finished_at'] = now(); save(folder/'progress.json', progress)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True); p.add_argument('--old-plan');
    p.add_argument('--selection'); p.add_argument('--worker',type=int); a=p.parse_args(); root=Path(a.root).resolve()
    if a.worker is not None:
        worker(root,a.worker); return
    if root.exists(): raise SystemExit('fresh output root required')
    old=json.loads(Path(a.old_plan).read_text()); cfg=old['config']
    cfg['source']=moved(cfg['source']); cfg['analysis_source']=moved(cfg['analysis_source'])
    cfg['models']={k:moved(v) for k,v in cfg['models'].items()}
    cfg['model_sha256']={moved(k):v for k,v in cfg['model_sha256'].items()}
    cfg['input_files']=[moved(x) for x in cfg['input_files']]
    for key in ('model','train','wiki_validation','wiki_test','data_dir'):
        cfg['base_spec'][key]=moved(cfg['base_spec'][key])
    if sha(cfg['source']) != old['source_sha256']: raise SystemExit('original source hash mismatch')
    data_hash={moved(k):v for k,v in old['data_sha256'].items()}
    for path, expected in data_hash.items():
        if sha(path)!=expected: raise SystemExit('input data hash mismatch: '+path)
    for model, files in cfg['model_sha256'].items():
        for name, expected in files.items():
            if sha(Path(model)/name)!=expected: raise SystemExit('model hash mismatch: '+name)
    selection=json.loads(Path(a.selection).read_text()); candidate=selection['candidate']
    devices=[device(i) for i in range(2,8)]
    deadline=time.time()+7*3600
    plan={'started_at':now(),'deadline_epoch':deadline,'deadline_utc':datetime.datetime.fromtimestamp(deadline,datetime.timezone.utc).isoformat(),
          'source_sha256':old['source_sha256'],'queue_sha256':sha(__file__),'data_sha256':data_hash,
          'original_plan_sha256':sha(a.old_plan),'selection_sha256':sha(a.selection),'selection':selection,
          'config':cfg,'devices':devices,'jobs':build_jobs(cfg,candidate)}
    root.mkdir(); save(root/'plan.json',plan); children=[]
    for lane in range(6):
        folder=root/f'lane{lane}'; folder.mkdir()
        with (folder/'supervisor.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--root',str(root),
                '--worker',str(lane)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children.append({'lane':lane,'gpu':devices[lane]['gpu'],'pid':child.pid})
    save(root/'launch.json',{'at':now(),'children':children,'deadline_utc':plan['deadline_utc']})
    print(json.dumps({'status':'launched','jobs':len(plan['jobs']),'children':children,'deadline_utc':plan['deadline_utc']}))

if __name__=='__main__': main()
