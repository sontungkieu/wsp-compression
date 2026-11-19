"""A bounded eight-GPU queue for copy retention and natural loop repair."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import time

from six_gpu_campaign import device, now, save, sha


SOURCES = ('loop_deletion.py', 'loop_gradient.py', 'natural_loop.py',
           'six_gpu_campaign.py', 'wsp_functional.py')


def copy_cases(replica, model, phase):
    for module, width, copy_weight, loss in itertools.product(
            ('gate_proj', 'down_proj'), (16, 64), (0, 1, 4, 16), ('kl', 'target')):
        if copy_weight == 0 and loss == 'target':
            continue
        period = (8, 4, 12)[replica % 3]
        yield ('copy', dict(model=model, period=period, columns=width, mask_module=module,
                            copy_weight=copy_weight, copy_loss=loss, copy_probe_tokens=min(period, 4),
                            rows=64, test_rows=40, steps=64, eval_draws=1,
                            offset=4096 + (replica * 9749 + phase * 7919) % 53000,
                            test_offset=1024 + (replica * 5999 + phase * 4327) % 28500,
                            seed=9171 + replica * 500 + phase, magnitude_control=True))


def transfer_cases(lane, phase):
    modules = ('gate_proj', 'down_proj') if lane % 2 == 0 else ('down_proj', 'gate_proj')
    models = ('q05', 'q15', 'q7i') if phase else ('q05', 'q15')
    for model, module, test, copies, copy_weight in itertools.product(
            models, modules, (8, 4, 12, 16), (3, 2), (0, 4)):
        yield ('transfer', dict(model=model, period=8, test_period=test, test_repetitions=copies,
                                columns=16, mask_module=module, copy_weight=copy_weight,
                                rows=48, test_rows=40, steps=48, eval_draws=1,
                                offset=12288 + phase * 4951,
                                test_offset=2048 + phase * 3049, seed=17021 + phase * 331,
                                magnitude_control=True))


def natural_case(lane, block):
    panel = (lane // 2) % 2
    settings = [('sample', 1.5, 4096, 24), ('sample', 1.5, 8192, 24),
                ('greedy', 1.5, 4096, 24), ('sample', 0., 8192, 24),
                ('greedy', 0., 8192, 24), ('sample', 1.5, 16384, 20)]
    decoder, presence, cap, count = settings[block % len(settings)]
    return ('natural', dict(detector='recurrent', conclusion_beta=0.5 if lane % 2 else 0.,
                            decoder=decoder, presence=presence, tokens=cap, batch=12,
                            train=24, test=count, seed=251 + block * 19 + panel * 177,
                            train_offset=768 + panel * 1792 + block * 36,
                            test_offset=256 + panel * 448 + block * 24))


def jobs(lane):
    if lane < 4:
        yield ('canary', dict(model='q05' if lane % 2 == 0 else 'q15', period=8,
                              columns=8, rows=8, test_rows=12, steps=6, eval_draws=1,
                              random_draws=2, copy_weight=4, copy_loss='kl' if lane < 2 else 'target',
                              magnitude_control=True, seed=91001 + lane))
    if lane >= 4:
        yield natural_case(lane, 0)
        yield natural_case(lane, 1)
        for block in range(2, 16):
            # Interleave a short mask control after each natural panel.
            yield ('bridge', dict(model='q15' if lane % 2 else 'q05', period=(4, 8, 12)[block % 3],
                                  mask_module='down_proj' if block % 2 else 'gate_proj', columns=16,
                                  copy_weight=(0, 4, 16)[block % 3], copy_loss='target',
                                  rows=40, test_rows=40, steps=32, eval_draws=1,
                                  seed=24471 + block * 91 + lane, magnitude_control=True))
            yield natural_case(lane, block)
        yield from transfer_cases(lane, 1)
    elif lane < 2:
        model = 'q05' if lane == 0 else 'q15'
        for rep in range(4):
            yield from copy_cases(rep, model, 0)
            yield natural_case(lane, 6 + rep)
        yield from transfer_cases(lane, 2)
    else:
        for rep in range(3):
            yield from transfer_cases(lane, rep)
            yield natural_case(lane, 10 + rep)


def digest(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def gpu_idle(index, uuid, deadline):
    stop = min(deadline - 35, time.time() + 90)
    while time.time() < stop:
        try:
            device(index, uuid)
            return True
        except RuntimeError as e:
            if 'UUID' in str(e):
                raise
        time.sleep(5)
    return False


def argv_for(plan, lane, case, folder):
    entry = case['settings'].copy()
    home = Path(__file__).resolve().parent
    if case['kind'] == 'natural':
        entry.update(assets=plan['assets'], root=folder, gpu=lane, lane=lane,
                     uuid=plan['uuids'][lane], deadline=plan['deadline'])
        script, flags = 'natural_loop.py', ['--worker']
    else:
        model = entry.pop('model')
        entry.update(model=plan['models'][model], train=plan['train'], test=plan['test'],
                     output=folder/'result.json')
        script, flags = 'loop_deletion.py', []
    argv = [sys.executable, '-B', str(home/script), *flags]
    for key, value in entry.items():
        if value is False:
            continue
        argv.append('--'+key.replace('_','-'))
        if value is not True:
            argv.append(str(value))
    return argv


def worker(root, index):
    plan = json.loads((root/'plan.json').read_text())
    if plan['host'] != socket.gethostname():
        raise RuntimeError('wrong host')
    uuid, deadline = plan['uuids'][index], plan['deadline']
    home = Path(__file__).resolve().parent
    for name, expected in plan['source_sha256'].items():
        if digest(home/name) != expected:
            raise RuntimeError(f'source changed: {name}')
    folder = root/f'gpu{index}'
    folder.mkdir()
    path = folder/'status.json'
    state = {'gpu':index,'uuid':uuid,'deadline':deadline,'status':'running','jobs':[]}
    save(path,state)
    env = {k:v for k,v in os.environ.items() if k.lower() not in
           {'http_proxy','https_proxy','all_proxy','hf_token','hugging_face_hub_token','wandb_api_key'}}
    env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(index), HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false',
               PYTHONDONTWRITEBYTECODE='1')
    active = None
    def stop_child(sig=signal.SIGTERM):
        if active and active.poll() is None:
            os.killpg(active.pid,sig)
    def on_stop(_sig,_frame):
        stop_child()
        raise InterruptedError('worker interrupted')
    signal.signal(signal.SIGTERM,on_stop)
    signal.signal(signal.SIGINT,on_stop)
    failed_runtime = {}
    try:
        for case in plan['jobs'][index]:
            if deadline-time.time() < 120:
                state['status']='deadline';break
            kind = case['kind']
            if kind=='natural' and deadline-time.time()<600:
                continue
            runtime = 'natural' if kind=='natural' else case['settings']['model']
            if failed_runtime.get(runtime,0)>=3:
                continue
            if not gpu_idle(index,uuid,deadline):
                state['status']='device_became_busy';break
            dest=folder/case['id'];dest.mkdir()
            entry={'id':case['id'],'kind':kind,'status':'running','at':now()}
            state['jobs'].append(entry)
            with (dest/'stdout.log').open('x') as log:
                active=subprocess.Popen(argv_for(plan,index,case,dest),env=env,stdout=log,
                                        stderr=subprocess.STDOUT,start_new_session=True)
                entry['pid']=active.pid;save(path,state)
                max_seconds=6600 if kind=='natural' else 1200
                try:
                    code=active.wait(timeout=max(1,min(max_seconds,deadline-time.time()-20)))
                except subprocess.TimeoutExpired:
                    entry['timed_out']=True;stop_child()
                    try:code=active.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        stop_child(signal.SIGKILL);code=active.wait()
                active=None
            expected=('original.json','sparse_projection.json','function_fit.json') if kind=='natural' else ('result.json',)
            complete=code==0 and all((dest/k).is_file() for k in expected)
            entry.update(status='completed' if complete else 'failed',exit_code=code,finished=now(),
                         sha256={k:digest(dest/k) for k in expected if (dest/k).is_file()})
            if not complete and not (dest/'calibration.json').is_file() and not entry.get('timed_out'):
                failed_runtime[runtime]=failed_runtime.get(runtime,0)+1
            save(path,state)
        else:
            state['status']='exhausted_grid'
    except (InterruptedError,KeyboardInterrupt):
        state['status']='cancelled';stop_child()
    except Exception as e:
        state.update(status='failed',error=type(e).__name__+': '+str(e));stop_child();raise
    finally:
        if active and active.poll() is None:
            try:active.wait(timeout=8)
            except subprocess.TimeoutExpired:
                stop_child(signal.SIGKILL);active.wait()
        state['finished']=now();save(path,state)


def plan_for(args):
    p=json.loads(args.parent.read_text()); cfg=p['config']
    data={Path(path).name:path for path,h in p['data_sha256'].items() if Path(path).name in {'train.txt','test.txt'}}
    for path in data.values():
        if sha(path)!=p['data_sha256'][path]:
            raise RuntimeError('data hash changed')
    for key in ('q05','q15','q7i'):
        model=cfg['models'][key]
        for name,expected in cfg['model_sha256'][model].items():
            if sha(Path(model)/name)!=expected:
                raise RuntimeError('model hash changed: '+key)
    for i,uuid in enumerate(args.uuid):device(i,uuid)
    home=Path(__file__).resolve().parent
    return {'host':args.host,'deadline':args.deadline,'uuids':args.uuid,'train':data['train.txt'],
            'test':data['test.txt'],'models':cfg['models'],'assets':str(args.assets),
            'parent_sha256':digest(args.parent),'started':now(),
            'source_sha256':{name:digest(home/name) for name in (*SOURCES,Path(__file__).name)},
            'jobs':[[{'id':f'{n:03d}-{kind}','kind':kind,'settings':settings}
                     for n,(kind,settings) in enumerate(jobs(i))] for i in range(8)]}


def report(root):
    plan = json.loads((root/'plan.json').read_text())
    cache = {}
    def summarize(path, event):
        expected = event['sha256']
        for name, pin in expected.items():
            if digest(path/name) != pin:
                return {'integrity_error': name}
        if event['kind'] == 'natural':
            scores = {}
            for name in expected:
                metrics = json.loads((path/name).read_text())['per_problem']
                scores[name[:-5]] = {
                    k: statistics.fmean(r[k] for r in metrics) for k in
                    ('correct','correct_after_thinking','hit_cap','recurrent_16gram_steps','length')}
            return scores
        d = json.loads((path/'result.json').read_text())
        return {'eligible': d['eligible'], 'budget': d['edit_budget_by_layer'],
                'effects': {name: {k:v['mean'] for k,v in row.items() if k in (
                    'greedy_recurrence','sampling_recurrence','copy_motif_completed','greedy_length')}
                            for name,row in d['paired_deltas'].items()}}
    while True:
        rows, states = [], []
        for index in range(8):
            lane = root/f'gpu{index}'
            f = lane/'status.json'
            if not f.is_file():
                states.append({'gpu':index,'status':'unobserved'});continue
            s = json.loads(f.read_text())
            states.append({'gpu':index,'status':s['status'],
                           'completed':sum(j['status']=='completed' for j in s['jobs']),
                           'failed':sum(j['status']=='failed' for j in s['jobs'])})
            cases = {case['id']:case for case in plan['jobs'][index]}
            for event in s['jobs']:
                key = (index,event['id'])
                if event['status'] == 'completed' and key not in cache:
                    cache[key] = summarize(lane/event['id'],event)
                if key in cache:
                    rows.append({'gpu':index,**cases[event['id']],'metrics':cache[key]})
        save(root/'audit.json',{'observed':now(),'deadline':plan['deadline'],'lanes':states,'completed':rows})
        if time.time() >= plan['deadline'] or all(s['status'] not in ('running','unobserved') for s in states):
            break
        time.sleep(min(120,max(1,plan['deadline']-time.time())))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--worker',type=int,choices=range(8))
    p.add_argument('--report',action='store_true')
    p.add_argument('--host')
    p.add_argument('--uuid',nargs=8)
    p.add_argument('--parent',type=Path)
    p.add_argument('--assets',type=Path)
    p.add_argument('--deadline',type=int)
    args=p.parse_args()
    if args.worker is not None:
        worker(args.root,args.worker);return
    if args.report:
        report(args.root);return
    if not all((args.host,args.uuid,args.parent,args.assets,args.deadline)):
        p.error('host, eight UUIDs, parent, assets and deadline required')
    if args.host!=socket.gethostname() or len(set(args.uuid))!=8:
        p.error('incorrect host or duplicate GPUs')
    if args.root.exists() or not 180<args.deadline-time.time()<=36000:
        p.error('fresh root and active window of at most ten hours required')
    plan=plan_for(args)
    args.root.mkdir()
    save(args.root/'plan.json',plan)
    children={}
    for index in range(8):
        with (args.root/f'gpu{index}.log').open('x') as log:
            proc=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--root',str(args.root),
                                   '--worker',str(index)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children[index]=proc.pid
    with (args.root/'audit.log').open('x') as log:
        reporter=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--root',str(args.root),
                                   '--report'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(args.root/'launch.json',{'worker_pids':children,'deadline':args.deadline,'host':args.host,
                                  'reporter_pid':reporter.pid})
    print(json.dumps({'pid_by_gpu':children,'root':str(args.root),'deadline':args.deadline,
                      'reporter_pid':reporter.pid}))


if __name__=='__main__':
    main()
