#!/usr/bin/env python3
"""Run paired mask and ridge ablations on two assigned GPUs per host."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import six_gpu_campaign as campaign


def build_jobs(config, candidate, mask_offset):
    jobs = []
    for fraction in (.25, .5):
        for regularizer in (1.0, .1, 10.0):
            for metric in ('energy', 'variance'):
                for model in ('q05', 'q15'):
                    jobs.append({
                        'id': f'{model}_p{fraction}_m{mask_offset}_{metric}_l{regularizer}',
                        'group': 'mask_regularization',
                        'overrides': {
                            'model': config['models'][model],
                            'remove_fraction': fraction,
                            'mask_metric': metric,
                            'mask_offset': mask_offset,
                            'fit_offset': 0,
                            'ridge_lambda': regularizer,
                            'candidate': candidate,
                            'wiki_test_windows': 32,
                            'lambada_offset': 4352,
                            'lambada_n': 256,
                            'arc_offset': 1664,
                            'arc_n': 128,
                        },
                    })
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-plan', required=True, type=Path)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--uuid', required=True, nargs=2)
    parser.add_argument('--mask-offset', required=True, type=int, choices=(24576, 32768))
    args = parser.parse_args()
    if args.root.exists():
        raise SystemExit('fresh output root required')
    parent = json.loads(args.parent_plan.read_text())
    config = parent['config']
    if campaign.sha(config['source']) != parent['source_sha256']:
        raise SystemExit('experiment source hash mismatch')
    for path, expected in parent['data_sha256'].items():
        if campaign.sha(path) != expected:
            raise SystemExit('input data hash mismatch: ' + path)
    for model in ('q05', 'q15'):
        path = config['models'][model]
        for name, expected in config['model_sha256'][path].items():
            if campaign.sha(Path(path) / name) != expected:
                raise SystemExit('model hash mismatch: ' + name)
    devices = [campaign.device(index, uuid) for index, uuid in enumerate(args.uuid)]
    if parent['deadline_epoch'] - time.time() < 120:
        raise SystemExit('parent campaign deadline is too close')
    worker_path = Path(campaign.__file__).resolve()
    jobs = build_jobs(config, parent['selection']['candidate'], args.mask_offset)
    plan = {
        **{key: parent[key] for key in ('config', 'source_sha256', 'data_sha256',
                                      'selection', 'selection_sha256', 'deadline_epoch', 'deadline_utc')},
        'launched_at': campaign.now(),
        'parent_plan_sha256': campaign.sha(args.parent_plan),
        'queue_sha256': campaign.sha(worker_path),
        'launcher_sha256': campaign.sha(__file__),
        'devices': devices,
        'jobs': jobs,
    }
    root = args.root.resolve()
    root.mkdir()
    campaign.save(root / 'plan.json', plan)
    children = []
    for lane, device in enumerate(devices):
        folder = root / f'lane{lane}'
        folder.mkdir()
        with (folder / 'supervisor.log').open('x') as log:
            child = subprocess.Popen([sys.executable, '-B', str(worker_path), '--root',
                                      str(root), '--worker', str(lane)], stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        children.append({'lane': lane, 'gpu': device['gpu'], 'pid': child.pid})
    receipt = {'status': 'launched', 'jobs': len(jobs), 'children': children,
               'deadline_utc': plan['deadline_utc']}
    campaign.save(root / 'launch.json', receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
