#!/usr/bin/env python3
"""Read compact progress, selected candidates and recent verified receipts."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('root')
parser.add_argument('--last', type=int, default=5)
args = parser.parse_args()
root = Path(args.root)
for lane_root in sorted(root.glob('lane*')):
    if not (lane_root / 'progress.json').is_file():
        continue
    progress = json.loads((lane_root / 'progress.json').read_text())
    status = {key: progress.get(key) for key in ('status','gpu','pid','error','deadline_utc')}
    status['lane'] = lane_root.name
    status['completed'] = sum(job['status']=='completed' for job in progress['jobs'])
    status['active'] = [job['id'] for job in progress['jobs'] if job['status']=='running']
    status['failed'] = [job['id'] for job in progress['jobs'] if job['status']=='failed']
    print(json.dumps(status))
    if (lane_root / 'selection.json').is_file():
        print(json.dumps({'selection':json.loads((lane_root/'selection.json').read_text())}))
    completed = [job for job in progress['jobs'] if job['status']=='completed']
    for job in completed[-args.last:]:
        raw = Path(job['output']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=job['sha256']:
            raise RuntimeError('receipt digest mismatch')
        result = json.loads(raw)
        scores = {name:{corpus:{key:value for key,value in metrics.items() if key!='records'}
                        for corpus,metrics in arm['scores'].items()}
                  for name,arm in result['results'].items()}
        print(json.dumps({'id':job['id'],'candidate':result['spec'].get('candidate'),
                          'scores':scores}))
