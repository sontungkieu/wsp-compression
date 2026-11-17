#!/usr/bin/env python3
"""Run one small paired experiment on a verified, assigned GPU."""
import json
import os
from pathlib import Path
import subprocess
import sys
import six_gpu_campaign as campaign

root=Path(sys.argv[1]); old=Path(sys.argv[2]); selection=Path(sys.argv[3])
if root.exists(): raise SystemExit('fresh root required')
plan=json.loads(old.read_text()); cfg=plan['config']; spec=cfg['base_spec'].copy()
for key in ('model','train','wiki_validation','wiki_test','data_dir'):
    spec[key]=campaign.moved(spec[key])
spec.update({'gpu':2, 'model':campaign.moved(cfg['models']['q05']), 'mask_windows':1,
    'fit_windows':1, 'wiki_test_windows':2, 'lambada_offset':4352, 'lambada_n':2,
    'arc_offset':1664, 'arc_n':2, 'candidate':json.loads(selection.read_text())['candidate']})
source=campaign.moved(cfg['source'])
if campaign.sha(source)!=plan['source_sha256']: raise SystemExit('source mismatch')
campaign.device(2)
root.mkdir(); campaign.save(root/'spec.json',spec)
env={k:v for k,v in os.environ.items() if k.lower() not in
     {'http_proxy','https_proxy','all_proxy','hf_token','hugging_face_hub_token','wandb_api_key'}}
env.update({'CUDA_DEVICE_ORDER':'PCI_BUS_ID','CUDA_VISIBLE_DEVICES':'2','HF_HUB_OFFLINE':'1',
            'TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false',
            'PYTHONDONTWRITEBYTECODE':'1'})
with (root/'stdout.log').open('x') as log:
    done=subprocess.run([sys.executable,'-B',source,'--spec',str(root/'spec.json'),
                         '--output',str(root/'result.json')],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
if done.returncode: raise SystemExit('canary failed; inspect stdout.log')
result=json.loads((root/'result.json').read_text())
if result['spec']!=spec or result['source_sha256']!=plan['source_sha256']:
    raise SystemExit('canary result contract mismatch')
print(json.dumps({'status':'passed','arms':list(result['results']),'runtime':result['runtime'],
                  'result_sha256':campaign.sha(root/'result.json')}))
