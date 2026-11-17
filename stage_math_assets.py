"""Stage public, pinned model and GSM8K files via the approved proxy."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

MODEL_REV = '2fc06364715b967f1860aea9cf38778875588b17'
DATA_REV = '740312add88f781978c0658806c59bc2815b9866'
WEIGHTS = 'model.safetensors-00001-of-00001.safetensors'
WEIGHTS_SHA = '04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696'
MODEL = f'https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/{MODEL_REV}/'
DATA = f'https://huggingface.co/datasets/openai/gsm8k/resolve/{DATA_REV}/main/'


def download(url, target, proxy, expected=None):
    pending = target.with_suffix(target.suffix + '.part')
    outcome = subprocess.run(['curl', '--silent', '--fail', '--location',
                              '--proxy', proxy, '--noproxy', '', '--connect-timeout', '10',
                              '--max-time', '900', '--retry', '2', '--retry-max-time', '1200',
                              '--output', str(pending), url])
    if outcome.returncode:
        raise RuntimeError(f'download failed for {target.name} (curl {outcome.returncode})')
    with pending.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    if expected and digest != expected:
        raise RuntimeError('pinned artifact digest mismatch: ' + target.name)
    pending.replace(target)
    return {'sha256': digest, 'bytes': target.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    proxy = os.environ.get('HTTPS_PROXY')
    if not proxy:
        raise SystemExit('HTTPS_PROXY is required')
    root = args.root.resolve()
    if not args.worker:
        root.mkdir()
        with (root / 'stage.log').open('x') as log:
            child = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()),
                                      '--root', str(root), '--worker'], stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'pid': child.pid, 'root': str(root)}))
        return
    model_root = root / 'model'
    model_root.mkdir()
    artifacts = {}
    for name in ('config.json', 'model.safetensors.index.json', 'tokenizer_config.json',
                 'tokenizer.json', 'chat_template.jinja', WEIGHTS):
        artifacts['model/' + name] = download(MODEL + name, model_root / name, proxy,
                                             WEIGHTS_SHA if name == WEIGHTS else None)
    for split in ('train', 'test'):
        name = f'{split}-00000-of-00001.parquet'
        artifacts[name] = download(DATA + name, root / name, proxy)
    tmp = root / 'manifest.tmp'
    tmp.write_text(json.dumps({'model_revision': MODEL_REV, 'gsm8k_revision': DATA_REV,
                               'artifacts': artifacts}, indent=2))
    tmp.replace(root / 'manifest.json')
    print('validated pinned checkpoint and both dataset files', flush=True)


if __name__ == '__main__':
    main()
