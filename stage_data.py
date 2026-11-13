#!/usr/bin/env python3
"""Stage small pinned public evaluation files; no package installation."""
import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path

FILES = [
    ('lambada', 'EleutherAI/lambada_openai', '900124bf3b8235c6daf21033af9948b3f07346c4',
     'en/test/en.parquet', '99d8a9e54cda761c6340ddd4fa6feda7f846ffb0e716afa52968dea75495eea7'),
    ('arc_validation', 'allenai/ai2_arc', '210d026faf9955653af8916fad021475a3f00453',
     'ARC-Easy/validation-00000-of-00001.parquet', 'ed890ff1e4cef7a7140d3a30dcea3ed2c9d467c6458f447ad9ef0176d8dcbb74'),
    ('arc_test', 'allenai/ai2_arc', '210d026faf9955653af8916fad021475a3f00453',
     'ARC-Easy/test-00000-of-00001.parquet', '4160597d618ae851c7eb04e281574f3f654776216ac6b6641588d64527b47177')]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--proxy-file', required=True)
    a = p.parse_args()
    root = Path(a.output)
    root.mkdir(exist_ok=False)
    proxy = Path(a.proxy_file).read_text().strip()
    if not proxy.startswith('http://') or '@' in proxy:
        raise SystemExit('expected credential-free HTTP proxy')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({'http': proxy, 'https': proxy}))
    import pyarrow.parquet as pq
    manifest = []
    for name, repo, rev, filename, expected in FILES:
        url = f'https://huggingface.co/datasets/{repo}/resolve/{rev}/{filename}?download=true'
        dest = root / (name + '.parquet')
        h = hashlib.sha256()
        with opener.open(url, timeout=120) as response, dest.open('xb') as out:
            while True:
                chunk = response.read(1024*1024)
                if not chunk:
                    break
                h.update(chunk)
                out.write(chunk)
        if h.hexdigest() != expected:
            raise RuntimeError('download hash mismatch: ' + name)
        rows = pq.read_table(dest).to_pylist()
        converted = root / (name + '.jsonl')
        with converted.open('x') as out:
            for i, row in enumerate(rows):
                if name == 'lambada':
                    record = {'id': i, 'text': row['text']}
                else:
                    record = {'id': row['id'], 'question': row['question'],
                              'choices': row['choices']['text'],
                              'gold': row['choices']['label'].index(row['answerKey'])}
                out.write(json.dumps(record, ensure_ascii=False) + '\n')
        entry = {'name': name, 'repo': repo, 'revision': rev, 'file': filename,
                 'sha256': expected, 'rows': len(rows),
                 'jsonl_sha256': hashlib.sha256(converted.read_bytes()).hexdigest()}
        manifest.append(entry)
        print(json.dumps(entry), flush=True)
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

if __name__ == '__main__':
    main()
