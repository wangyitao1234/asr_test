#!/usr/bin/env python3
"""下载/转换固定版本 AISHELL 分片，生成去重的 mono16k PCM16 WAV 和清单。
依赖 Python>=3.10、numpy、soundfile、pyarrow；需要约1GB磁盘和网络（下载时）。
在仓库根目录运行：python corpus/prepare_corpus.py --download \
 --parquet data/aishell.parquet --output data/corpus_v1
参数：parquet 是源文件，output 必须不存在；download 仅在源文件缺失时下载。
输出 manifest.jsonl 含 id/path/text/文件与PCM哈希，provenance.json 含来源与版本。
固定 revision 与 SHA256 保证同一语料；不包含数据集授权，请遵守来源使用条款。
复用已有源文件时省略 download；使用其他数据集应另写 manifest 适配器，
不要把不同数据集伪装成此固定 SHA 的来源。正式压测应使用独立预热/测量集合。
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

REPO = 'AudioLLMs/aishell_1_zh_test'
REVISION = 'e73dc7f096ecea5c8e5c722446c8d3b4014fdd18'
FILE = 'data/test-00000-of-00003.parquet'
SHA256 = '2466b1e851147943d4a2c46e1dddc8a70a45b733065c7f6a86e44bd88d5579c6'
URL = f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{FILE}'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parquet', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--download', action='store_true')
    a = p.parse_args()
    if not a.parquet.exists() and a.download:
        a.parquet.parent.mkdir(parents=True, exist_ok=True)
        temp = a.parquet.with_suffix('.partial')
        with urllib.request.urlopen(URL, timeout=60) as response, temp.open('wb') as f:
            while block := response.read(4 * 1024 * 1024):
                f.write(block)
        temp.rename(a.parquet)
    h = hashlib.sha256()
    with a.parquet.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    if h.hexdigest() != SHA256:
        raise ValueError('Source parquet SHA256 mismatch')
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output/'audio').mkdir()
    seen, rows, duplicates = set(), [], 0
    speakers = set()
    for batch in pq.ParquetFile(a.parquet).iter_batches(batch_size=64):
        for item in batch.to_pylist():
            context = item['context']
            if not isinstance(context, dict) or not context.get('bytes'):
                raise ValueError('Expected embedded audio bytes')
            pcm, sr = sf.read(io.BytesIO(context['bytes']), dtype='float32', always_2d=True)
            if sr != 16000 or pcm.shape[1] != 1 or not len(pcm) or not np.isfinite(pcm).all():
                raise ValueError(f'Unexpected source format: {sr}, {pcm.shape}')
            # 统一输出 PCM16 WAV，对真正上传后解码的 PCM 哈希，避免封装掩盖重复。
            relative = Path('audio') / f'{len(rows):06d}.wav'
            path = a.output / relative
            sf.write(path, pcm, sr, subtype='PCM_16')
            decoded, _ = sf.read(path, dtype='float32')
            pcm_hash = hashlib.sha256(decoded.astype('<f4').tobytes()).hexdigest()
            if pcm_hash in seen:
                path.unlink()
                duplicates += 1
                continue
            seen.add(pcm_hash)
            source_path = context.get('path') or ''
            # 保存原始路径作为来源证据；只有来源含说话人标识时才记录。
            import re
            match = re.search(r'(S\d{4})', source_path)
            speaker = match[1] if match else None
            if speaker:
                speakers.add(speaker)
            rows.append({'id': f'aishell-test-shard0-{len(rows):06d}', 'path': str(relative),
                         'text': item['answer'], 'duration_s': len(decoded)/sr,
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                         'pcm_sha256': pcm_hash, 'source_path': source_path, 'speaker': speaker})
    (a.output/'manifest.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in rows))
    meta = {'dataset': REPO, 'revision': REVISION, 'file': FILE, 'url': URL,
            'source_sha256': SHA256, 'unique_recordings': len(rows), 'duplicates_removed': duplicates,
            'total_audio_s': sum(x['duration_s'] for x in rows),
            'duration_min_s': min(x['duration_s'] for x in rows),
            'duration_max_s': max(x['duration_s'] for x in rows),
            'speaker_ids_in_source_paths': sorted(speakers),
            'note': 'Public read-speech subset; not a business-domain corpus or a 60-second workload.'}
    (a.output/'provenance.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
