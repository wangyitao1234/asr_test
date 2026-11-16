#!/usr/bin/env python3
"""生成明确标注的约60秒拼接压力语料，保持原始语音采样，不加噪声。
依赖 Python>=3.10、numpy、soundfile；输入为 prepare_corpus.py 生成的 manifest。
在仓库根目录运行：python corpus/make_composites.py \
 --manifest data/corpus_v1/manifest.jsonl --output data/corpus_composite60 \
 --count 2032 --target-seconds 60 --seed 20261112
参数：count 默认2032条，target-seconds 默认至少60秒，seed 固定选择顺序。
输出 WAV、manifest.jsonl 和 provenance.json，output 必须不存在；路径相对输出目录。
完整波形去重，但不同拼接会复用底层句子，不能当作独立自然长录音。
复用时修改 manifest、数量、时长和种子；评测需关闭内容缓存，另报长音频结果。
"""
import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import soundfile as sf


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--count', type=int, default=2032)
    p.add_argument('--target-seconds', type=float, default=60)
    p.add_argument('--seed', type=int, default=20261112)
    a = p.parse_args()
    source = [json.loads(x) for x in a.manifest.read_text().splitlines() if x.strip()]
    if a.count < 1 or a.target_seconds <= 0 or len(source) < 2:
        p.error('Need positive count/duration and at least two source recordings')
    audio = []
    for item in source:
        path = Path(item['path'])
        if not path.is_absolute():
            path = a.manifest.parent/path
        pcm, sr = sf.read(path, dtype='int16')
        if sr != 16000 or pcm.ndim != 1:
            raise ValueError('Source must be mono PCM16 at 16 kHz')
        audio.append(pcm)
    if sum(len(x) for x in audio)/16000 < a.target_seconds:
        raise ValueError('Insufficient unique source speech to create a composite')
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output/'audio').mkdir()
    rng = random.Random(a.seed)
    silence = np.zeros(2400, dtype=np.int16)  # 完整句子之间加入150毫秒静音，不裁切词句。
    seen = set()
    total = 0
    with (a.output/'manifest.jsonl').open('w') as manifest:
        for i in range(a.count):
            indices = rng.sample(range(len(source)), len(source))
            selected, chunks, frames = [], [], 0
            for j in indices:
                if selected:
                    chunks.append(silence)
                    frames += len(silence)
                selected.append(j)
                chunks.append(audio[j])
                frames += len(audio[j])
                if frames >= a.target_seconds*16000:
                    break
            pcm = np.concatenate(chunks)
            path = Path('audio')/f'{i:06d}.wav'
            sf.write(a.output/path, pcm, 16000, subtype='PCM_16')
            digest = hashlib.sha256((pcm.astype('float32')/32768).astype('<f4').tobytes()).hexdigest()
            if digest in seen:
                raise ValueError('Duplicate composite; choose a different seed')
            seen.add(digest)
            item = {'id': f'aishell-composite60-{i:06d}', 'path': str(path),
                    'text': ''.join(source[j]['text'] for j in selected), 'duration_s': len(pcm)/16000,
                    'sha256': hashlib.sha256((a.output/path).read_bytes()).hexdigest(),
                    'pcm_sha256': digest, 'source_ids': [source[j]['id'] for j in selected],
                    'derived_composite': True}
            manifest.write(json.dumps(item, ensure_ascii=False)+'\n')
            total += item['duration_s']
            if (i+1) % 200 == 0:
                print('prepared', i+1, flush=True)
    meta = {'profile': 'derived_composite60', 'count': a.count, 'seed': a.seed,
            'target_min_s': a.target_seconds, 'total_audio_s': total,
            'source_manifest_sha256': hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
            'gap_s': .15, 'source_utterances_reused_across_composites': True,
            'note': 'Duration stress test; not natural continuous recordings or independent business audio.'}
    (a.output/'provenance.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == '__main__':
    main()
