"""语料与指标公共函数，源于已验证的 vLLM benchmark（原实验版本512ec02）。
用途：加载独立预热/正式语料，校验 PCM 去重，计算同口径 CER 与分位数。
依赖：numpy、soundfile；由同目录 benchmark.py 导入，无自动执行入口。
复用：通过 load_audio(manifest,total,warmup,seed,allow_repeat) 显式传参；
manifest 为含 id/path/text 的 JSONL，音频为 mono16k；返回预加载音频字节。
"""
import hashlib, io, json, random, statistics, unicodedata
from pathlib import Path
import numpy as np
import soundfile as sf

def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

def percentile(values, q):
    # 沿用旧基准的排序后线性插值，保证 p50/p95/p99 口径一致。
    if not values:
        return None
    a = sorted(values)
    pos = (len(a) - 1) * q
    i = int(pos)
    return a[i] + (a[min(i + 1, len(a)-1)] - a[i]) * (pos - i)

def distribution(values):
    return {'count': len(values), 'mean': statistics.mean(values) if values else None,
            **{k: percentile(values, q) for k, q in [('p50', .5), ('p95', .95), ('p99', .99)]},
            'max': max(values) if values else None}

def normalize_text(s):
    # 固定 CER 归一化口径；不做简繁、数字/中文数词转换。
    s = unicodedata.normalize('NFKC', s).lower()
    return ''.join(c for c in s if not c.isspace() and not unicodedata.category(c).startswith(('P', 'S')))

def edit_distance(a, b):
    # 按字符计算 Levenshtein 距离，插入/删除/替换各计一次；仅保留一行状态。
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        new = [i]
        for j, y in enumerate(b, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j-1] + (x != y)))
        row = new
    return row[-1]

def load_audio(manifest, total, warmup, seed, allow_repeat):
    # 所有磁盘读取、解码和哈希均在计时外完成；预热集与正式集隔离。
    lines = [json.loads(s) for s in manifest.read_text().splitlines() if s.strip()]
    ids = [r['id'] for r in lines]
    if len(ids) != len(set(ids)):
        raise ValueError('Manifest contains duplicate IDs')
    random.Random(seed).shuffle(lines)
    if len(lines) < total + warmup and not allow_repeat:
        raise ValueError(f'Need {total + warmup} distinct recordings; found {len(lines)}. Refusing implicit reuse.')
    if len(lines) <= warmup:
        raise ValueError('Need disjoint warmup and measured recordings')
    chosen = lines[:warmup] + (lines[warmup:warmup+total] if not allow_repeat else
                              [lines[warmup + i % (len(lines)-warmup)] for i in range(total)])
    seen = {}
    prepared = []
    for index, item in enumerate(chosen):
        path = Path(item['path'])
        if not path.is_absolute():
            path = manifest.parent / path
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if item.get('sha256') and digest != item['sha256']:
            raise ValueError(f'File SHA256 mismatch: {path}')
        pcm, sr = sf.read(io.BytesIO(data), dtype='float32', always_2d=True)
        # 对解码 PCM 去重，防止改文件名或封装元信息掩盖重复音频。
        if sr != 16000 or pcm.shape[1] != 1 or not len(pcm) or not np.isfinite(pcm).all():
            raise ValueError(f'Require nonempty finite mono 16 kHz audio: {path}')
        pcm_hash = hashlib.sha256(pcm[:, 0].astype('<f4').tobytes()).hexdigest()
        if item.get('pcm_sha256') and pcm_hash != item['pcm_sha256']:
            raise ValueError(f'PCM SHA256 mismatch: {path}')
        if pcm_hash in seen and (not allow_repeat or index < warmup or seen[pcm_hash] < warmup):
            raise ValueError(f'Duplicate decoded audio: {path}; first index={seen[pcm_hash]}')
        seen.setdefault(pcm_hash, index)
        prepared.append({**item, 'path': str(path), 'data': data, 'sha256': digest,
                         'pcm_sha256': pcm_hash, 'duration_s': len(pcm) / sr})
    return prepared[:warmup], prepared[warmup:]
