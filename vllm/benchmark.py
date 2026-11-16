#!/usr/bin/env python3
"""ASR 闭环压测（沿用已验证的客户端逻辑）。

依赖 Python>=3.10、aiohttp、numpy、soundfile；输入为 JSONL manifest 和其音频文件，
目标服务须支持 /v1/audio/transcriptions。任意目录执行：
python /路径/benchmark.py --manifest /语料/manifest.jsonl --output ./bench01 \
 --base-url http://127.0.0.1:8001 --model /模型目录 --requests 2000 --concurrency 1000
主要参数：--warmup=32、--seed=20261112、--timeout=600 秒、--cache-mode=disabled；
使用独立服务时增加 --warmup-server-log /运行/server.log --expected-api-workers 4，
确保四个 API 都处理过预热。--output 必须不存在。输出逐请求文本/耗时/PCM 哈希、
CER、summary.json、预热覆盖、缓存 metrics；预热不进入正式计时和 CER。
复用时更换 manifest、设备外部服务地址、模型、请求数和并发；唯一 PCM 与缓存
检查保留。推荐 run_service.py 自动传入服务快照、预热日志和阶段文件。
"""
import argparse
import asyncio
from collections import Counter
import hashlib
import io
import json
import math
import mimetypes
from pathlib import Path
import random
import re
import statistics
import time
import unicodedata
from urllib.parse import urlsplit

import aiohttp
import numpy as np
import soundfile as sf


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def percentile(values, q):
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
    # Reproducible CER policy; no simplified/traditional or number conversion.
    s = unicodedata.normalize('NFKC', s).lower()
    return ''.join(c for c in s if not c.isspace() and not unicodedata.category(c).startswith(('P', 'S')))


def edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        new = [i]
        for j, y in enumerate(b, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j-1] + (x != y)))
        row = new
    return row[-1]


def load_audio(manifest, total, warmup, seed, allow_repeat):
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
        # Require canonical uploads so file metadata/encodings cannot disguise duplicate PCM.
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


def metric_totals(text):
    values = Counter()
    for line in text.splitlines():
        if not line or line.startswith('#'):
            continue
        match = re.match(r'([^\s{]+)(?:\{.*\})?\s+([^\s]+)', line)
        if match:
            try:
                value = float(match[2])
                if math.isfinite(value):
                    values[match[1]] += value
            except ValueError:
                pass
    return dict(values)


def cache_config(info):
    cfg = info.get('vllm_config')
    if not isinstance(cfg, dict):
        raise ValueError('/server_info?config_format=json did not return a JSON vllm_config')
    prefix = cfg.get('cache_config', {}).get('enable_prefix_caching')
    mm = cfg.get('model_config', {}).get('multimodal_config') or cfg.get('multimodal_config') or {}
    size = mm.get('mm_processor_cache_gb')
    return {'enable_prefix_caching': prefix, 'mm_processor_cache_gb': size}


def launch_cache_config(path, endpoint):
    snapshot = json.loads(path.read_text())
    command = snapshot['command']
    # A launch manifest alone is insufficient: verify the local live process and endpoint.
    url = urlsplit(endpoint)
    if url.hostname not in ('127.0.0.1', 'localhost'):
        raise ValueError('Launch snapshot fallback requires a local dedicated service')
    runtime = Path(f"/proc/{int(snapshot['server_pid'])}/cmdline").read_bytes().decode().strip('\0').split('\0')
    if runtime[-(len(command)-1):] != command[1:]:
        raise ValueError('Launch snapshot does not match the live server command')
    def option(name):
        if command.count(name) != 1:
            raise ValueError(f'Launch snapshot must explicitly set {name} exactly once')
        return command[command.index(name)+1]
    if int(option('--port')) != url.port:
        raise ValueError('Launch snapshot port differs from benchmark endpoint')
    prefix_flags = [flag for flag in command if flag in ('--enable-prefix-caching', '--no-enable-prefix-caching')]
    if len(prefix_flags) != 1:
        raise ValueError('Launch snapshot must explicitly set exactly one prefix cache flag')
    return {'enable_prefix_caching': prefix_flags[0] == '--enable-prefix-caching', 'mm_processor_cache_gb': float(option('--mm-processor-cache-gb'))}


async def run(args):
    # Decoding, hashing and reads happen before any measured request.
    warm, items = load_audio(args.manifest, args.requests, args.warmup, args.seed, args.allow_repeat)
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output
    save_json(out/'input_order.json', [{k: v for k, v in x.items() if k != 'data'} for x in warm + items])
    endpoint = args.base_url.rstrip('/')
    if endpoint.endswith('/v1'):
        endpoint = endpoint[:-3]
    headers = {'Authorization': f'Bearer {args.api_key}'}
    warm_concurrency=getattr(args,'warmup_concurrency',None) or min(args.concurrency,8)
    connection_limit=max(args.concurrency,warm_concurrency)
    connector = aiohttp.TCPConnector(limit=connection_limit, limit_per_host=connection_limit)
    timeout = aiohttp.ClientTimeout(total=args.timeout, connect=min(30, args.timeout))
    async with aiohttp.ClientSession(connector=connector, timeout=timeout, headers=headers,
                                    trust_env=False) as session:
        async def get(path):
            async with session.get(endpoint + path) as r:
                r.raise_for_status()
                return await r.text()

        try:
            info = json.loads(await get('/server_info?config_format=json'))
            config = cache_config(info)
            config_evidence = 'server_info'
        except aiohttp.ClientResponseError as exc:
            if exc.status != 404 or not getattr(args, 'launch_snapshot', None):
                raise
            config = launch_cache_config(args.launch_snapshot, endpoint)
            config_evidence = 'launch_snapshot_and_live_process'
            info = {'server_info_status': 404, 'cache_config': config,
                    'config_evidence': config_evidence,
                    'launch_snapshot': json.loads(args.launch_snapshot.read_text())}
        save_json(out/'server_info.json', info)
        if args.cache_mode == 'disabled' and not (
                config['enable_prefix_caching'] is False and config['mm_processor_cache_gb'] == 0):
            raise ValueError(f'Cache guard failed: {config}. Launch an isolated service with '
                             '--no-enable-prefix-caching --mm-processor-cache-gb 0; '
                             'use --cache-mode observe only for a deliberate cache experiment.')
        if args.cache_mode == 'prefix-only' and not (
                config['enable_prefix_caching'] is True and config['mm_processor_cache_gb'] == 0):
            raise ValueError(f'Prefix-only cache guard failed: {config}')
        save_json(out/'run_config.json', {k: str(v) if isinstance(v, Path) else v
                                         for k, v in vars(args).items() if k != 'api_key'})
        print(json.dumps({'stage': 'prepared', 'cache': config, 'warmup': len(warm),
                          'requests': len(items), 'concurrency': args.concurrency,
                          'unique_measured_pcm': len({x['pcm_sha256'] for x in items}),
                          'upload_bytes': sum(len(x['data']) for x in items)}, ensure_ascii=False), flush=True)

        async def request(item, index, origin):
            form = aiohttp.FormData()
            form.add_field('model', args.model)
            form.add_field('language', args.language)
            form.add_field('response_format', 'json')
            form.add_field('temperature', '0')
            form.add_field('file', item['data'], filename=Path(item['path']).name,
                           content_type=mimetypes.guess_type(item['path'])[0] or 'application/octet-stream')
            start = time.perf_counter()
            record = {'index': index, 'id': item['id'], 'pcm_sha256': item['pcm_sha256'],
                      'duration_s': item['duration_s'], 'reference': item.get('text'),
                      'start_offset_s': start - origin}
            try:
                async with session.post(endpoint + '/v1/audio/transcriptions', data=form,
                                        headers={'X-Request-Id': f'asrbench-{out.name}-{index}',
                                                 **({'Connection': 'close'} if index < 0 else {})}) as r:
                    body = await r.text()
                    record['http_status'] = r.status
                    if r.status != 200:
                        raise RuntimeError(f'HTTP {r.status}: {body}')
                    payload = json.loads(body)
                    if not isinstance(payload.get('text'), str):
                        raise ValueError('Response has no text string')
                    record.update(success=True, text=payload['text'], response=payload)
            except Exception as exc:
                record.update(success=False, error=f'{type(exc).__name__}: {exc}')
            end = time.perf_counter()
            record.update(latency_s=end-start, end_offset_s=end-origin)
            return record

        if getattr(args, 'stage_file', None):
            args.stage_file.write_text('warmup')
        # Bounded warmup with a separate, disjoint input set; excluded from timings/metrics deltas.
        # 低并发正式测量也须预热所有 API；连接容量与预热并发独立配置。
        sem = asyncio.Semaphore(warm_concurrency)
        async def warm_request(i, item):
            async with sem:
                return await request(item, -1-i, time.perf_counter())
        warm_log = getattr(args, 'warmup_server_log', None)
        expected_workers = getattr(args, 'expected_api_workers', 0)
        log_offset = warm_log.stat().st_size if warm_log else 0
        warm_results, warm_workers = [], set()
        for warm_round in range(16 if expected_workers else 1):
            current = await asyncio.gather(*(warm_request(warm_round*len(warm)+i, x)
                                             for i, x in enumerate(warm)))
            warm_results.extend(current)
            save_json(out/'warmup.json', warm_results)
            if any(not r['success'] for r in current):
                raise RuntimeError('Warmup failed; see warmup.json')
            if not expected_workers:
                break
            # Only our isolated service writes this log; capture current warmup responses.
            with warm_log.open() as f:
                f.seek(log_offset)
                warm_workers.update(re.findall(r'ApiServer_(\d+).*POST /v1/audio/transcriptions.*200 OK', f.read()))
            coverage = dict(expected=expected_workers, observed=sorted(warm_workers),
                            unique_audio=len(warm), warmup_requests=len(warm_results),
                            verified=len(warm_workers)==expected_workers)
            save_json(out/'warmup_workers.json',coverage)
            if coverage['verified']:
                break
            await asyncio.sleep(.25)
        else:
            raise RuntimeError('Warmup did not reach every API worker; refusing measurement')
        if getattr(args, 'stage_file', None):
            args.stage_file.write_text('measurement')
            await asyncio.sleep(.1)  # Let the CPU-only phase marker reach the model runner.
        before_text = await get('/metrics')
        (out/'metrics_before.txt').write_text(before_text)
        before = metric_totals(before_text)
        results, lag = [], []
        active = peak = done = 0
        integral = 0.0
        origin = last_change = time.perf_counter()
        task_iterator = iter(enumerate(items))
        results_file = (out/'requests.jsonl').open('w')

        def change_active(delta):
            nonlocal active, peak, integral, last_change
            now = time.perf_counter()
            integral += active * (now-last_change)
            active += delta
            peak = max(peak, active)
            last_change = now

        async def worker():
            nonlocal done
            for i, item in task_iterator:
                change_active(1)
                try:
                    result = await request(item, i, origin)
                finally:
                    change_active(-1)
                results.append(result)
                results_file.write(json.dumps(result, ensure_ascii=False) + '\n')
                done += 1
                if done % 64 == 0:
                    results_file.flush()

        async def monitor():
            last_report = origin
            while True:
                start = time.perf_counter()
                await asyncio.sleep(.05)
                now = time.perf_counter()
                lag.append(max(0.0, now-start-.05))
                if now-last_report > 5:
                    print(json.dumps({'stage': 'progress', 'done': done, 'active': active,
                                      'elapsed_s': now-origin}), flush=True)
                    last_report = now

        monitoring = asyncio.create_task(monitor())
        try:
            await asyncio.gather(*(worker() for _ in range(min(args.concurrency, len(items)))))
            elapsed = time.perf_counter() - origin
        finally:
            monitoring.cancel()
            await asyncio.gather(monitoring, return_exceptions=True)
            results_file.close()
        if getattr(args, 'stage_file', None):
            args.stage_file.write_text('complete')
        after_text = await get('/metrics')
        (out/'metrics_after.txt').write_text(after_text)
    after = metric_totals(after_text)
    # Deltas are meaningful only on a dedicated server without other clients.
    delta = {k: v-before.get(k, 0) for k, v in after.items()
             if k.endswith(('_total', '_sum', '_count'))}
    save_json(out/'metric_deltas.json', delta)
    good = [r for r in results if r['success']]
    errors = chars = compared = 0
    for r in good:
        if r['reference'] is not None:
            ref, hyp = normalize_text(r['reference']), normalize_text(r['text'])
            errors += edit_distance(ref, hyp)
            chars += len(ref)
            compared += 1
    hits = {k: v for k, v in delta.items() if k in (
        'vllm:prefix_cache_hits_total', 'vllm:mm_cache_hits_total', 'vllm:external_prefix_cache_hits_total')}
    counters_missing = [k for k in ('vllm:prefix_cache_hits_total', 'vllm:mm_cache_hits_total') if k not in delta]
    counters_reset = [k for k, v in delta.items() if k.endswith('_total') and v < 0]
    cache_verified = (args.cache_mode == 'disabled' and not counters_missing and not counters_reset
                      and all(v == 0 for v in hits.values()))
    prefix_verified = (args.cache_mode == 'prefix-only' and not counters_missing and not counters_reset
                       and delta.get('vllm:mm_cache_hits_total') == 0
                       and delta.get('vllm:mm_cache_queries_total') == 0
                       and delta.get('vllm:prefix_cache_hits_total', -1) >= 0
                       and delta.get('vllm:prefix_cache_queries_total', -1) > 0)
    queries = delta.get('vllm:prefix_cache_queries_total', 0)
    summary = {
        'load_model': 'closed_loop_fixed_concurrency', 'requests': len(results),
        'success': len(good), 'failed': len(results)-len(good), 'elapsed_s': elapsed,
        'successful_requests_per_s': len(good)/elapsed,
        'successful_audio_s_per_wall_s': sum(r['duration_s'] for r in good)/elapsed,
        'request_latency_s': distribution([r['latency_s'] for r in good]),
        'request_rtf': distribution([r['latency_s']/r['duration_s'] for r in good]),
        'audio_duration_s': distribution([r['duration_s'] for r in results]),
        'client_in_flight_peak': peak, 'client_in_flight_time_mean': integral/elapsed,
        'client_event_loop_lag_s': distribution(lag),
        'initial_dispatch_span_s': (max(r['start_offset_s'] for r in results if r['index'] < min(args.concurrency, len(items)))
                                    - min(r['start_offset_s'] for r in results)),
        'unique_measured_pcm': len({r['pcm_sha256'] for r in results}),
        'cache_config': config, 'cache_config_evidence': config_evidence, 'cache_hit_deltas': hits, 'cache_counters_missing': counters_missing,
        'counters_reset': counters_reset, 'cache_disabled_verified': cache_verified,
        'prefix_only_verified': prefix_verified,
        'prefix_cache_hit_rate': delta.get('vllm:prefix_cache_hits_total', 0)/queries if queries else None,
        'cer': errors/chars if chars else None, 'cer_edit_distance': errors,
        'cer_reference_chars': chars, 'cer_compared_successful_requests': compared,
        'cer_normalization': 'NFKC+lower; remove Unicode punctuation, symbols and whitespace',
        'generation_tokens_server_delta': delta.get('vllm:generation_tokens_total'),
        'warmup_connections': 'fresh_connection_per_request',
        'actual_warmup_requests': len(warm_results),
        'warmup_api_workers_verified': len(warm_workers)==expected_workers if expected_workers else None,
        'retries': 0,
    }
    summary['valid'] = (len(results) == args.requests and len(good) == args.requests
                        and not counters_reset and (args.cache_mode == 'observe' or cache_verified or prefix_verified))
    save_json(out/'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary['valid'] else 2


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='New output directory; never overwritten')
    p.add_argument('--base-url', default='http://127.0.0.1:8001')
    p.add_argument('--model', default='/data/models/Qwen3-ASR-1.7B')
    p.add_argument('--language', default='zh')
    p.add_argument('--concurrency', type=int, default=1000)
    p.add_argument('--requests', type=int, default=2000)
    p.add_argument('--warmup-server-log', type=Path)
    p.add_argument('--expected-api-workers', type=int, default=0)
    p.add_argument('--warmup', type=int, default=32)
    p.add_argument('--warmup-concurrency',type=int,help='独立预热并发；默认 min(正式并发,8)')
    p.add_argument('--launch-snapshot', type=Path, help='Fallback for a local dedicated server without /server_info; live PID/command/port are checked')
    p.add_argument('--stage-file', type=Path, help='Optional private server instrumentation phase marker')
    p.add_argument('--seed', type=int, default=20261112)
    p.add_argument('--timeout', type=float, default=600)
    p.add_argument('--api-key', default='dummy')
    p.add_argument('--cache-mode', choices=['disabled', 'prefix-only', 'observe'], default='disabled')
    p.add_argument('--allow-repeat', action='store_true', help='Deliberate repeated-input experiment only')
    a = p.parse_args()
    if a.expected_api_workers and (not a.warmup_server_log or a.warmup < 1):
        p.error('Worker coverage requires a server log and at least one disjoint warmup audio')
    if min(a.concurrency, a.requests) < 1 or a.warmup < 0 or a.timeout <= 0:
        p.error('concurrency, requests, timeout must be positive and warmup nonnegative')
    if a.warmup_concurrency is not None and a.warmup_concurrency<1:
        p.error('warmup-concurrency 必须为正')
    return a


if __name__ == '__main__':
    raise SystemExit(asyncio.run(run(parse_args())))
