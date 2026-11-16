#!/usr/bin/env python3
"""FunASR HTTP 闭环性能测试；使用原 Qwen benchmark 的语料和 CER 公共函数。

依赖 Python>=3.10、aiohttp、numpy、soundfile；目标须支持 OpenAI 转写接口、
/health、/v1/models。无需 GPU，建议在服务所在容器执行，排除外网干扰。
任意目录运行（corpus.py 必须与本脚本同目录）：
python /实验目录/benchmark.py --manifest /语料/manifest.jsonl --output /结果/run1 \
 --base-url http://127.0.0.1:8000 --model custom --requests 2000 --concurrency 32
参数：warmup=32（独立音频，不进入计时），seed=20261112，timeout=600 秒；
output 必须不存在。requests/concurrency 分别控制总请求数/闭环最大在途数。
输出：逐请求 JSONL、输入顺序/PCM 哈希、预热、服务状态、吞吐、延迟、CER。
计时含上传、服务排队、推理和响应；不含加载语料、预热及 CER 计算。不重试。
缓存：校验不同 PCM 和预热隔离；FunASR 无缓存计数器，绝不伪造“缓存为零”。
复用：更换 manifest、URL、model、并发和请求数；新接口只改 request()。
注意 2000 条有限闭环有自然收尾阶段，不能代表持续固定并发的稳定性测试。
"""
import argparse
import asyncio
import json
from pathlib import Path
import time

import aiohttp
from corpus import load_audio, save_json, distribution, normalize_text, edit_distance


async def run(args):
    warm, items = load_audio(args.manifest, args.requests, args.warmup, args.seed, False)
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output
    save_json(out/'config.json', {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()})
    save_json(out/'input_order.json', [{k:v for k,v in x.items() if k!='data'} for x in warm+items])
    endpoint = args.base_url.rstrip('/').removesuffix('/v1')
    connector = aiohttp.TCPConnector(limit=max(args.concurrency, args.warmup_concurrency))
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout, trust_env=False) as session:
        async def get(path):
            async with session.get(endpoint+path) as response:
                response.raise_for_status()
                return await response.json()
        health, models = await get('/health'), await get('/v1/models')
        if health.get('status') != 'ok' or args.model not in [x['id'] for x in models['data']]:
            raise ValueError(f'服务检查失败：{health}, {models}')
        save_json(out/'server.json', dict(health=health, models=models))

        async def request(item, index, origin):
            form = aiohttp.FormData()
            for key, value in [('model',args.model),('language',args.language),('response_format','json')]:
                form.add_field(key,value)
            form.add_field('file',item['data'],filename=Path(item['path']).name,content_type='audio/wav')
            start=time.perf_counter()
            record=dict(index=index,id=item['id'],pcm_sha256=item['pcm_sha256'],
                        duration_s=item['duration_s'],reference=item.get('text'),start_offset_s=start-origin)
            try:
                async with session.post(endpoint+'/v1/audio/transcriptions',data=form) as response:
                    body=await response.text()
                    record['http_status']=response.status
                    if response.status!=200:
                        raise RuntimeError(f'HTTP {response.status}: {body[:2000]}')
                    payload=json.loads(body)
                    if not isinstance(payload.get('text'),str):
                        raise ValueError('响应缺少 text 字符串')
                    record.update(success=True,text=payload['text'])
            except Exception as exc:
                record.update(success=False,error=f'{type(exc).__name__}: {exc}')
            end=time.perf_counter()
            record.update(latency_s=end-start,end_offset_s=end-origin)
            return record

        sem=asyncio.Semaphore(args.warmup_concurrency)
        async def warm_request(i,item):
            async with sem:
                return await request(item,-i-1,time.perf_counter())
        warmed=await asyncio.gather(*(warm_request(i,x) for i,x in enumerate(warm)))
        save_json(out/'warmup.json',warmed)
        if any(not x['success'] for x in warmed):
            raise RuntimeError('预热失败；见 warmup.json')

        results,lag=[],[]
        iterator=iter(enumerate(items))
        active=peak=0
        integral=0.
        origin=last_change=time.perf_counter()
        def change(delta):
            nonlocal active,peak,integral,last_change
            now=time.perf_counter()
            integral+=active*(now-last_change)
            active+=delta
            peak=max(peak,active)
            last_change=now
        # 固定数量 worker 闭环发压：一个请求完成后才发送下一个。
        with (out/'requests.jsonl').open('w') as f:
            async def worker():
                for index,item in iterator:
                    change(1)
                    try:
                        result=await request(item,index,origin)
                    finally:
                        change(-1)
                    results.append(result)
                    f.write(json.dumps(result,ensure_ascii=False)+'\n')
                    f.flush()
            async def monitor():
                last=origin
                while True:
                    t=time.perf_counter()
                    await asyncio.sleep(.05)
                    now=time.perf_counter()
                    lag.append(max(0.,now-t-.05))
                    if now-last>=10:
                        print(json.dumps(dict(done=len(results),active=active,elapsed_s=now-origin)),flush=True)
                        last=now
            monitoring=asyncio.create_task(monitor())
            try:
                await asyncio.gather(*(worker() for _ in range(min(args.concurrency,len(items)))))
                elapsed=time.perf_counter()-origin
            finally:
                monitoring.cancel()
                await asyncio.gather(monitoring,return_exceptions=True)
    good=[r for r in results if r['success']]
    chars=errors=compared=0
    for r in good:
        if r['reference'] is not None:
            ref,hyp=normalize_text(r['reference']),normalize_text(r['text'])
            chars+=len(ref)
            errors+=edit_distance(ref,hyp)
            compared+=1
    summary=dict(requests=len(results),success=len(good),failed=len(results)-len(good),elapsed_s=elapsed,
        successful_requests_per_s=len(good)/elapsed,
        successful_audio_s_per_wall_s=sum(r['duration_s'] for r in good)/elapsed,
        request_latency_s=distribution([r['latency_s'] for r in good]),
        request_rtf=distribution([r['latency_s']/r['duration_s'] for r in good]),
        audio_duration_s=distribution([r['duration_s'] for r in results]),
        client_in_flight_peak=peak,client_in_flight_time_mean=integral/elapsed,
        client_event_loop_lag_s=distribution(lag),unique_measured_pcm=len({r['pcm_sha256'] for r in results}),
        cer=errors/chars if chars else None,cer_edit_distance=errors,cer_reference_chars=chars,
        cer_compared_successful_requests=compared,actual_warmup_requests=len(warm),
        cer_normalization='NFKC+lower; remove Unicode punctuation, symbols and whitespace',
        load_model='closed_loop_fixed_concurrency',retries=0,
        cache_counters_available=False,cache_disabled_verified=None,
        cache_evidence='unique PCM; disjoint warmup; backend cache counters unavailable',
        valid=len(good)==args.requests)
    save_json(out/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    return 0 if summary['valid'] else 2


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--base-url',default='http://127.0.0.1:8000')
    p.add_argument('--model',default='custom')
    p.add_argument('--language',default='zh')
    p.add_argument('--requests',type=int,default=2000)
    p.add_argument('--concurrency',type=int,default=32)
    p.add_argument('--warmup',type=int,default=32)
    p.add_argument('--warmup-concurrency',type=int,default=8)
    p.add_argument('--seed',type=int,default=20261112)
    p.add_argument('--timeout',type=float,default=600)
    args=p.parse_args()
    if min(args.requests,args.concurrency,args.warmup_concurrency,args.timeout)<=0 or args.warmup<0:
        p.error('请求数、并发、超时必须为正；预热数非负')
    return args


if __name__=='__main__':
    raise SystemExit(asyncio.run(run(parse_args())))
