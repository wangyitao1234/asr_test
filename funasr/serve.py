#!/usr/bin/env python3
"""Paraformer 独立实验服务：原始服务对照或单执行器动态合批。

依赖：容器 FunASR 1.4.26、PyTorch/HIP、FastAPI、uvicorn、numpy<2、soundfile。
准备：python3 -m pip install --target /实验/deps_numpy125 --no-deps numpy==1.25.0
任意目录：HIP_VISIBLE_DEVICES=4 PYTHONPATH=/实验/deps_numpy125 python3 /实验/serve.py \
 --model-path /模型目录 --port 8002 --mode original
优化：同上改为 --mode batch --port 8003 --batch-size 16 --threads 4。
参数：batch-size=16 是合批上限；wait-ms=5 为收集窗口毫秒；batch-audio-s=120
限制单批音频总秒数（单条超预算仍单独处理）；threads 控制 torch CPU 线程数，
默认不改。queue-size=2048 为排队上限，满则503，不重试；max-audio-s=300。
可选 --retain-allocator：跳过 AutoModel 每次推理末尾的 empty_cache()，保留
PyTorch GPU 内存池（不是音频/识别结果缓存）；默认保留原行为，便于独立消融。
输出：stdout 每批推理耗时、实际批量、音频秒数；/health 含批处理统计。
复用：换模型路径、设备可见性、端口、批量/等待窗口和线程；基准模式保留
安装包 create_app 原始实现。仅 batch 模式实现 custom、spk=false 的转写接口，
不提供 VAD、标点、说话人和 /asr；不改变精度，不做结果缓存，不用多线程共享模型。
输出适配复用 FunASR 的文本清理及 verbose_json 逻辑。模型输入为解码后的
float32 波形数组；支持声道合并、重采样。max-audio-s 防止超长请求占满内存。
"""
import argparse
import asyncio
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import io
import json
import re
import time


def create_batched_app(args):
    import numpy as np
    import soundfile as sf
    import torch
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import JSONResponse, PlainTextResponse
    from funasr import AutoModel
    from funasr.bin._server_app import (prepare_audio_for_inference,
        build_openai_fallback_segments, build_openai_verbose_json)

    model=AutoModel(model=args.model_path,device=args.device,hub=args.hub,
                    disable_update=True,disable_pbar=True,ncpu=args.threads or 4)
    queue=asyncio.Queue(maxsize=args.queue_size)
    executor=ThreadPoolExecutor(max_workers=1)
    stats=dict(batches=0,requests=0,max_batch=0,inference_s=0.,failed_batches=0)

    def infer(entries):
        # AutoModel 含可变 kwargs/hotword 状态，只允许一个执行器串行访问。
        start=time.perf_counter()
        empty_cache=torch.cuda.empty_cache
        # 仅独立进程的单推理执行器范围内替换；finally 恢复。模型和权重不变。
        # AutoModel.inference 尾部无条件 empty_cache，可能反复释放可复用分配。
        if args.retain_allocator:
            torch.cuda.empty_cache=lambda: None
        try:
            result=model.generate(input=[x['audio'] for x in entries],batch_size=len(entries),
                                  language=entries[0]['language'],disable_pbar=True)
        finally:
            torch.cuda.empty_cache=empty_cache
        if len(result)!=len(entries):
            raise RuntimeError(f'模型返回数量不匹配：{len(result)} != {len(entries)}')
        elapsed=time.perf_counter()-start
        print(json.dumps(dict(batch=len(entries),inference_s=elapsed,
                             audio_s=sum(x['duration'] for x in entries))),flush=True)
        return result,elapsed

    async def worker():
        pending=None
        while True:
            first=pending if pending is not None else await queue.get()
            pending=None
            entries=[first]
            seconds=first['duration']
            deadline=asyncio.get_running_loop().time()+args.wait_ms/1000
            while len(entries)<args.batch_size:
                remaining=deadline-asyncio.get_running_loop().time()
                if queue.empty() and remaining<=0:
                    break
                try:
                    item=queue.get_nowait() if not queue.empty() else await asyncio.wait_for(queue.get(),remaining)
                except asyncio.TimeoutError:
                    break
                if seconds+item['duration']>args.batch_audio_s:
                    pending=item
                    break
                entries.append(item)
                seconds+=item['duration']
            # 不同语言选项分开执行；避免把单个请求参数误用于同批其他请求。
            groups=defaultdict(list)
            for entry in entries:
                groups[entry['language']].append(entry)
            for group in groups.values():
                try:
                    outputs,elapsed=await asyncio.get_running_loop().run_in_executor(executor,infer,group)
                    stats['batches']+=1
                    stats['requests']+=len(group)
                    stats['max_batch']=max(stats['max_batch'],len(group))
                    stats['inference_s']+=elapsed
                    for item,result in zip(group,outputs):
                        if not item['future'].done():
                            item['future'].set_result(result)
                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    stats['failed_batches']+=1
                    for item in group:
                        if not item['future'].done():
                            item['future'].set_exception(RuntimeError(str(exc)))

    @asynccontextmanager
    async def lifespan(app):
        task=asyncio.create_task(worker())
        try:
            yield
        finally:
            task.cancel()
            await asyncio.gather(task,return_exceptions=True)
            executor.shutdown(wait=True)

    app=FastAPI(lifespan=lifespan)

    @app.get('/health')
    async def health():
        return dict(status='ok',device=args.device,models_loaded=['custom'],
                    queue_size=queue.qsize(),stats=stats,configuration=vars(args))

    @app.get('/v1/models')
    async def models():
        return dict(object='list',data=[dict(id='custom',object='model')])

    # 不使用 postponed annotations，保证 FastAPI 可解析局部导入的 UploadFile。
    @app.post('/v1/audio/transcriptions')
    async def transcribe(file: UploadFile=File(...), model: str=Form('custom'),
                         language: str=Form('zh'), response_format: str=Form('json'),
                         spk: bool=Form(False)):
        if model not in ('custom','whisper-1') or spk:
            raise HTTPException(400,'此实验服务仅支持 custom/whisper-1 且 spk=false')
        if response_format not in ('json','text','verbose_json'):
            raise HTTPException(400,'不支持的 response_format')
        content=await file.read()
        try:
            audio,sr=sf.read(io.BytesIO(content),dtype='float32')
            audio,sr=prepare_audio_for_inference(audio,sr)
            duration=len(audio)/sr
            if not 0<duration<=args.max_audio_s or not np.isfinite(audio).all():
                raise ValueError('音频为空、非有限或超过最大时长')
        except Exception as exc:
            raise HTTPException(400,f'音频解码失败：{exc}') from exc
        future=asyncio.get_running_loop().create_future()
        try:
            queue.put_nowait(dict(audio=audio,duration=duration,language=language,future=future))
        except asyncio.QueueFull:
            raise HTTPException(503,'推理队列已满')
        result=await future
        text=re.sub(r'<\|[^|]*\|>','',result['text']).strip()
        if response_format=='text':
            return PlainTextResponse(text)
        if response_format=='verbose_json':
            return JSONResponse(build_openai_verbose_json(dict(text=text,duration=duration,
                segments=build_openai_fallback_segments(text,duration)),requested_language=language))
        return dict(text=text)
    return app


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path',required=True)
    p.add_argument('--mode',choices=['original','batch'],default='batch')
    p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--port',type=int,default=8003)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--hub',default='ms')
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--wait-ms',type=float,default=5.)
    p.add_argument('--batch-audio-s',type=float,default=120.)
    p.add_argument('--threads',type=int)
    p.add_argument('--retain-allocator',action='store_true')
    p.add_argument('--queue-size',type=int,default=2048)
    p.add_argument('--max-audio-s',type=float,default=300.)
    a=p.parse_args()
    if min(a.batch_size,a.queue_size,a.batch_audio_s,a.max_audio_s)<=0 or a.wait_ms<0 or (a.threads is not None and a.threads<1):
        p.error('批量/队列/时长/线程必须为正，等待时间非负')
    if a.mode=='original' and (a.threads is not None or a.retain_allocator):
        p.error('原始对照保留 AutoModel 默认行为；threads/retain-allocator 仅用于 batch 模式')
    return a


def main():
    args=parse_args()
    import torch
    if args.threads:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
    print(json.dumps(dict(config=vars(args),torch=torch.__version__,threads=torch.get_num_threads())),flush=True)
    if args.mode=='original':
        from funasr.bin._server_app import create_app
        app=create_app(device=args.device,model_path=args.model_path,hub=args.hub)
    else:
        app=create_batched_app(args)
    import uvicorn
    uvicorn.run(app,host=args.host,port=args.port,access_log=False)


if __name__=='__main__':
    main()
