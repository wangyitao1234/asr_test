#!/usr/bin/env python3
"""启动独立合批服务，验证健康后压测，最后关闭本脚本创建的进程组。
依赖：serve.py/benchmark.py 同目录及其容器软件环境；numpy-dir 指向隔离包目录。
任意目录：python /实验/run_private.py --numpy-dir /实验/deps_numpy125 \
 --model-path /模型目录 --manifest /语料/manifest.jsonl --output /结果/private1 \
 --gpu 4 --port 8005 --batch-audio-s 256 --requests 64 --concurrency 8 --warmup 8
参数：batch-size=16，threads=4，wait-ms=5毫秒，batch-audio-s=120音频秒；
requests=2000，concurrency=32，warmup=32；output 必须不存在。
可用 --wait-for /前序测试/summary.json 串行衔接，等候上限1200秒。
输出：launch.json（源码哈希、命令、PID、环境）、服务日志及 bench/ 全量结果。
复用：更换 GPU、模型、语料、批量/音频预算和负载；不要占用已有服务端口。
只关闭本脚本启动的服务；异常也执行清理，不修改任何安装包。
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('numpy-dir','model-path','manifest','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gpu',default='4')
    p.add_argument('--port',type=int,default=8005)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--batch-audio-s',type=float,default=120.)
    p.add_argument('--wait-ms',type=float,default=5.)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--requests',type=int,default=2000)
    p.add_argument('--concurrency',type=int,default=32)
    p.add_argument('--warmup',type=int,default=32)
    p.add_argument('--wait-for',type=Path)
    a=p.parse_args()
    if a.wait_for:
        start=last=time.monotonic()
        while not a.wait_for.exists():
            if time.monotonic()-start>1200:
                raise TimeoutError(a.wait_for)
            if time.monotonic()-last>30:
                print('等待前序测试结束',flush=True)
                last=time.monotonic()
            time.sleep(2)
        if not json.loads(a.wait_for.read_text())['valid']:
            raise RuntimeError('前序基准失败')
    # 避免健康检查误连已经存在的服务，先检查本地端口是否占用。
    import socket
    with socket.socket() as s:
        s.bind(('127.0.0.1',a.port))
    a.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parent
    env=dict(os.environ,HIP_VISIBLE_DEVICES=a.gpu,
             PYTHONPATH=str(a.numpy_dir.resolve())+os.pathsep+os.environ.get('PYTHONPATH',''))
    cmd=[sys.executable,str(root/'serve.py'),'--mode','batch','--port',str(a.port),
         '--model-path',str(a.model_path),'--batch-size',str(a.batch_size),
         '--batch-audio-s',str(a.batch_audio_s),'--wait-ms',str(a.wait_ms),'--threads',str(a.threads)]
    with (a.output/'server.log').open('w') as log:
        proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            evidence=dict(command=cmd,server_pid=proc.pid,gpu=a.gpu,pythonpath=env['PYTHONPATH'],
                source_sha256={n:hashlib.sha256((root/n).read_bytes()).hexdigest()
                               for n in ('serve.py','benchmark.py','corpus.py')})
            (a.output/'launch.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2))
            start=time.monotonic()
            while True:
                if proc.poll() is not None:
                    raise RuntimeError('服务启动退出；检查 server.log')
                if time.monotonic()-start>180:
                    raise TimeoutError('服务启动超时')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{a.port}/health',timeout=2) as response:
                        health=json.load(response)
                    if health.get('status')=='ok':
                        break
                except OSError:
                    pass
                time.sleep(1)
            command=[sys.executable,str(root/'benchmark.py'),'--manifest',str(a.manifest),
                '--output',str(a.output/'bench'),'--base-url',f'http://127.0.0.1:{a.port}',
                '--requests',str(a.requests),'--concurrency',str(a.concurrency),
                '--warmup',str(a.warmup)]
            subprocess.run(command,check=True)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL)
                    proc.wait()


if __name__=='__main__':
    main()
