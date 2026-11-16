#!/usr/bin/env python3
"""串行运行 HTTP 基准矩阵，禁止本脚本自身的压测互相重叠。
依赖 Python>=3.10、同目录 benchmark.py 及其依赖，服务须提前启动。
任意目录：python /实验/run_matrix.py --manifest /语料/manifest.jsonl \
 --output /结果 --case batch16_r1,http://127.0.0.1:8003,2000,1000 \
 --case batch16_r2,http://127.0.0.1:8003,2000,1000
可加 --wait-for /正在运行的基准/summary.json，等其成功后才开始，默认上限1200秒。
case 格式为 名称,URL,请求数,并发；warmup=32、seed=20261112、timeout=600秒
传给每次 benchmark。每个 case 在 output/name 新建结果目录，不覆盖旧结果。
输出：各 case 日志与 summary，末尾 matrix.json；遇到失败立即退出。
复用：修改 case 集合、语料、预热/超时；设备/合批配置由外部服务明确设置。
等待期间每30秒报告一次，不根据部分请求结果误判基准已经结束。
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case',action='append',required=True)
    p.add_argument('--wait-for',type=Path)
    p.add_argument('--wait-timeout',type=float,default=1200)
    p.add_argument('--warmup',type=int,default=32)
    p.add_argument('--seed',type=int,default=20261112)
    p.add_argument('--timeout',type=float,default=600)
    a=p.parse_args()
    cases=[]
    for case in a.case:
        name,url,n,c=case.split(',')
        if Path(name).name!=name or name in ('.','..') or min(int(n),int(c))<1:
            p.error(f'无效 case：{case}')
        cases.append((name,url,n,c))
    a.output.mkdir(parents=True,exist_ok=True)
    if len({x[0] for x in cases})!=len(cases) or any((a.output/x[0]).exists() for x in cases):
        raise ValueError('case 名称重复或输出已存在')
    if a.wait_for:
        start=last=time.monotonic()
        while not a.wait_for.exists():
            now=time.monotonic()
            if now-start>a.wait_timeout:
                raise TimeoutError(a.wait_for)
            if now-last>30:
                print(f'等待原始基准完成：{now-start:.0f}s',flush=True)
                last=now
            time.sleep(2)
        if not json.loads(a.wait_for.read_text())['valid']:
            raise RuntimeError('前序基准无效，停止矩阵')
    results={}
    for name,url,n,c in cases:
        command=[sys.executable,str(Path(__file__).with_name('benchmark.py')),
            '--manifest',str(a.manifest),'--output',str(a.output/name),
            '--base-url',url,'--requests',n,'--concurrency',c,
            '--warmup',str(a.warmup),'--seed',str(a.seed),'--timeout',str(a.timeout)]
        print(json.dumps(dict(case=name,command=command)),flush=True)
        with (a.output/f'{name}.log').open('x') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        results[name]=json.loads((a.output/name/'summary.json').read_text())
        print(json.dumps(dict(case=name,summary=results[name])),flush=True)
    target=a.output/f'matrix_{time.time_ns()}.json'
    target.write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    main()
