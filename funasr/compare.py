#!/usr/bin/env python3
"""比较同一批音频的原始/优化 HTTP 结果，防止吞吐提升掩盖错误或输入不一致。
依赖 Python>=3.10、同目录 corpus.py 的 CER 函数及其 numpy/soundfile 依赖。
任意目录：python /实验/compare.py --baseline /结果/A --candidate /结果/B \
 --output /结果/AB.json --max-cer-increase 0.001
输入为 benchmark.py 输出目录；逐条核对 index/id/PCM SHA/参考文本/成功状态。
输出包含吞吐倍数、CER 绝对变化（0.001=0.1个百分点）、文本差异和完整性。
阈值默认0（不得退化）；不同阈值须由使用者明确指定，不会将差异自动视为正确。
复用时更换 A/B 路径、输出和允许 CER 变化；不要求使用完全相同的并发，
但性能公平比较应在同机器/设备、同并发、无并行负载下运行，并独立复测。
"""
import argparse
import json
from pathlib import Path
from corpus import normalize_text, save_json


def compare(a,b,max_increase):
    def read(p):
        rows=[json.loads(s) for s in (p/'requests.jsonl').read_text().splitlines()]
        result={r['index']:r for r in rows}
        if len(result)!=len(rows):
            raise ValueError('结果含重复 index')
        return result,json.loads((p/'summary.json').read_text())
    ar,sa=read(a)
    br,sb=read(b)
    if ar.keys()!=br.keys() or not sa['valid'] or not sb['valid']:
        raise ValueError('输入索引不一致或运行失败')
    changed=[]
    exact=0
    for i,x in ar.items():
        y=br[i]
        if any(x[k]!=y[k] for k in ('id','pcm_sha256','reference')) or not x['success'] or not y['success']:
            raise ValueError(f'输入不匹配/请求失败：{i}')
        if x['text']==y['text']:
            exact+=1
        if normalize_text(x['text'])!=normalize_text(y['text']):
            changed.append(dict(index=i,id=x['id'],reference=x['reference'],
                                baseline=x['text'],candidate=y['text']))
    delta=sb['cer']-sa['cer']
    return dict(requests=len(ar),raw_text_identical=exact,normalized_text_changed=len(changed),
                baseline_rps=sa['successful_requests_per_s'],candidate_rps=sb['successful_requests_per_s'],
                speedup=sb['successful_requests_per_s']/sa['successful_requests_per_s'],
                baseline_cer=sa['cer'],candidate_cer=sb['cer'],cer_absolute_change=delta,
                max_cer_increase=max_increase,accuracy_threshold_passed=delta<=max_increase,
                differences=changed)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-cer-increase',type=float,default=0.)
    a=p.parse_args()
    result=compare(a.baseline,a.candidate,a.max_cer_increase)
    if a.output.exists():
        raise FileExistsError(a.output)
    save_json(a.output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='differences'},ensure_ascii=False,indent=2))
    return 0 if result['accuracy_threshold_passed'] else 2


if __name__=='__main__':
    raise SystemExit(main())
