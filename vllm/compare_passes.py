#!/usr/bin/env python3
"""比较输入完全相同的两组 ASR 结果，包含配对 CER bootstrap 区间。

依赖 Python>=3.10 标准库；输入是 benchmark.py 生成的结果目录。
运行：python /路径/compare_passes.py --baseline /基线/short --candidate /候选/short
 --output ./compare.json --bootstrap 2000。相对路径相对于调用目录。
参数 bootstrap 默认 2000 次；输出吞吐变化百分比、延迟、CER 差值（百分点）、
逐条文本变化和按整条音频重采样的 95% 区间。拒绝无效缓存/失败请求/不匹配输入。
拼接长音频共享短句，区间不代表独立真实业务总体；复用只需更换两个目录和输出。
"""
import argparse
import json
from pathlib import Path
import random
import statistics
import unicodedata


def normalized(s):
    s = unicodedata.normalize('NFKC', s).lower()
    return ''.join(c for c in s if not c.isspace() and not unicodedata.category(c).startswith(('P', 'S')))


def distance(a, b):
    prev = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(cur[-1]+1, prev[j]+1, prev[j-1]+(x != y)))
        prev = cur
    return prev[-1]


def read(root):
    summary = json.loads((root/'summary.json').read_text())
    rows = [json.loads(x) for x in (root/'requests.jsonl').read_text().splitlines()]
    records = {x['id']: x for x in rows}
    if len(rows) != len(records):
        raise ValueError('Duplicate IDs in measured requests')
    if not summary['valid'] or not (summary.get('cache_disabled_verified') or summary.get('prefix_only_verified')) or not all(x['success'] for x in rows):
        raise ValueError('Invalid run / failures / unverified caches')
    return summary, records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--bootstrap', type=int, default=2000)
    args = p.parse_args()
    a, ar = read(args.baseline)
    b, br = read(args.candidate)
    if set(ar) != set(br):
        raise ValueError('Measured audio IDs differ')
    deltas, lengths, changes = [], [], []
    exact = normal = 0
    ae = be = 0
    for key in sorted(ar):
        x, y = ar[key], br[key]
        if (x['index'], x['pcm_sha256'], x['reference']) != (y['index'], y['pcm_sha256'], y['reference']):
            raise ValueError('Input order/content/reference differs: '+key)
        ref, hypa, hypb = normalized(x['reference']), normalized(x['text']), normalized(y['text'])
        ea, eb = distance(ref, hypa), distance(ref, hypb)
        ae += ea
        be += eb
        deltas.append(eb-ea)
        lengths.append(len(ref))
        exact += x['text'] == y['text']
        normal += hypa == hypb
        if hypa != hypb:
            changes.append({'id': key, 'reference': x['reference'], 'baseline': x['text'],
                            'candidate': y['text'], 'baseline_errors': ea, 'candidate_errors': eb,
                            'error_delta': eb-ea})
    if ae != a['cer_edit_distance'] or be != b['cer_edit_distance']:
        raise ValueError('Recomputed CER does not match benchmark summary')
    rng = random.Random(20261112)
    n = len(deltas)
    pairs = list(zip(deltas, lengths))
    draws = []
    for _ in range(args.bootstrap):
        sample = rng.choices(pairs, k=n)
        draws.append(sum(x[0] for x in sample)/sum(x[1] for x in sample))
    draws.sort()
    lo, hi = draws[int(.025*len(draws))], draws[min(len(draws)-1, int(.975*len(draws))) ]
    ci = [lo, hi]
    changes.sort(key=lambda x: x['error_delta'], reverse=True)
    result = {
        'baseline': str(args.baseline), 'candidate': str(args.candidate), 'paired_requests': n,
        'baseline_elapsed_s': a['elapsed_s'], 'candidate_elapsed_s': b['elapsed_s'],
        'throughput_gain_percent': (b['successful_requests_per_s']/a['successful_requests_per_s']-1)*100,
        'elapsed_reduction_percent': (1-b['elapsed_s']/a['elapsed_s'])*100,
        'baseline_latency_s': a['request_latency_s'], 'candidate_latency_s': b['request_latency_s'],
        'baseline_cer': a['cer'], 'candidate_cer': b['cer'],
        'cer_delta_percentage_points': (b['cer']-a['cer'])*100,
        'paired_cer_delta_ci95_percentage_points': [v*100 for v in ci],
        'bootstrap_unit': 'complete uploaded audio; constituent utterances can repeat across composites',
        'bootstrap_resamples': args.bootstrap,
        'exact_text_equal': exact, 'normalized_text_equal': normal,
        'baseline_server_generation_tokens': a['generation_tokens_server_delta'],
        'candidate_server_generation_tokens': b['generation_tokens_server_delta'],
        'baseline_cache_config': a['cache_config'],
        'candidate_cache_config': b['cache_config'],
        'baseline_prefix_hit_rate': a.get('prefix_cache_hit_rate'),
        'candidate_prefix_hit_rate': b.get('prefix_cache_hit_rate'),
        'candidate_p95_change_percent': (b['request_latency_s']['p95']/a['request_latency_s']['p95']-1)*100,
        'screening': {'throughput_gain_at_least_5_percent': b['successful_requests_per_s'] >= 1.05*a['successful_requests_per_s'],
                      'p95_no_more_than_5_percent_worse': b['request_latency_s']['p95'] <= 1.05*a['request_latency_s']['p95'],
                      'cer_delta_upper_ci_at_most_0_1_percentage_point': hi <= .001},
        'normalized_output_changes': changes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='normalized_output_changes'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
