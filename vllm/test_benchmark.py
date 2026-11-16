"""复用现有 benchmark 的 10 项合约检查；使用本机模拟 HTTP 服务，无需模型/GPU。

依赖 Python>=3.10、aiohttp、numpy、soundfile 与同目录 benchmark.py。
运行：python -m unittest discover -s /脚本目录 -p test_benchmark.py。
无自定义参数；临时输入和输出由 TemporaryDirectory 管理，结果由 unittest 打印。
覆盖预热、API 覆盖补充、唯一音频、缓存验证和计时/CER 口径。复用新 benchmark
时保持同目录导入接口；这些检查不是模型精度或 GPU 性能验证。
"""
import argparse
import asyncio
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from aiohttp import web
import numpy as np
import soundfile as sf
import benchmark as b


class BenchmarkContract(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        rows = []
        for i in range(16):
            path = self.root / f'{i}.wav'
            sf.write(path, np.full(1600, (i+1)/64, dtype='float32'), 16000, subtype='PCM_16')
            rows.append({'id': str(i), 'path': str(path), 'text': '你好'})
        self.manifest = self.root/'manifest.jsonl'
        self.manifest.write_text(''.join(json.dumps(x)+'\n' for x in rows))
        self.active = self.peak = self.calls = 0
        self.cached = False
        self.fail = False
        self.hits = 0
        self.mm_size = 4
        self.connection_close = []
        self.worker_log = None
        self.prefix_hits = 0
        async def info(r):
            return web.json_response({'vllm_config': {
                'cache_config': {'enable_prefix_caching': self.cached},
                'model_config': {'multimodal_config': {'mm_processor_cache_gb': self.mm_size if self.cached else 0}}}})
        async def metrics(r):
            return web.Response(text=f'vllm:prefix_cache_hits_total{{engine="0"}} {self.prefix_hits}\n'
                f'vllm:prefix_cache_queries_total{{engine="0"}} {self.calls*100}\n'
                f'vllm:mm_cache_queries_total{{engine="0"}} 0\n'
                f'vllm:mm_cache_hits_total{{engine="0"}} {self.hits}\n'
                f'vllm:generation_tokens_total{{engine="0"}} {self.calls*2}\n')
        async def transcribe(r):
            self.connection_close.append(r.headers.get('Connection'))
            data = await r.post()
            self.assertIn('file', data)
            self.calls += 1
            if self.worker_log:
                with self.worker_log.open('a') as log:
                    log.write(f'(ApiServer_{0 if self.calls <= 2 else 1} pid=100) POST /v1/audio/transcriptions HTTP/1.1 200 OK\n')
            if self.cached and self.mm_size == 0:
                self.prefix_hits += 32
            self.active += 1
            self.peak = max(self.peak, self.active)
            await asyncio.sleep(.02)
            self.active -= 1
            return web.json_response({'text': '你好。'}, status=500 if self.fail else 200)
        app = web.Application()
        app.add_routes([web.get('/server_info', info), web.get('/metrics', metrics),
                        web.post('/v1/audio/transcriptions', transcribe)])
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, '127.0.0.1', 0)
        await site.start()
        self.url = f'http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}'

    async def asyncTearDown(self):
        await self.runner.cleanup()
        self.tmp.cleanup()

    def args(self, name, **kw):
        values = dict(manifest=self.manifest, output=self.root/name, base_url=self.url,
                      model='test', language='zh', concurrency=3, requests=10,
                      warmup=2, seed=42, timeout=2, api_key='dummy', cache_mode='disabled', allow_repeat=False)
        values.update(kw)
        return argparse.Namespace(**values)

    async def test_closed_loop_results_and_warmup_isolation(self):
        args = self.args('good')
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(await b.run(args), 0)
        s = json.loads((args.output/'summary.json').read_text())
        self.assertEqual(self.calls, 12)
        self.assertEqual(self.connection_close[:2], ['close', 'close'])
        self.assertTrue(all(x != 'close' for x in self.connection_close[2:]))
        self.assertEqual(self.peak, 3)
        self.assertEqual(s['client_in_flight_peak'], 3)
        self.assertEqual(s['unique_measured_pcm'], 10)
        self.assertEqual(s['generation_tokens_server_delta'], 20)  # warmup excluded
        self.assertEqual(s['cer'], 0)
        self.assertTrue(s['valid'])
        rows = [json.loads(x) for x in (args.output/'requests.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 10)
        self.assertEqual({r['text'] for r in rows}, {'你好。'})
        self.assertLess(min(r['start_offset_s'] for r in rows if r['index'] == 3),
                        max(r['end_offset_s'] for r in rows if r['index'] < 3))

    async def test_adaptive_warmup_requires_all_api_workers(self):
        self.worker_log = self.root/'server.log'
        self.worker_log.write_text('')
        args = self.args('coverage', warmup_server_log=self.worker_log, expected_api_workers=2)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(await b.run(args),0)
        coverage=json.loads((args.output/'warmup_workers.json').read_text())
        self.assertTrue(coverage['verified'])
        self.assertEqual(coverage['warmup_requests'],4)
        summary=json.loads((args.output/'summary.json').read_text())
        self.assertTrue(summary['warmup_api_workers_verified'])
        self.assertEqual(summary['generation_tokens_server_delta'],20)

    async def test_missing_worker_refuses_measurement(self):
        self.worker_log = self.root/'server.log'
        self.worker_log.write_text('')
        args = self.args('missing_worker', warmup=1, warmup_server_log=self.worker_log, expected_api_workers=4)
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,'refusing measurement'):
                await b.run(args)
        self.assertEqual(self.calls,16)
        self.assertTrue(all(x=='close' for x in self.connection_close))
        self.assertFalse((args.output/'summary.json').exists())

    async def test_parallel_warmup_preserves_serial_measurement(self):
        """预热可并行，但正式 concurrency=1 不能被连接池容量意外扩大。"""
        args=self.args('serial',concurrency=1,warmup=4,warmup_concurrency=4)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(await b.run(args),0)
        summary=json.loads((args.output/'summary.json').read_text())
        self.assertEqual(self.peak,4)
        self.assertEqual(summary['client_in_flight_peak'],1)
        self.assertEqual(summary['generation_tokens_server_delta'],20)
        rows=[json.loads(x) for x in (args.output/'requests.jsonl').read_text().splitlines()]
        rows.sort(key=lambda r:r['index'])
        self.assertTrue(all(y['start_offset_s']>=x['end_offset_s'] for x,y in zip(rows,rows[1:])))

    async def test_cache_guard_before_any_request(self):
        self.cached = True
        with self.assertRaisesRegex(ValueError, 'Cache guard failed'):
            await b.run(self.args('cache'))
        self.assertEqual(self.calls, 0)

    async def test_prefix_only_accepts_prefix_hits_but_keeps_mm_disabled(self):
        self.cached = True
        self.mm_size = 0
        args = self.args('prefix', cache_mode='prefix-only')
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(await b.run(args), 0)
        summary = json.loads((args.output/'summary.json').read_text())
        self.assertTrue(summary['prefix_only_verified'])
        self.assertGreater(summary['prefix_cache_hit_rate'], 0)
        self.assertFalse(summary['cache_disabled_verified'])
        self.assertTrue(summary['valid'])

    async def test_prefix_only_rejects_mm_cache(self):
        self.cached = True
        with self.assertRaisesRegex(ValueError, 'Prefix-only cache guard'):
            await b.run(self.args('prefix_bad', cache_mode='prefix-only'))
        self.assertEqual(self.calls, 0)

    async def test_failure_is_not_retried(self):
        self.fail = True
        args = self.args('failure', warmup=0)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(await b.run(args), 2)
        s = json.loads((args.output/'summary.json').read_text())
        self.assertEqual(self.calls, 10)
        self.assertEqual(s['failed'], 10)
        self.assertFalse(s['valid'])

    async def test_duplicates_detected_after_decode(self):
        rows = [json.loads(x) for x in self.manifest.read_text().splitlines()]
        # A second WAV filename with identical samples must still be rejected.
        samples, _ = sf.read(rows[0]['path'])
        sf.write(rows[1]['path'], samples, 16000, subtype='FLOAT')
        with self.assertRaisesRegex(ValueError, 'Duplicate decoded audio'):
            b.load_audio(self.manifest, 16, 0, 42, False)
        with self.assertRaisesRegex(ValueError, 'Refusing implicit reuse'):
            b.load_audio(self.manifest, 17, 0, 42, False)

class BatchHookContract(unittest.TestCase):
    @unittest.skipUnless(Path('/proc/self/cmdline').exists(), 'Linux live-process check')
    def test_launch_fallback_checks_live_process_and_port(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'launch.json'
            command = [sys.executable, '-c', 'import time; time.sleep(10)',
                       '--port', '8001', '--mm-processor-cache-gb', '0', '--no-enable-prefix-caching']
            process = subprocess.Popen(command)
            try:
                path.write_text(json.dumps({'command': command, 'server_pid': process.pid}))
                cfg = b.launch_cache_config(path, 'http://127.0.0.1:8001')
                self.assertEqual(cfg, {'enable_prefix_caching': False, 'mm_processor_cache_gb': 0})
                with self.assertRaisesRegex(ValueError, 'port differs'):
                    b.launch_cache_config(path, 'http://127.0.0.1:8002')
                path.write_text(json.dumps({'command': command + ['--fake'], 'server_pid': process.pid}))
                with self.assertRaisesRegex(ValueError, 'live server command'):
                    b.launch_cache_config(path, 'http://127.0.0.1:8001')
            finally:
                process.terminate()
                process.wait(timeout=3)



if __name__ == '__main__':
    unittest.main()
