"""HTTP 基准契约测试，不需要 GPU；验证并发、输入隔离、错误和 CER。
依赖：Python>=3.10、aiohttp、numpy、soundfile，同目录 benchmark.py/corpus.py。
任意目录：cd /实验目录 && python3 -m unittest -v test_benchmark.py
参数/输出：unittest 标准参数与通过/失败日志；临时 WAV 和结果自动清理。
复用：增加模拟响应场景即可测试新服务协议，勿用这里的虚拟吞吐作为性能结果。
"""
import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from aiohttp import web
import numpy as np
import soundfile as sf
import benchmark


class BenchmarkContracts(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp=TemporaryDirectory()
        self.root=Path(self.temp.name)
        lines=[]
        for i in range(8):
            p=self.root/f'{i}.wav'
            sf.write(p,np.full(1600,i*.01,dtype='float32'),16000)
            lines.append(dict(id=str(i),path=str(p),text='测试'))
        self.manifest=self.root/'manifest.jsonl'
        self.manifest.write_text('\n'.join(json.dumps(x) for x in lines))
        self.active=self.peak=self.calls=0
        self.fail=False
        async def transcribe(request):
            self.active+=1
            self.peak=max(self.peak,self.active)
            await request.post()
            self.calls+=1
            await asyncio.sleep(.01)
            self.active-=1
            return web.Response(status=500,text='failure') if self.fail else web.json_response(dict(text='测 试。'))
        app=web.Application()
        app.router.add_get('/health',lambda r:web.json_response(dict(status='ok')))
        app.router.add_get('/v1/models',lambda r:web.json_response(dict(data=[dict(id='custom')])))
        app.router.add_post('/v1/audio/transcriptions',transcribe)
        self.runner=web.AppRunner(app)
        await self.runner.setup()
        site=web.TCPSite(self.runner,'127.0.0.1',0)
        await site.start()
        port=site._server.sockets[0].getsockname()[1]
        self.args=SimpleNamespace(manifest=self.manifest,requests=6,warmup=2,seed=1,
            output=self.root/'result',base_url=f'http://127.0.0.1:{port}',model='custom',
            language='zh',concurrency=3,warmup_concurrency=2,timeout=10)

    async def asyncTearDown(self):
        await self.runner.cleanup()
        self.temp.cleanup()

    async def test_closed_loop_unique_and_cer(self):
        self.assertEqual(await benchmark.run(self.args),0)
        s=json.loads((self.args.output/'summary.json').read_text())
        warm=json.loads((self.args.output/'warmup.json').read_text())
        rows=[json.loads(x) for x in (self.args.output/'requests.jsonl').read_text().splitlines()]
        self.assertEqual(self.calls,8)
        self.assertEqual(self.peak,3)
        self.assertEqual(s['cer'],0)
        self.assertEqual(s['unique_measured_pcm'],6)
        self.assertIsNone(s['cache_disabled_verified'])
        self.assertFalse({x['pcm_sha256'] for x in warm}&{x['pcm_sha256'] for x in rows})

    async def test_failed_measurement_invalid_no_retry(self):
        self.args.warmup=0
        self.fail=True
        self.assertEqual(await benchmark.run(self.args),2)
        s=json.loads((self.args.output/'summary.json').read_text())
        self.assertFalse(s['valid'])
        self.assertEqual((s['failed'],self.calls),(6,6))

    async def test_warmup_failure_aborts(self):
        self.fail=True
        with self.assertRaisesRegex(RuntimeError,'预热失败'):
            await benchmark.run(self.args)
        self.assertEqual(self.calls,2)
        self.assertFalse((self.args.output/'summary.json').exists())

    async def test_duplicate_pcm_rejected(self):
        rows=[json.loads(x) for x in self.manifest.read_text().splitlines()]
        rows[1]['path']=rows[0]['path']
        self.manifest.write_text('\n'.join(json.dumps(x) for x in rows))
        with self.assertRaisesRegex(ValueError,'Duplicate decoded audio'):
            await benchmark.run(self.args)
        self.assertEqual(self.calls,0)


if __name__=='__main__':
    unittest.main()
