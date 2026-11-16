# FunASR / Paraformer

## 测试现有服务

```bash
python funasr/benchmark.py --manifest data/corpus_v1/manifest.jsonl \
  --base-url http://127.0.0.1:8000 --model custom \
  --requests 2000 --concurrency 32 --warmup 32 --output results/funasr_A
```

`--model-path` 启动的FunASR服务一般以 `custom` 注册；以 `/v1/models` 实际返回为准。此客户端也要求 `/health`。返回顶层text字符串即为转写结果。严格PCM去重，预热与正式集合隔离；无缓存计数器，缓存验证记录为null。

## 可选动态合批实验服务

`serve.py` 是本项目的独立Paraformer实验入口，已在FunASR1.4.26、厂商HIP PyTorch2.10上验证。原始模式导入安装包 `funasr.bin._server_app.create_app`，batch模式使用本目录的队列与单执行器动态合批。其他FunASR版本应核实相关接口后测试；不提供VAD、标点、说话人或 `/asr`。

在服务专用环境安装与硬件匹配的FunASR/PyTorch、FastAPI、uvicorn、python-multipart和soundfile。客户端requirements不包含GPU服务依赖。

```bash
HIP_VISIBLE_DEVICES=4 python funasr/serve.py \
  --mode batch --host 127.0.0.1 --port 8003 \
  --model-path /path/to/paraformer --device cuda:0 \
  --threads 4 --batch-size 16 --wait-ms 5 --batch-audio-s 120
```

这里最多16条/120音频秒一批，空队列收集窗口5ms；长音频可改为 `--batch-audio-s 256`，避免约63秒音频在120秒预算下退化为batch=1。只允许一个线程进入有可变状态的AutoModel，模型返回逐条映射给请求；不做结果缓存、不改精度。

支持 `custom`/`whisper-1`、`spk=false`、json/text/verbose_json。队列满返回503；错误不自动重试。`--max-audio-s` 默认300秒；最大时长在解码后检查，本服务未实现上传字节限额，不是面向不受信任公网的完整生产网关。

原始对照改用 `--mode original --port 8002`，不传threads或retain-allocator；保留安装包默认行为。可选 `--retain-allocator` 跳过每次推理结束的 `empty_cache()`，仅用于消融，不保证收益。

如果遇到 `Numpy is not available` / `_ARRAY_API not found`，先验证服务环境的PyTorch与NumPyABI：

```bash
python -m pip install --target "$PWD/deps_numpy125" --no-deps numpy==1.25.0
PYTHONPATH="$PWD/deps_numpy125" python -c \
 'import numpy,torch; print(numpy.__version__,torch.from_numpy(numpy.zeros(1)))'
```

通过后将该PYTHONPATH显式用于服务。不要盲目全局降级或覆盖厂商环境；隔离依赖路径按部署位置设置。本项目不会复制其他容器的依赖。

## 批量对照与生命周期

对已经启动的两个服务串行发压：

```bash
python funasr/run_matrix.py --manifest data/corpus_v1/manifest.jsonl --output results \
  --case A,http://127.0.0.1:8002,2000,32 \
  --case B,http://127.0.0.1:8003,2000,32
python funasr/compare.py --baseline results/A --candidate results/B \
  --output results/AB.json
```

Linux中可由启动器管理独立服务，结束后只关闭自己创建的进程组：

```bash
python funasr/run_private.py --numpy-dir "$PWD/deps_numpy125" \
  --model-path /path/to/paraformer --manifest data/corpus_v1/manifest.jsonl \
  --gpu 4 --port 8005 --batch-size 16 --batch-audio-s 120 \
  --requests 2000 --concurrency 32 --warmup 32 --output results/private_B
```

该启动器按HIP设备选择，保存命令、源码哈希、PID及环境路径。使用CUDA需按实际平台调整设备可见性，不能直接把HIP参数当CUDA控制。`--wait-for` 可等待前序成功summary，避免同时压测。所有输出目录使用新名称。
