# ASR 模拟压测系统

面向 Qwen3-ASR/vLLM 与 Paraformer/FunASR 的 HTTP 非流式转写性能测试。支持固定请求数、闭环并发、独立预热、PCM 去重、逐请求记录、CER 和 A/B 对比。

“模拟”指模拟并发请求；短音频使用公开真实录音或你自己的语料，不靠重复上传同一文件制造吞吐。默认2000正式请求、32条独立预热，需要至少2032条不同音频。

## 目录

| 目录 | 内容 |
|---|---|
| `vllm/` | vLLM 客户端、缓存配置/命中验证、配对比较、模拟服务契约测试 |
| `funasr/` | FunASR 客户端、串行压测矩阵、配对比较、Paraformer 动态合批实验服务及生命周期启动器 |
| `corpus/` | 固定版本 AISHELL 下载转换、长音频拼接工具 |
| `docs/measurement.md` | 计时、CER、缓存和公平比较口径 |

## 安装

Python 3.10+。客户端不要求 GPU，也不要求安装模型框架。

```bash
git clone https://github.com/wangyitao1234/asr_test.git
cd asr_test
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-corpus.txt
```

只使用已有语料时，安装 `requirements.txt` 即可。模型服务需在其兼容环境单独启动；不要通过客户端依赖文件替换厂商 PyTorch。部分服务的 PyTorch 与 NumPy2 ABI 不兼容，需在服务环境单独验证或隔离安装 `numpy<2`，详见 [FunASR 说明](funasr/README.md)。

## 准备语料

```bash
python corpus/prepare_corpus.py --download \
  --parquet data/aishell.parquet --output data/corpus_v1
```

固定来源为 [AudioBench AISHELL-1 测试集](https://huggingface.co/datasets/AudioLLMs/aishell_1_zh_test)，revision `e73dc7f096ecea5c8e5c722446c8d3b4014fdd18` 的第0分片。工具校验源 SHA256，转换得到2307条不同录音。数据集使用条件以来源为准；仓库不包含音频文件。

已有语料也可提供 JSONL 清单：

```json
{"id":"audio-0001","path":"audio/0001.wav","text":"参考转写文本"}
```

每行一个对象，`id` 唯一；路径可以绝对，也可相对清单目录。要求非空、有限值、单声道16kHz音频。`text` 缺失时只能测性能，CER 无法计算；A/B 精度比较须提供参考文本。可选 `sha256`、`pcm_sha256` 用于额外校验。

## 测试已启动的服务

下面命令均在仓库根目录运行。每次 `--output` 使用新目录，客户端不重试失败请求。

### Qwen3-ASR / vLLM

```bash
python vllm/benchmark.py \
  --manifest data/corpus_v1/manifest.jsonl \
  --base-url http://127.0.0.1:8001 --model /path/to/Qwen3-ASR-1.7B \
  --requests 2000 --concurrency 1000 --warmup 32 \
  --cache-mode disabled --output results/qwen_run1
```

`--model` 替换为 `/v1/models` 中实际模型标识。默认校验 `/server_info?config_format=json` 和 `/metrics`，要求 prefix caching关闭、MM processor cache为0、相关命中增量为0。路由不开放的本地专用服务可传 `--launch-snapshot`，格式及启动示例见 [vLLM 说明](vllm/README.md)。要测前缀复用，使用显式的 `--cache-mode prefix-only`；`observe` 仅观察，不能标为无缓存验证通过。

### Paraformer / FunASR

```bash
python funasr/benchmark.py \
  --manifest data/corpus_v1/manifest.jsonl \
  --base-url http://127.0.0.1:8000 --model custom \
  --requests 2000 --concurrency 32 --warmup 32 \
  --output results/paraformer_run1
```

目标需提供 `/health`、`/v1/models` 和 `/v1/audio/transcriptions`。FunASR 以 `--model-path` 启动时使用 `custom`，避免误触其他模型别名。此后端没有缓存计数器，客户端如实记录 `cache_disabled_verified=null`；不伪装成零命中。

把 `--concurrency` 改为1即可测单并发。比较不同模型时还应固定 `--warmup`、`--seed`（默认20261112）、`--requests` 和 manifest，确保同一批音频。不能把一次64条与另一次2000条的平均值直接当公平对照。

## 结果与对比

`summary.json` 包含总耗时、成功/失败、吞吐、延迟p50/p95/p99、RTF、CER、实际在途峰值与均值、客户端事件循环延迟、PCM唯一数和缓存证据。`requests.jsonl` 保存完整文本、参考、状态、音频哈希与逐请求时间；`input_order.json`、`warmup.json` 保存输入顺序和独立预热。

```bash
python vllm/compare_passes.py \
  --baseline results/qwen_A --candidate results/qwen_B --output results/qwen_AB.json
python funasr/compare.py \
  --baseline results/paraformer_A --candidate results/paraformer_B \
  --output results/paraformer_AB.json
```

vLLM 比较要求两次缓存均有验证证据，并输出配对 CER bootstrap 区间；FunASR 比较核对完整输入与成功状态，默认不允许 CER 增加。默认归一化：NFKC+小写，移除Unicode标点、符号和空白，不做简繁或中文数字转换。

## 可选长音频压力集

```bash
python corpus/make_composites.py --manifest data/corpus_v1/manifest.jsonl \
  --output data/corpus_composite60 --count 2032 --target-seconds 60
```

不同拼接文件会复用源句；这是时长压力集，不是独立自然长录音。长音频2000请求会预加载约4GB上传内容，需准备足够客户端内存。先完成语料准备，再开始任何计时。

## 验证

```bash
python -m unittest discover -s vllm -p 'test_*.py' -v
python -m unittest discover -s funasr -p 'test_*.py' -v
```

两个命令分别启动进程，避免同名 `benchmark` 模块相互影响。测试使用本地模拟HTTP服务，不需要模型/GPU；不能将模拟吞吐当作硬件性能。模型服务端的运行要求与覆盖范围见各子目录说明。

本系统采用有限请求的闭环发压，有自然收尾阶段。并发是最大客户端在途数，既不是实际模型batch，也不是固定到达率；2000请求的结果不能代替长时间持续压测。完整口径见 [measurement.md](docs/measurement.md)。
