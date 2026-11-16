# vLLM / Qwen3-ASR 客户端

依赖与命令从仓库根目录运行，见根README。服务须支持音频转写接口，vLLM具体版本和Qwen3-ASR支持取决于你部署的后端。本工具不安装/替换vLLM或厂商算子。

启动专用服务的示例（按实际模型目录、设备环境与vLLM版本调整）：

```bash
HIP_VISIBLE_DEVICES=4 vllm serve /path/to/Qwen3-ASR-1.7B \
  --trust-remote-code --port 8001 \
  --no-enable-prefix-caching --mm-processor-cache-gb 0 \
  --max-num-seqs 1024 --max-num-batched-tokens 2048
```

CUDA设备部署应使用其对应的 `CUDA_VISIBLE_DEVICES`。是否支持这些参数，以实际 `vllm serve --help` 为准。压测已有服务时不要启动第二份服务争用GPU。

## 配置路由与启动快照

严格测试先读取 `/server_info?config_format=json`。一些vLLM部署不暴露该路由，返回404时可传 `--launch-snapshot /path/to/launch.json`。快照格式：

```json
{
  "server_pid": 12345,
  "command": [
    "vllm", "serve", "/path/to/Qwen3-ASR-1.7B",
    "--trust-remote-code", "--port", "8001",
    "--no-enable-prefix-caching", "--mm-processor-cache-gb", "0",
    "--max-num-seqs", "1024", "--max-num-batched-tokens", "2048"
  ]
}
```

将PID换成实际服务进程，command逐项保持与真实启动参数一致。客户端必须在同一Linux PID命名空间执行，目标URL为localhost或127.0.0.1；会核对 `/proc/PID/cmdline` 的参数尾部、目标端口和显式缓存选项。手写不匹配的快照不会绕过验证。容器外的客户端不能使用容器内PID快照。

无论配置来源如何，严格模式仍要求 `/metrics` 前后计数器存在、未重置且命中符合策略。服务必须专用，不能混入其他客户端请求。

## 前缀复用实验

将服务启动选项改为 `--enable-prefix-caching --mm-processor-cache-gb 0`，并使用 `--cache-mode prefix-only`，快照也须匹配。客户端允许前缀命中，但要求MM缓存queries/hits为0且prefix queries大于0。`--cache-mode observe` 放宽命中约束，只用于明确的缓存观察；不会验证为无缓存通过，严格比较脚本也会拒绝它。

`--api-key` 默认dummy，鉴权服务可显式传入；保存配置会排除api-key字段。不要把真实密钥或含密钥的命令历史提交到仓库。

## 预热与测试

默认预热32条、并发最多8。单并发正式测试仍可独立设置 `--warmup-concurrency 8`。多API进程的专用服务可传 `--warmup-server-log` 和 `--expected-api-workers` 来检查日志覆盖；依赖日志中的 `ApiServer_N` 格式，不适用任意后端。

```bash
python -m unittest discover -s vllm -p 'test_*.py' -v
python vllm/benchmark.py --help
python vllm/compare_passes.py --help
```

`compare_passes.py` 仅用于完整、成功且缓存验证通过的同输入配对，默认bootstrap2000次；输出吞吐变化、延迟、CER差值及文本变化。它提供筛选指标，不自动部署候选服务。
