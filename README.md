# LLM High-Concurrency Simulation Testing Platform

高性能仿真测试平台，持续制造"海量用户并发请求"，压测底层 vLLM 推理服务，验证系统的并行处理能力、吞吐、时延、稳定性。

## 功能特性

- **多模式压测**: 固定并发、阶梯升压、突发洪峰、长上下文、流式响应
- **基准评测**: 支持 GPQA、MMLU-Pro、MMLU-Redux、SuperGPQA 等标准评测集
- **Auto-Tuning**: 基于 Bayesian Optimization 的 vLLM 参数自动调优
- **数据集管理**: 支持导入（JSON/JSONL/CSV）和泛化生成两种模式，可指定文本字段如 `instruction`
- **全链路指标**: QPS、TPS、TTFT、TPOT、P50/P90/P99 时延
- **vLLM 集成**: 自动采集内部指标（batch size、KV Cache、GPU 利用率）
- **多厂商 GPU**: 支持 NVIDIA / 华为昇腾 / 寒武纪 / 壁仞 / Metax / 摩尔线程等国产 GPU
- **GPU 实时监控**: WebUI GPU Monitor 页面，逐卡展示利用率、显存、温度、功耗、进程（NVIDIA CUDA 优先）
- **WebUI**: Streamlit 可视化界面，支持评测、压测、调参、负载均衡管理、GPU 监控，切换标签页结果不丢失
- **报告生成**: JSON 格式输出，包含瓶颈分析

## 安装

```bash
pip install -e .
```

## 快速开始

### 1. 启动 vLLM 服务

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --tensor-parallel-size 1
```

### 2. 运行压测

```bash
# 固定并发压测
python bench.py run --vllm-host localhost --concurrency 100 --duration 60

# 阶梯升压
python bench.py run --vllm-host localhost --scenario step \
    --base 100 --increment 100 --steps 5

# 突发洪峰
python bench.py run --vllm-host localhost --scenario burst --peak 1000

# 长上下文压测
python bench.py run --vllm-host localhost --scenario long_context \
    --input-len 8192 --concurrency 50

# 流式响应压测
python bench.py run --vllm-host localhost --concurrency 50 --stream

# 使用配置文件导入 JSON 数据集，并读取 instruction 字段
python bench.py run --config config/default.yaml --vllm-host localhost --concurrency 50 --stream
```

## 基准评测

支持标准评测集评估模型知识能力：

```bash
# 查看可用评测集
python bench.py eval --list

# 运行 GPQA 评测
python bench.py eval --benchmark gpqa --vllm-host localhost --samples 100

# 运行 MMLU-Pro 指定学科
python bench.py eval --benchmark mmlu-pro --vllm-host localhost --subject math

# 使用 Chain-of-Thought 提示
python bench.py eval --benchmark gpqa --vllm-host localhost --prompt-style cot
```

| 评测集 | 说明 |
|--------|------|
| GPQA | 研究生级别问答（需 HF 登录） |
| MMLU-Pro | MMLU 增强版，10 选项 |
| MMLU-Redux | MMLU 纠错版 |
| SuperGPQA | 综合研究生级别评测 |
| C-Eval | 中文综合能力评测，52 学科 |

详细文档见 [src/eval/README.md](src/eval/README.md)。

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--vllm-host` | vLLM 服务器地址 | localhost |
| `--vllm-port` | vLLM 端口 | 8000 |
| `--concurrency`, `-n` | 固定并发数 | 100 |
| `--duration`, `-d` | 测试时长（秒） | 300 |
| `--scenario` | 压测模式 | fixed |
| `--base` | 阶梯起始并发 | 100 |
| `--increment` | 阶梯增量 | 100 |
| `--steps` | 阶梯级数 | 10 |
| `--step-duration` | 每级持续时间 | 60 |
| `--peak` | 突发峰值并发 | 2000 |
| `--warmup` | 预热时长 | 10 |
| `--input-len` | 最大输入长度 | 4096 |
| `--output-len` | 最大输出长度 | 1024 |
| `--stream` | 启用流式输出 | false |
| `--dataset` | 数据集文件路径 | - |
| `--generate` | 使用泛化生成 | false |
| `--short-ratio` | 短文本比例 | 0.7 |
| `--long-ratio` | 长文本比例 | 0.3 |
| `--output`, `-o` | 结果输出目录 | ./results |

## 压测模式

### Fixed (固定并发)
维持恒定数量的并发请求，适合持续压力测试。

### Step (阶梯升压)
分阶段逐步增加并发数，便于观察系统性能拐点。

### Burst (突发洪峰)
瞬时发起大量请求，测试系统峰值承载能力。

### Streaming (流式响应)
测试流式输出场景下的性能指标（TTFT、TPOT）。

### Long Context (长上下文)
针对长输入序列的专项压测，验证长上下文处理能力。

## 配置文件

编辑 `config/default.yaml`:

```yaml
vllm:
  host: "localhost"
  port: 8000
  model: "Qwen/Qwen2.5-7B-Instruct"
  tensor_parallel: 1

load:
  type: "fixed"
  base_concurrency: 100
  duration: 300
  warmup_duration: 10

dataset:
  mode: "import"
  import:
    path: "./datasets/historical_multi_turn.json"
    format: "json"
    text_field: "instruction"
    type_field: "type"
    max_tokens_field: "max_tokens"
  generate:
    short_ratio: 0.6
    long_ratio: 0.4
    max_input_len: 512
    max_output_len: 512

request:
  stream: true
  max_tokens: 512

output:
  path: "./results"
```

### 导入数据集说明

当前支持以下格式：
- `json`: 顶层必须是对象数组
- `jsonl`: 每行一个 JSON 对象
- `csv`: 表头列名映射字段

导入配置支持这些字段：
- `path`: 数据集路径
- `format`: `json` | `jsonl` | `csv`
- `text_field`: 文本字段名，例如 `instruction`
- `type_field`: 请求类型字段名，默认 `type`
- `max_tokens_field`: 单条样本输出长度字段名，默认 `max_tokens`

如果样本里没有 `max_tokens_field`，会回退到配置中的 `dataset.generate.max_output_len`。
如果运行时又传了 `--output-len`，则会覆盖 `request.max_tokens`，但不会覆盖样本里显式提供的 `max_tokens`。

### MFU / FLOPs 指标说明

如果你希望 vLLM 暴露 `estimated_flops_per_gpu_total` 等 MFU 指标，需要：
- vLLM `>= 0.18.0`
- 启动时加 `--enable-mfu-metrics`

例如：

```bash
vllm serve ./models/Qwen3.5-0.8B \
  --dtype auto \
  --gpu-memory-utilization 0.85 \
  --enable-mfu-metrics \
  --max-num-batched-tokens 8192 \
  --max-model-len 1024 \
  --max-num-seqs 64
```

注意：`estimated_flops_per_gpu_created` 不是实际 FLOPs 值，不应直接当作算力使用。

## 输出指标

### 吞吐量
- **QPS**: 每秒请求数
- **TPS**: 每秒 token 数

### 时延
- **TTFT**: 首 token 时延 (Time To First Token)
- **TPOT**: 单 token 生成时延 (Time Per Output Token)
- **P50/P90/P99**: 分位数时延

### 可靠性
- **Success Rate**: 请求成功率
- **Error Rate**: 错误率

### vLLM 内部指标
- **Batch Size**: 动态 batch 大小
- **KV Cache Usage**: KV 缓存使用率
- **GPU Utilization**: GPU 利用率

## 项目结构

```
llm_benchmark/
├── bench.py                    # CLI 主入口
├── config/
│   └── default.yaml           # 默认配置
├── src/
│   ├── cli.py                  # 命令行
│   ├── config.py               # 配置加载
│   ├── load/
│   │   ├── dataset.py          # 数据集管理
│   │   ├── generator.py        # 请求生成
│   │   └── controller.py       # 流量控制
│   ├── client/
│   │   └── openai_client.py    # OpenAI 兼容客户端
│   ├── metrics/
│   │   ├── collector.py        # 指标采集
│   │   └── vllm_exporter.py    # vLLM 指标
│   ├── scenario/
│   │   └── manager.py          # 场景管理
│   ├── eval/                   # 基准评测模块
│   │   ├── datasets/           # 评测数据集加载器
│   │   ├── runner.py           # 评测执行器
│   │   ├── prompts.py          # Prompt 模板
│   │   └── scorer.py           # 答案评分
│   ├── autotune/               # Auto-Tuning Agent
│   │   ├── config.py           # 搜索空间配置
│   │   ├── search.py           # 搜索策略 (Bayesian/Random/Grid)
│   │   ├── evaluator.py        # 配置评估器
│   │   ├── optimizer.py        # 主优化器
│   │   └── templates.py        # 部署模板生成
│   ├── device/                 # 多厂商 GPU 支持
│   │   ├── profile.py          # 设备配置 (厂商/环境变量/参数)
│   │   └── monitor.py          # 跨平台 GPU 利用率/详情/进程采集
│   ├── webui/                  # WebUI 模块
│   │   ├── app.py              # Streamlit 主入口
│   │   └── views/              # 各页面视图
│   └── report/
│       └── generator.py        # 报告生成
├── lb/                         # Load Balancer 模块
│   ├── app.py                  # FastAPI 主服务
│   ├── scheduler.py            # 请求调度
│   ├── process_manager.py      # 实例进程管理
│   └── monitor.py              # 指标监控
└── results/                    # 测试结果输出目录
```

## 使用 Python API

```python
from src.config import load_config
from src.scenario.manager import ScenarioManager
from src.load.generator import LoadGenerator
from src.load.controller import TrafficController
from src.client.openai_client import OpenAIClient
from src.metrics.collector import MetricsCollector
from src.report.generator import ReportGenerator

# 配置
config = load_config("config.yaml")

# 初始化组件
manager = ScenarioManager(config)
generator = LoadGenerator(config)
controller = TrafficController(config)
client = OpenAIClient(config)
collector = MetricsCollector(config)
report_gen = ReportGenerator(config)

# 执行测试
scenario = generator.create_scenario()
collector.start()
results = controller.run(scenario, generator, client)
collector.stop()

# 生成报告
report = report_gen.generate(results, collector.get_metrics())
report_gen.save(report, "results/benchmark.json")
```

## vLLM 多实例负载均衡

`lb/` 目录提供了一个独立的多实例 vLLM 负载均衡子系统，支持：

- **统一 OpenAI 兼容接口**: `/v1/chat/completions`、`/v1/models`
- **智能调度**: 基于 vLLM 运行队列与等待队列的 least-load 策略
- **进程管理**: 自动拉起/停止多个 vLLM 实例
- **Web 控制台**: 实例状态监控、配置编辑、启停操作
- **流式代理**: 原样透传 SSE 流式响应

### 安装

```bash
# 1. 安装本工具包（包含 vllm-lb 命令）
pip install -e .

# 2. 安装 vLLM（如果尚未安装）
pip install vllm

# 3. 验证安装
vllm-lb --help
```

### 快速启动

```bash
# 启动负载均衡器
vllm-lb serve --config lb/config/default.yaml
```

默认监听 `0.0.0.0:9000`，Web UI 在 `http://localhost:9000/`。

**注意**: 
- 首次启动会自动从 ModelScope 下载模型，请确保网络畅通
- 根据你的 GPU 数量修改 `lb/config/default.yaml` 中的 `instances` 配置
- 单 GPU 环境只需保留一个实例配置

### 配置示例

```yaml
server:
  host: "0.0.0.0"
  port: 9000
  request_timeout: 180

scheduler:
  strategy: "least_load"
  refresh_interval: 2
  queue_weight: 2.0
  inflight_weight: 1.0

instances:
  # 单 GPU 配置示例
  - id: "gpu0"
    enabled: true
    managed: true
    host: "127.0.0.1"
    port: 8001
    # ModelScope 模型 ID，vLLM 会自动下载
    model: "Qwen/Qwen2.5-0.5B-Instruct"
    gpu_ids: "0"
    gpu_memory_utilization: 0.80
    max_model_len: 4096
    enable_mfu_metrics: true
    extra_args: []

  # 多 GPU 配置示例（取消注释启用）
  # - id: "gpu1"
  #   enabled: true
  #   managed: true
  #   host: "127.0.0.1"
  #   port: 8002
  #   model: "Qwen/Qwen2.5-0.5B-Instruct"
  #   gpu_ids: "1"
  #   gpu_memory_utilization: 0.80
  #   max_model_len: 4096
  #   enable_mfu_metrics: true
```

### API 端点

**对外接口**:
- `GET /health` - 健康检查
- `GET /v1/models` - 聚合模型列表
- `POST /v1/chat/completions` - 聊天完成（支持流式）

**管理接口**:
- `GET /admin/state` - 完整状态
- `GET /admin/metrics` - 代理指标
- `GET /admin/config` - 当前配置
- `PUT /admin/config` - 更新配置
- `POST /admin/reload` - 重新加载配置
- `POST /admin/instances/{id}/start` - 启动实例
- `POST /admin/instances/{id}/stop` - 停止实例

### 调用示例

```bash
# 查看可用模型
curl http://localhost:9000/v1/models

# 非流式请求（不指定模型会自动选择）
curl -X POST http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 100
  }'

# 流式请求（指定模型）
curl -X POST http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true,
    "max_tokens": 100
  }'
```

### 与现有压测联调

负载均衡器对下游工具透明，可直接作为 `bench.py` 的目标：

```bash
python bench.py run --vllm-host 127.0.0.1 --vllm-port 9000 --concurrency 50 --duration 60
```

## Auto-Tuning Agent (vLLM 参数自动调优)

`src/autotune/` 模块提供了基于 Bayesian Optimization 的 vLLM 参数自动调优功能，帮助找到最优的部署配置。

### 功能特性

- **智能搜索**: 使用 Optuna TPE 采样器进行贝叶斯优化
- **多策略支持**: Bayesian / Random / Grid 三种搜索策略
- **灵活目标**: 支持吞吐量优先、延迟优先、平衡模式
- **预设模板**: 内置 Default / High Throughput / Low Latency 三种搜索空间
- **WebUI 集成**: 可视化配置、实时进度、结果分析

### 快速开始

```bash
# 基本用法 - 优化吞吐量
python bench.py tune --model ./models/Qwen3.5-4B --gpu-ids "0"

# 多 GPU + 更多试验
python bench.py tune --model ./models/Qwen3.5-4B --gpu-ids "0,1,2,3" --max-trials 30

# 优化低延迟
python bench.py tune --model ./models/Qwen3.5-4B --gpu-ids "0" --objective latency

# 使用随机搜索
python bench.py tune --model ./models/Qwen3.5-4B --gpu-ids "0" --strategy random
```

### CLI 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model`, `-m` | 模型路径或名称 | 必填 |
| `--gpu-ids` | GPU ID 列表 | 必填 |
| `--device` | GPU 厂商类型 (auto/nvidia/rocm/ascend/cambricon/biren/metax/moorethreads) | auto |
| `--strategy` | 搜索策略 (bayesian/random/grid) | bayesian |
| `--objective` | 优化目标 (throughput/latency/balanced) | throughput |
| `--max-trials` | 最大尝试次数 | 20 |
| `--concurrency` | 负载测试并发数 | 100 |
| `--duration` | 每次评估时长（秒） | 60 |
| `--startup-timeout` | vLLM 启动超时（秒） | 300 |
| `--output`, `-o` | 输出目录 | ./results/autotune |

### 搜索参数

默认搜索空间：

| 参数 | 范围 |
|------|------|
| `gpu_memory_utilization` | 0.70 - 0.95 (步长 0.05) |
| `tensor_parallel` | 基于 GPU 数量自动调整 |
| `max_model_len` | [4096, 8192, 16384, 32768] |
| `max_num_seqs` | 32 - 256 (步长 32) |

### 输出文件

- `tuning_report.json`: 完整调参报告
- `tuning_history.csv`: 所有试验记录
- `best_config.yaml`: 最优配置模板

### WebUI 使用

启动 WebUI 后，导航到 **Auto-Tuning** 页面：

```bash
streamlit run src/webui/app.py
```

在 WebUI 中可以：
1. 配置模型路径、GPU、优化目标
2. 选择或自定义搜索空间
3. 实时查看调参进度
4. 分析参数影响
5. 下载最优配置

### Python API

```python
from src.autotune import AutoTuner, get_default_vllm_space

# 创建调参器
tuner = AutoTuner(
    model_path="./models/Qwen3.5-4B",
    gpu_ids="0",
    strategy="bayesian",
    objective="throughput",
    max_trials=20,
)

# 运行调参
best_result = tuner.run()

# 查看最优配置
print(f"Best TPS: {best_result.tps:.2f}")
print(f"Best config: {best_result.config}")
```

## 多厂商 GPU 支持

支持在多种 GPU 厂商平台上运行压测和自动调参：

| 厂商 | 标识 | 设备环境变量 | GPU 监控 | `--gpu-memory-utilization` |
|------|------|------------|---------|---------------------------|
| NVIDIA | `nvidia` | `CUDA_VISIBLE_DEVICES` | pynvml | 支持 |
| AMD ROCm | `rocm` | `HIP_VISIBLE_DEVICES` | rocm-smi | 支持 |
| 华为昇腾 | `ascend` | `ASCEND_RT_VISIBLE_DEVICES` | npu-smi | 不支持 |
| 寒武纪 | `cambricon` | `MLU_VISIBLE_DEVICES` | cnmon | 不支持 |
| 壁仞 | `biren` | `BR_VISIBLE_DEVICES` | biren-smi | 不支持 |
| Metax | `metax` | `METAX_VISIBLE_DEVICES` | metax-smi | 不支持 |
| 摩尔线程 | `moorethreads` | `MT_VISIBLE_DEVICES` | mthreads-gmi | 不支持 |

### 命令行使用

```bash
# 自动检测 GPU 类型（默认）
python bench.py tune --model ./models/Qwen3.5-4B --gpu-ids "0" --device auto

# 手动指定华为昇腾
python bench.py tune --model ./models/Qwen3.5-4B --gpu-ids "0,1" --device ascend

# 寒武纪 MLU
python bench.py tune --model ./models/Qwen3.5-4B --gpu-ids "0" --device cambricon

# 壁仞 GPU
python bench.py tune --model ./models/Qwen3.5-4B --gpu-ids "0" --device biren
```

### 负载均衡配置

在 `lb/config/default.yaml` 中指定设备类型：

```yaml
instances:
  - id: "npu0"
    device: ascend           # 指定设备类型
    model: "Qwen/Qwen2.5-7B-Instruct"
    gpu_ids: "0"
    # device 为 ascend/cambricon/biren 等时，无需 gpu_memory_utilization
    max_model_len: 4096
```

### 自动检测逻辑

`--device auto` 按以下优先级自动检测：

1. pynvml 初始化成功 → `nvidia`
2. `rocm-smi` 可用 → `rocm`
3. `npu-smi` 可用 → `ascend`
4. `cnmon` 可用 → `cambricon`
5. `biren-smi` / `brsmi` 可用 → `biren`
6. `metax-smi` 可用 → `metax`
7. `mthreads-gmi` 可用 → `moorethreads`
8. 默认 → `nvidia`

### Python API

```python
from src.device import detect_device, get_device_profile, get_gpu_utilization, get_gpu_details, get_gpu_processes

# 自动检测
device = detect_device()
print(f"Detected: {device}")

# 获取设备配置
profile = get_device_profile("ascend")
print(f"Env var: {profile.visible_devices_env}")
print(f"Supports gpu_memory_utilization: {profile.supports_gpu_mem_util}")

# 跨平台 GPU 利用率采集
util = get_gpu_utilization()

# 逐卡详细信息（NVIDIA 支持最完整：名称/UUID/利用率/显存/温度/功耗/风扇）
gpus = get_gpu_details()
for g in gpus:
    print(f"GPU {g.index}: {g.name} | {g.gpu_util:.0f}% | {g.mem_used_mb:.0f}/{g.mem_total_mb:.0f}MB | {g.temperature_c:.0f}°C | {g.power_draw_w:.0f}W")

# 查看 GPU 上运行的进程（仅 NVIDIA）
procs = get_gpu_processes()
for p in procs:
    print(f"GPU {p.gpu_index}: PID {p.pid} ({p.process_name}) - {p.used_memory_mb:.0f}MB")
```

## WebUI

提供可视化操作界面：

```bash
streamlit run src/webui/app.py
```

### GPU Monitor

实时监控每张 GPU 卡的状态，NVIDIA CUDA 优先支持完整指标：

- GPU / 显存利用率进度条（颜色编码：<60% 绿 / <85% 黄 / >=85% 红）
- 显存使用量（已用/总量 MB）
- 温度、功耗、风扇转速
- GPU 上运行的进程列表（PID、进程名、显存占用）
- 自动刷新（可调 1-10 秒间隔）
- 支持多厂商自动检测

功能模块：
- **Model Check**: 快速检测模型是否正常工作
- **Evaluation**: 运行基准评测
- **Load Testing**: 配置并运行压测
- **Auto-Tuning**: vLLM 参数自动调优
- **Load Balancer**: 管理 vLLM 实例
- **GPU Monitor**: 实时 GPU 卡级监控（利用率/显存/温度/功耗/进程）
- **Results**: 查看历史结果

## License

MIT
