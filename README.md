# LLM High-Concurrency Simulation Testing Platform

高性能仿真测试平台，持续制造"海量用户并发请求"，压测底层 vLLM 推理服务，验证系统的并行处理能力、吞吐、时延、稳定性。

## 功能特性

- **多模式压测**: 固定并发、阶梯升压、突发洪峰、长上下文、流式响应
- **基准评测**: 支持 GPQA、MMLU-Pro、MMLU-Redux、SuperGPQA 等标准评测集
- **数据集管理**: 支持导入（JSON/JSONL/CSV）和泛化生成两种模式，可指定文本字段如 `instruction`
- **全链路指标**: QPS、TPS、TTFT、TPOT、P50/P90/P99 时延
- **vLLM 集成**: 自动采集内部指标（batch size、KV Cache、GPU 利用率）
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
│   └── report/
│       └── generator.py        # 报告生成
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
    gpu_memory_utilization: 0.85
    max_model_len: 4096
    enable_mfu_metrics: true
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

## License

MIT
