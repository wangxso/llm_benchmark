# LLM High-Concurrency Simulation Testing Platform

高性能仿真测试平台，持续制造"海量用户并发请求"，压测底层 vLLM 推理服务，验证系统的并行处理能力、吞吐、时延、稳定性。

## 功能特性

- **多模式压测**: 固定并发、阶梯升压、突发洪峰、长上下文、流式响应
- **数据集管理**: 支持导入（JSONL/CSV）和泛化生成两种模式
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
```

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
  mode: "generate"
  generate:
    short_ratio: 0.7
    long_ratio: 0.3
    max_input_len: 4096
    max_output_len: 2048

output:
  path: "./results"
```

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

## License

MIT
