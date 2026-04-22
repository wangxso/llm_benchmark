# LLM 基准评测模块

评估大模型在标准基准数据集上的知识和推理能力。

## 支持的评测集

| 评测集 | 数据来源 | 样本数 | 选项数 | 说明 |
|--------|----------|--------|--------|------|
| GPQA | `idavidrein/gpqa` | 448 | 4 | 研究生级别问答，需要 HuggingFace 登录 |
| MMLU-Pro | `TIGER-Lab/MMLU-Pro` | 12,032 | 10 | MMLU 增强版，更具挑战性 |
| MMLU-Redux | `edinburgh-dawg/mmlu-redux` | - | 4 | MMLU 纠错版 |
| SuperGPQA | `m-a-p/SuperGPQA` | - | 4 | 综合研究生级别评测 |
| C-Eval | `ceval/ceval-exam` | 13,941 | 4 | 中文综合能力评测，52 个学科 |

## 快速开始

### 1. 启动 vLLM 服务（本地测试）

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
```

### 2. 运行评测

```bash
# 本地 vLLM 服务
bench.py eval --benchmark gpqa --vllm-host localhost --samples 100

# 远程 OpenAI 兼容 API
bench.py eval --benchmark ceval \
    --api-base-url https://api.openai.com/v1 \
    --api-key sk-xxx \
    --model gpt-4

# Anthropic API
bench.py eval --benchmark gpqa \
    --api-type anthropic \
    --api-base-url https://api.anthropic.com \
    --api-key sk-ant-xxx \
    --model claude-3-haiku-20240307

# 其他兼容 API（如 MiniMax）
bench.py eval --benchmark gpqa \
    --api-type anthropic \
    --api-base-url https://api.minimaxi.com/anthropic \
    --api-key YOUR_KEY \
    --model MiniMax-M2.7
```

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--benchmark`, `-b` | 评测集名称 | `gpqa` |
| `--vllm-host` | vLLM 服务器地址（本地） | - |
| `--vllm-port` | vLLM 端口 | `8000` |
| `--model`, `-m` | 模型名称 | 自动检测 |
| `--samples`, `-n` | 最大评测样本数 | 全部 |
| `--subject` | 筛选学科 | 全部 |
| `--prompt-style` | Prompt 风格 | `zero_shot` |
| `--concurrency`, `-c` | 并发请求数 | `8` |
| `--output`, `-o` | 输出目录 | `./results` |
| `--hf-token` | HuggingFace Token | 自动检测 |
| `--api-base-url` | 远程 API 地址 | - |
| `--api-key` | 远程 API Key | - |
| `--api-type` | API 类型 (`openai`/`anthropic`) | `openai` |
| `--list` | 列出可用评测集 | - |

## Prompt 风格

| 风格 | 说明 |
|------|------|
| `zero_shot` | 零样本，直接提问 |
| `few_shot` | 少样本，包含示例 |
| `cot` | Chain-of-Thought，思维链推理 |
| `zero_shot_cn` | 零样本中文版 |

```bash
# 使用 CoT 提示
bench.py eval --benchmark gpqa --vllm-host localhost --prompt-style cot
```

## 输出示例

```
==================================================
EVALUATION SUMMARY: GPQA
==================================================

Model: Qwen/Qwen2.5-7B-Instruct
Prompt style: zero_shot

Overall Accuracy: 28.00%
Correct: 28 / 100

[By Subject]
  Biology: 35.0% (7/20)
  Physics: 30.0% (9/30)
  Chemistry: 24.0% (12/50)

Report saved: results/eval_GPQA_20260421_100000.json
==================================================
```

## 输出文件

评测完成后生成两个文件：

- `eval_{benchmark}_{timestamp}.json` - 汇总报告
- `eval_{benchmark}_{timestamp}_details.json` - 详细结果

### 报告格式

```json
{
  "benchmark": "GPQA",
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "prompt_style": "zero_shot",
  "overall_accuracy": 0.28,
  "total_questions": 100,
  "correct": 28,
  "subjects": {
    "Biology": {"accuracy": 0.35, "correct": 7, "total": 20},
    "Physics": {"accuracy": 0.30, "correct": 9, "total": 30},
    "Chemistry": {"accuracy": 0.24, "correct": 12, "total": 50}
  },
  "timestamp": "2024-04-21T10:00:00"
}
```

## HuggingFace 认证

GPQA 等数据集需要 HuggingFace 认证：

```bash
# 方式1: 命令行登录
huggingface-cli login

# 方式2: 环境变量
export HF_TOKEN=your_token_here

# 方式3: 命令行传入
bench.py eval --benchmark gpqa --vllm-host localhost --hf-token your_token
```

## 学科列表

### GPQA
- Biology
- Physics
- Chemistry

### MMLU-Pro
- math, physics, chemistry, biology
- computer science, engineering, health
- economics, psychology, business
- philosophy, history, law, other

### MMLU-Redux
- abstract_algebra, anatomy, astronomy
- business_ethics, clinical_knowledge
- college_biology, college_chemistry
- computer_security, econometrics
- 等 57 个学科

### C-Eval (中文)
- STEM: computer_network, operating_system, college_physics, college_programming...
- 社科: business_administration, marxism, law...
- 人文: chinese_language_and_literature, art_studies, modern_chinese_history...
- 其他: civil_servant, accountant, physician...

## Python API

```python
from src.eval import get_benchmark, EvalRunner

# 获取评测集
benchmark_cls = get_benchmark("gpqa")
benchmark = benchmark_cls()

# 创建评测器
runner = EvalRunner(
    benchmark=benchmark,
    host="localhost",
    port=8000,
    concurrency=8,
)

# 运行评测
report = runner.run(
    prompt_style="zero_shot",
    max_samples=100,
    output_dir="./results",
)

print(f"Accuracy: {report['overall_accuracy']:.2%}")
```

## 模块结构

```
src/eval/
├── __init__.py      # 模块入口
├── base.py          # 评测基类
├── registry.py      # 评测集注册
├── prompts.py       # Prompt 模板
├── scorer.py        # 答案提取与评分
├── runner.py        # 评测执行器
└── datasets/        # 数据集加载器
    ├── gpqa.py
    ├── mmlu_pro.py
    ├── mmlu_redux.py
    └── super_gpqa.py
```

## 扩展新评测集

```python
# src/eval/datasets/my_benchmark.py
from ..base import BaseBenchmark
from ..registry import register_benchmark

@register_benchmark("my-benchmark")
class MyBenchmark(BaseBenchmark):
    name = "MyBenchmark"
    hf_path = "org/dataset-name"
    description = "My custom benchmark"

    def _parse_row(self, row):
        return {
            "question": row["question"],
            "choices": row["choices"],
            "answer": row["answer"],
            "subject": row.get("subject", "unknown"),
        }
```

然后在 `src/eval/datasets/__init__.py` 中导入即可。
