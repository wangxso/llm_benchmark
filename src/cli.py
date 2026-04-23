import click
import sys
from pathlib import Path

from .config import load_config
from .scenario.manager import ScenarioManager
from .load.generator import LoadGenerator
from .load.controller import TrafficController
from .client.openai_client import OpenAIClient
from .metrics.collector import MetricsCollector
from .report.generator import ReportGenerator


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """LLM High-Concurrency Simulation Testing Platform"""
    pass


@cli.command("eval")
@click.option("--benchmark", "-b", type=str, default="gpqa",
              help="Benchmark name (gpqa, mmlu-pro, mmlu-redux, super-gpqa, ceval)")
@click.option("--vllm-host", type=str, help="vLLM server host (for local)")
@click.option("--vllm-port", type=int, default=8000, help="vLLM server port")
@click.option("--model", "-m", type=str, help="Model name (required for remote API)")
@click.option("--samples", "-n", type=int, help="Max number of samples to evaluate")
@click.option("--subject", type=str, help="Filter by subject")
@click.option("--prompt-style", type=click.Choice(["zero_shot", "few_shot", "cot", "zero_shot_cn", "few_shot_cn", "cot_cn"]),
              default="zero_shot", help="Prompt style")
@click.option("--concurrency", "-c", type=int, default=8, help="Concurrent requests")
@click.option("--rate-limit", "-r", type=float, default=0, help="Max requests per minute (0=unlimited)")
@click.option("--timeout", "-t", type=int, default=60, help="Request timeout in seconds")
@click.option("--offline", is_flag=True, help="Use cached datasets only (no network)")
@click.option("--output", "-o", type=str, default="./results", help="Output directory")
@click.option("--list", "list_benchmarks", is_flag=True, help="List available benchmarks")
@click.option("--hf-token", type=str, help="HuggingFace token for gated datasets")
@click.option("--api-base-url", type=str, help="Remote API base URL (e.g., https://api.openai.com/v1)")
@click.option("--api-key", type=str, help="API key for remote API")
@click.option("--api-type", type=click.Choice(["openai", "anthropic"]), default="openai",
              help="API type: openai (default) or anthropic")
def eval_cmd(**kwargs):
    """Run benchmark evaluation (GPQA, MMLU-Pro, etc.)

    Examples:
        # Local vLLM server
        bench.py eval --benchmark gpqa --vllm-host localhost

        # Remote OpenAI-compatible API
        bench.py eval --benchmark ceval --api-base-url https://api.openai.com/v1 --api-key sk-xxx --model gpt-4

        # Anthropic API
        bench.py eval --benchmark gpqa --api-type anthropic --api-key sk-ant-xxx --model claude-3-haiku-20240307
    """
    from .eval import get_benchmark, list_benchmarks as get_list, EvalRunner

    if kwargs.get("list_benchmarks"):
        click.echo("Available benchmarks:")
        for name in get_list():
            benchmark_cls = get_benchmark(name)
            bench = benchmark_cls()
            click.echo(f"  - {name}: {bench.description}")
        return

    # Validate: need either vllm-host or api-base-url
    if not kwargs.get("vllm_host") and not kwargs.get("api_base_url"):
        click.echo("Error: Either --vllm-host or --api-base-url is required", err=True)
        click.echo("  Local:  --vllm-host localhost")
        click.echo("  Remote: --api-base-url https://api.example.com/v1 --api-key YOUR_KEY --model MODEL_NAME")
        sys.exit(1)

    # For remote API, model and api-key are required
    if kwargs.get("api_base_url"):
        if not kwargs.get("model"):
            click.echo("Error: --model is required when using --api-base-url", err=True)
            sys.exit(1)
        if not kwargs.get("api_key"):
            click.echo("Error: --api-key is required when using --api-base-url", err=True)
            sys.exit(1)

    benchmark_name = kwargs.get("benchmark", "gpqa")

    try:
        benchmark_cls = get_benchmark(benchmark_name)
        benchmark = benchmark_cls()
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo(f"[Eval] Starting evaluation: {benchmark.name}")
    click.echo(f"[Eval] Model: {kwargs.get('model', 'auto-detect')}")
    click.echo(f"[Eval] Prompt style: {kwargs['prompt_style']}")
    if kwargs.get("api_base_url"):
        click.echo(f"[Eval] API URL: {kwargs['api_base_url']}")
        click.echo(f"[Eval] API type: {kwargs.get('api_type', 'openai')}")
    else:
        click.echo(f"[Eval] Host: {kwargs['vllm_host']}:{kwargs['vllm_port']}")
    if kwargs.get("samples"):
        click.echo(f"[Eval] Max samples: {kwargs['samples']}")
    if kwargs.get("subject"):
        click.echo(f"[Eval] Subject: {kwargs['subject']}")

    runner = EvalRunner(
        benchmark=benchmark,
        host=kwargs.get("vllm_host", "localhost"),
        port=kwargs.get("vllm_port", 8000),
        model=kwargs.get("model"),
        concurrency=kwargs.get("concurrency", 8),
        rate_limit=kwargs.get("rate_limit", 0),
        timeout=kwargs.get("timeout", 60),
        offline=kwargs.get("offline", False),
        hf_token=kwargs.get("hf_token"),
        api_base_url=kwargs.get("api_base_url"),
        api_key=kwargs.get("api_key"),
        api_type=kwargs.get("api_type", "openai"),
    )

    report = runner.run(
        prompt_style=kwargs.get("prompt_style", "zero_shot"),
        max_samples=kwargs.get("samples"),
        subject=kwargs.get("subject"),
        output_dir=kwargs.get("output"),
    )

    print_eval_summary(report)

    # Show answer comparison from details
    if report.get("details"):
        print_answer_comparison(report["details"], limit=20)


def print_eval_summary(report):
    """Print evaluation summary"""
    click.echo("\n" + "=" * 50)
    click.echo(f"EVALUATION SUMMARY: {report.get('benchmark', 'Unknown')}")
    click.echo("=" * 50)

    click.echo(f"\nModel: {report.get('model', 'unknown')}")
    click.echo(f"Prompt style: {report.get('prompt_style', 'unknown')}")
    click.echo(f"\nOverall Accuracy: {report.get('overall_accuracy', 0) * 100:.2f}%")
    click.echo(f"Correct: {report.get('correct', 0)} / {report.get('total_questions', 0)}")

    # Show error statistics if there are failures
    failed = report.get("failed_count", 0)
    if failed > 0:
        click.echo(f"\n[Errors]")
        click.echo(f"  Failed requests: {failed}")
        error_types = report.get("error_types", {})
        error_details = report.get("error_details", {})
        for err_type, count in sorted(error_types.items(), key=lambda x: -x[1]):
            click.echo(f"  - {err_type}: {count}")
            # Show detailed error messages for this type
            if err_type in error_details:
                for detail, detail_count in sorted(error_details[err_type].items(), key=lambda x: -x[1])[:3]:
                    # Truncate long messages
                    detail_display = detail if len(detail) <= 80 else detail[:77] + "..."
                    click.echo(f"      [{detail_count}x] {detail_display}")

    # Print category breakdown if available
    categories = report.get("categories", {})
    if categories:
        click.echo(f"\n[By Category]")
        for cat, stats in sorted(categories.items()):
            if cat != "Average":
                acc = stats["accuracy"] * 100
                correct = stats["correct"]
                total = stats["total"]
                click.echo(f"  {cat}: {acc:.1f}% ({correct}/{total})")
        if "Average" in categories:
            avg = categories["Average"]["accuracy"] * 100
            click.echo(f"  ---\n  Category Avg: {avg:.1f}%")

    # Print subject breakdown
    subjects = report.get("subjects", {})
    if subjects:
        click.echo(f"\n[By Subject]")
        # Show top 10 subjects by accuracy
        sorted_subjects = sorted(subjects.items(), key=lambda x: -x[1]["accuracy"])[:10]
        for subject, stats in sorted_subjects:
            acc = stats["accuracy"] * 100
            correct = stats["correct"]
            total = stats["total"]
            click.echo(f"  {subject}: {acc:.1f}% ({correct}/{total})")
        if len(subjects) > 10:
            click.echo(f"  ... and {len(subjects) - 10} more subjects")

    if report.get("report_file"):
        click.echo(f"\nReport saved: {report['report_file']}")


def print_answer_comparison(details: list, limit: int = 20):
    """Print answer comparison table for debugging"""
    click.echo("\n" + "=" * 100)
    click.echo("ANSWER COMPARISON (first {} results)".format(min(limit, len(details))))
    click.echo("=" * 100)

    # Header
    click.echo("\n{:<4} {:<6} {:<6} {:<30} {:<50}".format(
        "#", "Actual", "Pred", "Subject", "Response (truncated)"
    ))
    click.echo("-" * 100)

    correct_count = 0
    shown = 0
    for i, d in enumerate(details):
        if not d.get("success"):
            continue

        actual = d.get("actual", "?")
        predicted = d.get("predicted", "N/A")
        subject = d.get("subject", "unknown")[:28]
        response = d.get("response", "")[:48]

        is_correct = actual == predicted
        if is_correct:
            correct_count += 1

        marker = "✓" if is_correct else "✗"
        click.echo("{:<4} {:<6} {:<6} {:<30} {:<50} {}".format(
            i + 1, actual, predicted or "N/A", subject, response, marker
        ))

        shown += 1
        if shown >= limit:
            break

    click.echo("-" * 100)
    click.echo(f"\nShown: {shown} | Correct in shown: {correct_count}/{shown}")

    click.echo("=" * 50 + "\n")


@cli.command("check")
@click.option("--api-base-url", type=str, help="API base URL")
@click.option("--api-key", type=str, help="API key")
@click.option("--api-type", type=click.Choice(["openai", "anthropic"]), default="openai", help="API type")
@click.option("--model", "-m", type=str, required=True, help="Model name")
@click.option("--timeout", "-t", type=int, default=60, help="Request timeout in seconds")
def check_cmd(**kwargs):
    """Quick model capability check to detect if model is degraded.

    Tests basic reasoning, math, and knowledge capabilities.

    Examples:
        bench.py check --model gpt-4 --api-key sk-xxx --api-base-url https://api.openai.com/v1
        bench.py check --model MiniMax-M2.7 --api-type anthropic --api-key sk-xxx --api-base-url https://api.minimaxi.com/anthropic
    """
    import asyncio
    import aiohttp
    import json

    # Test questions with expected answers
    tests = [
        {
            "name": "Simple Math",
            "prompt": "What is 15 + 27? Answer with just the number.",
            "check": lambda r: "42" in r,
        },
        {
            "name": "Basic Knowledge",
            "prompt": "What is the capital of France? Answer with just the city name.",
            "check": lambda r: "paris" in r.lower(),
        },
        {
            "name": "Logic Reasoning",
            "prompt": "If all cats are mammals, and Tom is a cat, is Tom a mammal? Answer yes or no.",
            "check": lambda r: "yes" in r.lower(),
        },
        {
            "name": "Code Ability",
            "prompt": "Write a Python function to calculate factorial. Just output the function code, no explanation.",
            "check": lambda r: "def " in r and "factorial" in r.lower(),
        },
        {
            "name": "Chinese",
            "prompt": "用中文回答：1+1等于几？只回答数字。",
            "check": lambda r: "2" in r or "二" in r,
        },
    ]

    base_url = kwargs.get("api_base_url", "").rstrip("/")
    api_key = kwargs.get("api_key")
    api_type = kwargs.get("api_type", "openai")
    model = kwargs.get("model")
    timeout = kwargs.get("timeout", 60)

    if not base_url:
        click.echo("Error: --api-base-url is required", err=True)
        return
    if not api_key:
        click.echo("Error: --api-key is required", err=True)
        return

    click.echo("=" * 50)
    click.echo("MODEL CAPABILITY CHECK")
    click.echo("=" * 50)
    click.echo(f"\nModel: {model}")
    click.echo(f"API: {base_url}")
    click.echo(f"Type: {api_type}")
    click.echo()

    # Prepare headers
    if api_type == "anthropic":
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
        # Handle MiniMax anthropic endpoint
        if base_url.endswith("/anthropic"):
            url = f"{base_url}/v1/messages"
        else:
            url = f"{base_url}/messages" if not base_url.endswith("/messages") else base_url
    else:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        url = f"{base_url}/chat/completions"

    async def run_check():
        results = []
        async with aiohttp.ClientSession() as session:
            for test in tests:
                try:
                    if api_type == "anthropic":
                        payload = {
                            "model": model,
                            "max_tokens": 500,
                            "messages": [{"role": "user", "content": test["prompt"]}],
                        }
                    else:
                        payload = {
                            "model": model,
                            "max_tokens": 500,
                            "temperature": 0.0,
                            "messages": [{"role": "user", "content": test["prompt"]}],
                        }

                    async with session.post(
                        url,
                        json=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=timeout),
                    ) as resp:
                        if resp.status != 200:
                            text = await resp.text()
                            # Try to extract error message
                            try:
                                err_data = json.loads(text)
                                detail = err_data.get("error", {}).get("message", text[:100])
                            except:
                                detail = text[:100]
                            results.append({
                                "name": test["name"],
                                "passed": False,
                                "error": f"HTTP {resp.status}: {detail}",
                            })
                            continue

                        data = await resp.json()

                        if api_type == "anthropic":
                            content = data.get("content", [{}])[0].get("text", "") if isinstance(data.get("content"), list) else data.get("content", "")
                        else:
                            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

                        passed = test["check"](content)
                        results.append({
                            "name": test["name"],
                            "passed": passed,
                            "response": content[:100] + "..." if len(content) > 100 else content,
                        })

                except asyncio.TimeoutError:
                    results.append({"name": test["name"], "passed": False, "error": "Timeout"})
                except Exception as e:
                    results.append({"name": test["name"], "passed": False, "error": str(e)[:100]})

        return results

    results = asyncio.run(run_check())

    # Print results
    passed = sum(1 for r in results if r["passed"])
    total = len(results)

    for r in results:
        status = "✓ PASS" if r["passed"] else "✗ FAIL"
        click.echo(f"  [{status}] {r['name']}")
        if r["passed"]:
            click.echo(f"         Response: {r.get('response', 'N/A')}")
        else:
            click.echo(f"         Error: {r.get('error', r.get('response', 'Unknown'))}")

    click.echo()
    click.echo("-" * 50)

    # Verdict
    pass_rate = passed / total * 100 if total > 0 else 0
    click.echo(f"Result: {passed}/{total} tests passed ({pass_rate:.0f}%)")

    if pass_rate >= 80:
        click.echo("Verdict: ✓ Model appears to be working normally")
    elif pass_rate >= 40:
        click.echo("Verdict: ⚠ Model may be partially degraded")
    else:
        click.echo("Verdict: ✗ Model appears to be degraded or misconfigured")

    click.echo("=" * 50)


@cli.command()
@click.option("--config", "-c", type=click.Path(exists=True), help="Config file path")
@click.option("--concurrency", "-n", type=int, help="Fixed concurrency")
@click.option("--duration", "-d", type=int, help="Test duration in seconds")
@click.option(
    "--scenario",
    type=click.Choice(["fixed", "step", "burst", "streaming", "long_context"]),
    help="Load type",
)
@click.option("--base", type=int, help="Base concurrency for step/burst")
@click.option("--increment", type=int, help="Increment for step load")
@click.option("--steps", type=int, help="Number of steps")
@click.option("--step-duration", type=int, help="Duration per step")
@click.option("--peak", type=int, help="Peak concurrency for burst")
@click.option("--warmup", type=int, help="Warmup duration")
@click.option("--input-len", type=int, help="Max input length for long_context")
@click.option("--output-len", type=int, help="Max output length")
@click.option("--stream/--no-stream", default=False, help="Enable streaming")
@click.option(
    "--dataset", type=click.Path(exists=True), help="Dataset path for prompts"
)
@click.option("--generate/--no-generate", default=False, help="Use synthetic prompts")
@click.option("--short-ratio", type=float, help="Short prompt ratio")
@click.option("--long-ratio", type=float, help="Long prompt ratio")
@click.option("--models", type=str, help="Comma-separated model names for multi-model")
@click.option("--mix-ratio", type=str, help="Model mix ratios")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option("--vllm-host", type=str, help="vLLM host")
@click.option("--vllm-port", type=int, help="vLLM port")
@click.option("--model", type=str, help="Model name to use in requests")
def run(**kwargs):
    """Run benchmark test"""
    config = load_config(kwargs.get("config"))

    config = merge_cli_config(config, kwargs)

    if not config.get("vllm", {}).get("host"):
        click.echo(
            "Error: vLLM host not configured (use --vllm-host or config file)", err=True
        )
        sys.exit(1)

    scenario_name = config.get("scenario", {}).get("name", "unnamed")
    concurrency = config.get("load", {}).get("base_concurrency", 100)
    duration = config.get("load", {}).get("duration", 300)
    model_name = config.get("vllm", {}).get("model", "unknown")

    click.echo(f"[Benchmark] Starting test: {scenario_name}")
    click.echo(f"[Config] Model: {model_name}")
    click.echo(f"[Config] Target: {config.get('vllm', {}).get('host')}:{config.get('vllm', {}).get('port')}")
    click.echo(f"[Config] Concurrency: {concurrency}")
    click.echo(f"[Config] Duration: {duration}s")
    click.echo(
        f"[Config] Dataset mode: {config.get('dataset', {}).get('mode', 'generate')}"
    )
    click.echo(
        f"[Config] Dataset path: {config.get('dataset', {}).get('import', {}).get('path', '-') }"
    )
    click.echo(
        f"[Config] Dataset text field: {config.get('dataset', {}).get('import', {}).get('text_field', 'prompt')}"
    )
    click.echo(
        f"[Config] Request max_tokens: {config.get('request', {}).get('max_tokens', 1024)}"
    )
    click.echo(
        f"[Config] Dataset max_output_len: {config.get('dataset', {}).get('generate', {}).get('max_output_len', 2048)}"
    )
    click.echo(
        f"[Config] Dataset max_input_len: {config.get('dataset', {}).get('generate', {}).get('max_input_len', 4096)}"
    )
    click.echo(f"[Config] Stream: {config.get('request', {}).get('stream', False)}")

    manager = ScenarioManager(config)
    generator = LoadGenerator(config)
    controller = TrafficController(config)
    client = OpenAIClient(config)
    collector = MetricsCollector(config)
    report_gen = ReportGenerator(config)

    click.echo("[Info] Initializing test scenario...")
    scenario = generator.create_scenario()

    collector.set_results(controller.get_results_live)
    click.echo("[Info] Starting metrics collection...")
    collector.start()

    click.echo("[Info] Executing load test...")
    results = controller.run(scenario, generator, client)

    click.echo("[Info] Stopping metrics collection...")
    collector.stop()

    import asyncio

    asyncio.run(client.close())

    click.echo("[Info] Generating report...")
    report = report_gen.generate(results, collector.get_metrics())

    output_path = config.get("output", {}).get("path", "./results")
    output_file = f"{output_path}/benchmark_{report['test_info']['timestamp']}.json"

    import os

    os.makedirs(output_path, exist_ok=True)
    report_gen.save(report, output_file)

    click.echo(f"[Success] Results saved to: {output_path}")

    print_summary(report)


@cli.command()
@click.option("--port", type=int, default=8080, help="Monitor port")
def monitor(port):
    """Start real-time monitor"""
    click.echo(f"[Monitor] Starting on port {port}")
    click.echo("[Monitor] Press Ctrl+C to exit")

    collector = MetricsCollector({})
    collector.start()

    try:
        import time

        while True:
            metrics = collector.get_current_metrics()
            print_metrics(metrics)
            time.sleep(1)
    except KeyboardInterrupt:
        click.echo("\n[Monitor] Stopped")


@cli.command()
@click.option(
    "--input",
    "-i",
    type=click.Path(exists=True),
    required=True,
    help="Input result JSON",
)
@click.option("--output", "-o", type=click.Path(), help="Output report file")
def report(input, output):
    """Generate report from result file"""
    import json

    with open(input, "r") as f:
        data = json.load(f)

    print_summary(data)

    if output:
        import shutil

        shutil.copy(input, output)
        click.echo(f"[Success] Report saved to: {output}")


def merge_cli_config(config, cli_args):
    """Merge CLI arguments into config"""
    if cli_args.get("concurrency"):
        config.setdefault("load", {})["base_concurrency"] = cli_args["concurrency"]
    if cli_args.get("duration"):
        config.setdefault("load", {})["duration"] = cli_args["duration"]
    if cli_args.get("scenario"):
        config.setdefault("load", {})["type"] = cli_args["scenario"]
    if cli_args.get("base"):
        config.setdefault("load", {})["base_concurrency"] = cli_args["base"]
    if cli_args.get("increment"):
        config.setdefault("load", {})["step_increment"] = cli_args["increment"]
    if cli_args.get("steps"):
        config.setdefault("load", {})["max_concurrency"] = (
            cli_args["base"] + cli_args["increment"] * cli_args["steps"]
        )
    if cli_args.get("step_duration"):
        config.setdefault("load", {})["step_duration"] = cli_args["step_duration"]
    if cli_args.get("peak"):
        config.setdefault("load", {})["peak_concurrency"] = cli_args["peak"]
    if cli_args.get("warmup"):
        config.setdefault("load", {})["warmup_duration"] = cli_args["warmup"]
    if cli_args.get("dataset"):
        config.setdefault("dataset", {})["mode"] = "import"
        config.setdefault("dataset", {}).setdefault("import", {})["path"] = cli_args[
            "dataset"
        ]
    if cli_args.get("stream"):
        config.setdefault("request", {})["stream"] = True
    if cli_args.get("output_len"):
        config.setdefault("request", {})["max_tokens"] = cli_args["output_len"]
    if cli_args.get("vllm_host"):
        config.setdefault("vllm", {})["host"] = cli_args["vllm_host"]
    if cli_args.get("vllm_port"):
        config.setdefault("vllm", {})["port"] = cli_args["vllm_port"]
    if cli_args.get("model"):
        config.setdefault("vllm", {})["model"] = cli_args["model"]
    if cli_args.get("short_ratio"):
        config.setdefault("dataset", {})["short_ratio"] = cli_args["short_ratio"]
    if cli_args.get("long_ratio"):
        config.setdefault("dataset", {})["long_ratio"] = cli_args["long_ratio"]

    return config


def print_summary(report):
    """Print summary to console"""
    click.echo("\n" + "=" * 50)
    click.echo("BENCHMARK SUMMARY")
    click.echo("=" * 50)

    metrics = report.get("metrics", {})
    click.echo(f"\n[Throughput]")
    click.echo(f"  QPS:     {metrics.get('qps', 0):.2f} req/s")
    click.echo(f"  TPS:     {metrics.get('tps', 0):.2f} tokens/s")

    click.echo(f"\n[Latency]")
    click.echo(f"  TTFT:    {metrics.get('ttft_ms', 0):.2f} ms")
    click.echo(f"  TPOT:    {metrics.get('tpot_ms', 0):.2f} ms")
    click.echo(f"  P50:     {metrics.get('latency_p50_ms', 0):.2f} ms")
    click.echo(f"  P90:     {metrics.get('latency_p90_ms', 0):.2f} ms")
    click.echo(f"  P99:     {metrics.get('latency_p99_ms', 0):.2f} ms")

    click.echo(f"\n[Reliability]")
    click.echo(f"  Success: {metrics.get('success_rate', 0) * 100:.2f}%")
    click.echo(f"  Errors: {metrics.get('error_rate', 0) * 100:.2f}%")

    vllm_metrics = report.get("vllm_metrics", {})
    if vllm_metrics:
        click.echo(f"\n[vLLM Metrics]")
        click.echo(f"  Avg Batch:    {vllm_metrics.get('batch_size', 0):.1f}")
        click.echo(
            f"  KV Cache:     {vllm_metrics.get('kv_cache_usage', 0) * 100:.1f}%"
        )
        click.echo(
            f"  GPU Util:     {vllm_metrics.get('gpu_utilization', 0) * 100:.1f}%"
        )
        click.echo(
            f"  Actual FLOPs: {vllm_metrics.get('actual_flops_per_second', 0):.3e} FLOPs/s"
        )
        click.echo(
            f"  Actual TFLOPs:{vllm_metrics.get('actual_tflops_per_second', 0):.6f} TFLOPs/s"
        )

    click.echo("=" * 50 + "\n")


def print_metrics(metrics):
    """Print current metrics"""
    click.echo(
        f"\rQPS: {metrics.get('qps', 0):.1f} | "
        f"P99: {metrics.get('latency_p99', 0):.0f}ms | "
        f"Active: {metrics.get('active_requests', 0)}",
        end="",
    )


if __name__ == "__main__":
    cli()
