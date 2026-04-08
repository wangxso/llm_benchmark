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

    click.echo(f"[Benchmark] Starting test: {scenario_name}")
    click.echo(f"[Config] Concurrency: {concurrency}")
    click.echo(f"[Config] Duration: {duration}s")

    manager = ScenarioManager(config)
    generator = LoadGenerator(config)
    controller = TrafficController(config)
    client = OpenAIClient(config)
    collector = MetricsCollector(config)
    report_gen = ReportGenerator(config)

    click.echo("[Info] Initializing test scenario...")
    scenario = manager.create_scenario()

    click.echo("[Info] Starting metrics collection...")
    collector.start()

    click.echo("[Info] Executing load test...")
    results = controller.run(scenario, generator, client)

    click.echo("[Info] Stopping metrics collection...")
    collector.stop()

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
    if cli_args.get("stream"):
        config.setdefault("request", {})["stream"] = True
    if cli_args.get("output_len"):
        config.setdefault("request", {})["max_tokens"] = cli_args["output_len"]
    if cli_args.get("vllm_host"):
        config.setdefault("vllm", {})["host"] = cli_args["vllm_host"]
    if cli_args.get("vllm_port"):
        config.setdefault("vllm", {})["port"] = cli_args["vllm_port"]
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
        click.echo(f"  Avg Batch:    {vllm_metrics.get('avg_batch_size', 0):.1f}")
        click.echo(
            f"  KV Cache:     {vllm_metrics.get('kv_cache_usage', 0) * 100:.1f}%"
        )
        click.echo(
            f"  GPU Util:     {vllm_metrics.get('gpu_utilization', 0) * 100:.1f}%"
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
