"""Template generation for Auto-Tuning results."""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml

from .config import TuningConfig, TuningResult


def generate_deploy_template(
    config: TuningConfig,
    output_path: Optional[Path] = None,
    model_path: Optional[str] = None,
    gpu_ids: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate deployment template from the best configuration."""

    template = {
        "vllm": {
            "model": model_path or "<MODEL_PATH>",
            "tensor_parallel_size": config.tensor_parallel,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "max_model_len": config.max_model_len,
            "max_num_seqs": config.max_num_seqs,
        },
        "performance": {
            "optimized_for": "throughput",
            "expected_tps": "<MEASURED_TPS>",
            "expected_latency_p99_ms": "<MEASURED_P99>",
        },
        "deployment": {
            "gpu_ids": gpu_ids or "<GPU_IDS>",
            "environment": {
                "CUDA_VISIBLE_DEVICES": gpu_ids or "<GPU_IDS>",
            },
        },
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "source": "llm_benchmark autotune",
        },
    }

    if config.max_num_batched_tokens:
        template["vllm"]["max_num_batched_tokens"] = config.max_num_batched_tokens

    if config.enforce_eager:
        template["vllm"]["enforce_eager"] = True

    template["command"] = _generate_vllm_command(config, model_path)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            yaml.dump(template, f, default_flow_style=False, sort_keys=False)

    return template


def _generate_vllm_command(config: TuningConfig, model_path: Optional[str] = None) -> str:
    """Generate vLLM launch command string."""
    parts = [
        "python -m vllm.entrypoints.openai.api_server",
        f"--model {model_path or '<MODEL_PATH>'}",
        f"--tensor-parallel-size {config.tensor_parallel}",
        f"--gpu-memory-utilization {config.gpu_memory_utilization}",
        f"--max-model-len {config.max_model_len}",
        f"--max-num-seqs {config.max_num_seqs}",
    ]

    if config.max_num_batched_tokens:
        parts.append(f"--max-num-batched-tokens {config.max_num_batched_tokens}")

    if config.enforce_eager:
        parts.append("--enforce-eager")

    return " \\\n  ".join(parts)


def generate_lb_config(
    config: TuningConfig,
    output_path: Optional[Path] = None,
    model_path: Optional[str] = None,
    gpu_ids: Optional[str] = None,
    port: int = 8000,
    lb_port: int = 9000,
) -> Dict[str, Any]:
    """Generate Load Balancer configuration from the best config."""

    gpu_list = gpu_ids.split(",") if gpu_ids else []

    lb_config = {
        "server": {
            "host": "0.0.0.0",
            "port": lb_port,
            "request_timeout": 180,
        },
        "scheduler": {
            "strategy": "least_load",
            "refresh_interval": 2,
        },
        "instances": [
            {
                "id": "vllm-optimized",
                "enabled": True,
                "managed": True,
                "host": "127.0.0.1",
                "port": port,
                "model": model_path or "<MODEL_PATH>",
                "tensor_parallel": config.tensor_parallel,
                "gpu_ids": gpu_ids,
                "gpu_memory_utilization": config.gpu_memory_utilization,
                "max_model_len": config.max_model_len,
                "extra_args": [
                    f"--max-num-seqs={config.max_num_seqs}",
                ],
            }
        ],
    }

    if config.max_num_batched_tokens:
        lb_config["instances"][0]["extra_args"].append(
            f"--max-num-batched-tokens={config.max_num_batched_tokens}"
        )

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            yaml.dump(lb_config, f, default_flow_style=False, sort_keys=False)

    return lb_config


def save_tuning_report(
    results: List[TuningResult],
    output_path: Path,
    search_space: Optional["SearchSpace"] = None,
    include_history: bool = True,
) -> Dict[str, Any]:
    """Save comprehensive tuning report."""

    # Get best result
    valid_results = [r for r in results if r.error is None]
    best_result = max(valid_results, key=lambda r: r.score) if valid_results else None

    # Build search space info from best config
    search_space_info = {}
    if results and results[0].config:
        config = results[0].config
        search_space_info = {
            "tuned_parameters": list(config.to_dict().keys()),
        }

    report = {
        "summary": {
            "total_trials": len(results),
            "successful_trials": len(valid_results),
            "best_trial_id": best_result.trial_id if best_result else None,
            "best_score": best_result.score if best_result else None,
            "completed_at": datetime.now().isoformat(),
        },
        "best_result": best_result.to_dict() if best_result else None,
        "search_space": search_space_info,
    }

    if include_history:
        report["history"] = [r.to_dict() for r in results]

    # Save JSON report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    # Also save CSV for easy analysis
    csv_path = output_path.with_suffix(".csv")
    save_history_csv(results, csv_path)

    return report


def save_history_csv(results: List[TuningResult], output_path: Path):
    """Save tuning history as CSV for analysis."""

    if not results:
        return

    fieldnames = [
        "trial_id",
        "score",
        "error",
        "gpu_memory_utilization",
        "tensor_parallel",
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
        "tps",
        "qps",
        "latency_p50",
        "latency_p90",
        "latency_p99",
        "ttft_p50",
        "ttft_p99",
        "success_rate",
        "constraint_violations",
    ]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            row = {
                "trial_id": r.trial_id,
                "score": r.score,
                "error": r.error or "",
                "gpu_memory_utilization": r.config.gpu_memory_utilization,
                "tensor_parallel": r.config.tensor_parallel,
                "max_model_len": r.config.max_model_len,
                "max_num_seqs": r.config.max_num_seqs,
                "max_num_batched_tokens": r.config.max_num_batched_tokens,
                "tps": r.tps,
                "qps": r.metrics.get("qps", 0),
                "latency_p50": r.metrics.get("latency_p50", 0),
                "latency_p90": r.metrics.get("latency_p90", 0),
                "latency_p99": r.latency_p99,
                "ttft_p50": r.metrics.get("ttft_p50", 0),
                "ttft_p99": r.metrics.get("ttft_p99", 0),
                "success_rate": r.success_rate,
                "constraint_violations": "; ".join(r.constraint_violations),
            }
            writer.writerow(row)


def generate_analysis_report(
    results: List[TuningResult],
    output_path: Path,
) -> Dict[str, Any]:
    """Generate detailed analysis report with parameter importance."""

    import statistics

    valid_results = [r for r in results if r.error is None]

    if not valid_results:
        return {"error": "No valid results to analyze"}

    # Basic statistics
    scores = [r.score for r in valid_results]
    tps_values = [r.tps for r in valid_results]
    latencies = [r.latency_p99 for r in valid_results]

    analysis = {
        "statistics": {
            "score": {
                "mean": statistics.mean(scores),
                "std": statistics.stdev(scores) if len(scores) > 1 else 0,
                "min": min(scores),
                "max": max(scores),
            },
            "throughput_tps": {
                "mean": statistics.mean(tps_values),
                "std": statistics.stdev(tps_values) if len(tps_values) > 1 else 0,
                "min": min(tps_values),
                "max": max(tps_values),
            },
            "latency_p99_ms": {
                "mean": statistics.mean(latencies),
                "std": statistics.stdev(latencies) if len(latencies) > 1 else 0,
                "min": min(latencies),
                "max": max(latencies),
            },
        },
        "parameter_analysis": _analyze_parameters(valid_results),
        "recommendations": _generate_recommendations(valid_results),
    }

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(analysis, f, indent=2)

    return analysis


def _analyze_parameters(results: List[TuningResult]) -> Dict[str, Any]:
    """Analyze parameter impact on performance."""

    param_impact = {}

    # Group by tensor_parallel
    tp_groups = {}
    for r in results:
        tp = r.config.tensor_parallel
        if tp not in tp_groups:
            tp_groups[tp] = []
        tp_groups[tp].append(r.score)

    if len(tp_groups) > 1:
        param_impact["tensor_parallel"] = {
            str(k): {
                "avg_score": sum(v) / len(v),
                "count": len(v),
            }
            for k, v in tp_groups.items()
        }

    # Group by gpu_memory_utilization (bin into ranges)
    mem_groups = {}
    for r in results:
        mem_bin = round(r.config.gpu_memory_utilization * 20) / 20  # 0.05 bins
        if mem_bin not in mem_groups:
            mem_groups[mem_bin] = []
        mem_groups[mem_bin].append(r.score)

    if len(mem_groups) > 1:
        param_impact["gpu_memory_utilization"] = {
            f"{k:.2f}": {
                "avg_score": sum(v) / len(v),
                "count": len(v),
            }
            for k, v in sorted(mem_groups.items())
        }

    return param_impact


def _generate_recommendations(results: List[TuningResult]) -> List[str]:
    """Generate recommendations based on tuning results."""

    recommendations = []
    best = max(results, key=lambda r: r.score)

    # Memory utilization recommendation
    best_mem = best.config.gpu_memory_utilization
    if best_mem >= 0.95:
        recommendations.append(
            "High GPU memory utilization (-0.95) achieved best results. "
            "Monitor for OOM errors in production."
        )
    elif best_mem <= 0.75:
        recommendations.append(
            "Lower GPU memory utilization worked best. "
            "Consider this for stability in multi-tenant environments."
        )

    # Tensor parallel recommendation
    best_tp = best.config.tensor_parallel
    if best_tp > 1:
        recommendations.append(
            f"Tensor parallelism of {best_tp} showed best performance. "
            "Ensure sufficient GPU communication bandwidth."
        )

    # Batch size recommendation
    best_seqs = best.config.max_num_seqs
    if best_seqs >= 200:
        recommendations.append(
            f"High max_num_seqs ({best_seqs}) optimal for throughput. "
            "May increase latency variance at high load."
        )

    return recommendations
