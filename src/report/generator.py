import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime


class ReportGenerator:
    """Report generator for benchmark results"""

    def __init__(self, config: Dict):
        self.config = config
        self.percentiles = config.get("metrics", {}).get("percentiles", [50, 90, 99])

    def generate(self, load_results: Dict, metrics_data: Dict = None) -> Dict[str, Any]:
        """Generate benchmark report"""
        vllm_metrics = {}

        if metrics_data and metrics_data.get("vllm_metrics"):
            vllm_metrics = self._extract_vllm_metrics(metrics_data["vllm_metrics"])

        report = {
            "test_info": {
                "scenario": self.config.get("scenario", {}).get("name", "default"),
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "start_time": datetime.now().isoformat(),
                "model": self.config.get("vllm", {}).get("model"),
                "backend": "vllm",
                "tensor_parallel": self.config.get("vllm", {}).get(
                    "tensor_parallel", 1
                ),
            },
            "load_config": {
                "type": self.config.get("load", {}).get("type", "fixed"),
                "concurrency": self.config.get("load", {}).get("base_concurrency", 100),
                "duration": self.config.get("load", {}).get("duration", 300),
            },
            "metrics": {
                "qps": load_results.get("qps", 0),
                "tps": load_results.get("tps", 0),
                "ttft_ms": load_results.get("ttft_p50", 0),
                "tpot_ms": load_results.get("tpot_p50", 0),
                "latency_p50_ms": load_results.get("latency_p50", 0),
                "latency_p90_ms": load_results.get("latency_p90", 0),
                "latency_p99_ms": load_results.get("latency_p99", 0),
                "success_rate": load_results.get("success_rate", 0),
                "error_rate": load_results.get("error_rate", 0),
            },
            "vllm_metrics": vllm_metrics,
            "request_stats": {
                "total_requests": load_results.get("total_requests", 0),
                "successful_requests": load_results.get("successful_requests", 0),
                "failed_requests": load_results.get("failed_requests", 0),
                "avg_output_tokens": load_results.get("avg_output_tokens", 0),
            },
        }

        if metrics_data and metrics_data.get("snapshots"):
            report["time_series"] = self._generate_time_series(
                metrics_data["snapshots"]
            )

        return report

    def _extract_vllm_metrics(self, vllm_metrics: Dict) -> Dict[str, Any]:
        """Extract key vLLM metrics"""
        result = {}

        key_mapping = {
            "vllm:num_requests_running": "active_requests",
            "vllm:num_requests_waiting": "waiting_requests",
            "vllm:batch_size": "batch_size",
            "vllm:kv_cache_usage": "kv_cache_usage",
            "vllm:prefill_latency": "prefill_latency_ms",
            "vllm:decode_latency": "decode_latency_ms",
            "vllm:gpu_utilization": "gpu_utilization",
            "vllm:token_issues_per_second": "token_ips",
            "vllm:num_prefill_tokens": "prefill_tokens",
            "vllm:num_decode_tokens": "decode_tokens",
        }

        for metric_name, display_name in key_mapping.items():
            if metric_name in vllm_metrics:
                result[display_name] = vllm_metrics[metric_name]

        return result

    def _generate_time_series(self, snapshots: List[Dict]) -> List[Dict]:
        """Generate time series data"""
        time_series = []

        for snap in snapshots:
            time_series.append(
                {
                    "timestamp": datetime.fromtimestamp(snap["timestamp"]).strftime(
                        "%H:%M:%S"
                    ),
                    "qps": snap.get("qps", 0),
                    "tps": snap.get("tps", 0),
                    "latency_p50": snap.get("p50_latency", 0),
                    "latency_p99": snap.get("p99_latency", 0),
                    "ttft": snap.get("ttft", 0),
                    "tpot": snap.get("tpot", 0),
                    "active_requests": snap.get("active_requests", 0),
                    "success_rate": snap.get("success_rate", 0),
                }
            )

        return time_series

    def save(self, report: Dict, path: str):
        """Save report to file"""
        with open(path, "w") as f:
            json.dump(report, f, indent=2)

    def save_json(self, report: Dict, path: str):
        """Save report as JSON lines"""
        with open(path, "w") as f:
            json.dump(report, f, indent=2)

    def analyze_bottleneck(self, report: Dict) -> Dict[str, Any]:
        """Analyze performance bottleneck"""
        bottleneck = {
            "identified": False,
            "type": "",
            "details": "",
            "recommendations": [],
        }

        metrics = report.get("metrics", {})
        vllm_metrics = report.get("vllm_metrics", {})

        if metrics.get("latency_p99", 0) > metrics.get("latency_p50", 0) * 3:
            bottleneck["identified"] = True
            bottleneck["type"] = "high_latency_variance"
            bottleneck["details"] = "P99 latency is 3x+ higher than P50"
            bottleneck["recommendations"].extend(
                [
                    "Check for request queuing",
                    "Increase batch size",
                    "Review KV Cache configuration",
                ]
            )

        if vllm_metrics.get("waiting_requests", 0) > 10:
            bottleneck["identified"] = True
            bottleneck["type"] = "request_queue_overflow"
            bottleneck["details"] = (
                f"High queue depth: {vllm_metrics.get('waiting_requests')}"
            )
            bottleneck["recommendations"].extend(
                [
                    "Increase worker threads",
                    "Reduce request timeout",
                    "Scale inference instances",
                ]
            )

        if vllm_metrics.get("kv_cache_usage", 0) > 0.95:
            bottleneck["identified"] = True
            bottleneck["type"] = "kv_cache_pressure"
            bottleneck["details"] = "KV Cache nearly full"
            bottleneck["recommendations"].extend(
                [
                    "Increase KV Cache size",
                    "Reduce max batch size",
                    "Enable KV Cache offload",
                ]
            )

        return bottleneck

    def generate_summary(self, report: Dict) -> str:
        """Generate text summary"""
        metrics = report.get("metrics", {})

        lines = [
            "=" * 50,
            "BENCHMARK SUMMARY",
            "=" * 50,
            f"Model: {report.get('test_info', {}).get('model')}",
            f"Backend: {report.get('test_info', {}).get('backend')}",
            "",
            "[Throughput]",
            f"  QPS: {metrics.get('qps', 0):.2f} req/s",
            f"  TPS: {metrics.get('tps', 0):.2f} tokens/s",
            "",
            "[Latency]",
            f"  TTFT: {metrics.get('ttft_ms', 0):.2f} ms",
            f"  TPOT: {metrics.get('tpot_ms', 0):.2f} ms",
            f"  P50: {metrics.get('latency_p50_ms', 0):.2f} ms",
            f"  P90: {metrics.get('latency_p90_ms', 0):.2f} ms",
            f"  P99: {metrics.get('latency_p99_ms', 0):.2f} ms",
            "",
            "[Reliability]",
            f"  Success: {metrics.get('success_rate', 0) * 100:.2f}%",
            f"  Errors: {metrics.get('error_rate', 0) * 100:.2f}%",
            "=" * 50,
        ]

        return "\n".join(lines)
