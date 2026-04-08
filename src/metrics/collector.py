import time
import threading
from typing import Callable, Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import deque


@dataclass
class MetricsSnapshot:
    """Single metrics snapshot"""

    timestamp: float
    qps: float = 0
    tps: float = 0
    active_requests: int = 0
    queue_size: int = 0
    avg_latency: float = 0
    p50_latency: float = 0
    p90_latency: float = 0
    p99_latency: float = 0
    ttft: float = 0
    tpot: float = 0
    success_rate: float = 0
    error_rate: float = 0
    vllm_metrics: Dict[str, float] = field(default_factory=dict)


class MetricsCollector:
    """Metrics collector for real-time monitoring"""

    def __init__(self, config: Dict):
        self.config = config
        self.enabled = config.get("metrics", {}).get("enabled", True)
        self.interval = config.get("metrics", {}).get("collection_interval", 1)

        self.host = config.get("vllm", {}).get("host", "localhost")
        self.port = config.get("vllm", {}).get("port", 8000)

        self._snapshots: deque = deque(maxlen=3600)
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._results: List[Dict] = []
        self._results_provider: Optional[Callable[[], List[Dict]]] = None
        self._start_time: float = 0

    def start(self):
        """Start metrics collection"""
        if not self.enabled:
            return

        self._start_time = time.time()
        self._running = True
        self._thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop metrics collection"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def set_results(self, results: List[Dict] | Callable[[], List[Dict]]):
        """Set results or live results provider for time-series computation"""
        if callable(results):
            self._results_provider = results
            self._results = []
        else:
            self._results_provider = None
            self._results = results

    def _collect_loop(self):
        """Main collection loop"""
        while self._running:
            try:
                snapshot = self._collect()
                self._snapshots.append(snapshot)
            except Exception as e:
                print(f"[MetricsCollector] Collection error: {e}")

            time.sleep(self.interval)

    def _collect(self) -> MetricsSnapshot:
        """Collect current metrics from results"""
        import requests

        snapshot = MetricsSnapshot(timestamp=time.time())

        results = self._results_provider() if self._results_provider else self._results
        if not results:
            return snapshot

        current_time = time.time()
        elapsed = current_time - self._start_time
        window = 1.0

        recent_results = [
            r for r in results if current_time - r.get("end_time", 0) <= window
        ]

        if recent_results and elapsed > 0.5:
            successful = [r for r in recent_results if r.get("success", False)]
            total_tokens = sum(r.get("output_tokens", 0) for r in successful)
            latencies = [r.get("total_latency", 0) for r in successful]
            ttfts = [r.get("ttft", 0) for r in successful if r.get("ttft", 0) > 0]
            tpots = [r.get("tpot", 0) for r in successful if r.get("tpot", 0) > 0]

            snapshot.qps = len(successful) / window if window > 0 else 0
            snapshot.tps = total_tokens / window if window > 0 else 0
            snapshot.active_requests = len(recent_results)

            if latencies:
                sorted_lat = sorted(latencies)
                snapshot.avg_latency = sum(latencies) / len(latencies)
                snapshot.p50_latency = self._percentile(sorted_lat, 50)
                snapshot.p90_latency = self._percentile(sorted_lat, 90)
                snapshot.p99_latency = self._percentile(sorted_lat, 99)

            if ttfts:
                snapshot.ttft = sum(ttfts) / len(ttfts)
            if tpots:
                snapshot.tpot = sum(tpots) / len(tpots)

        all_results = [
            r for r in results if r.get("start_time", 0) >= self._start_time
        ]
        if all_results:
            successful = [r for r in all_results if r.get("success", False)]
            failed = [r for r in all_results if not r.get("success", False)]
            total = len(all_results)
            if total > 0:
                snapshot.success_rate = len(successful) / total
                snapshot.error_rate = len(failed) / total

        try:
            response = requests.get(
                f"http://{self.host}:{self.port}/metrics", timeout=2
            )
            if response.status_code == 200:
                snapshot.vllm_metrics = self._parse_prometheus(response.text)
        except Exception:
            pass

        return snapshot

    def _parse_prometheus(self, text: str) -> Dict[str, float]:
        """Parse Prometheus metrics"""
        metrics = {}

        for line in text.split("\n"):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            if " " in line:
                parts = line.split(" ", 1)
                raw_name = parts[0]
                name = raw_name.split("{", 1)[0]
                value = parts[1] if len(parts) > 1 else "0"

                try:
                    metrics[name] = float(value)
                except ValueError:
                    continue

        return metrics

    def _percentile(self, sorted_list: List[float], percentile: int) -> float:
        """Calculate percentile"""
        if not sorted_list:
            return 0

        index = int(len(sorted_list) * percentile / 100)
        index = min(index, len(sorted_list) - 1)
        return sorted_list[index]

    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics"""
        snapshots = list(self._snapshots)

        if not snapshots:
            return {}

        vllm_metrics = {}

        for snap in snapshots:
            if snap.vllm_metrics:
                for key, value in snap.vllm_metrics.items():
                    if key not in vllm_metrics:
                        vllm_metrics[key] = []
                    vllm_metrics[key].append(value)

        avg_vllm = {}
        for key, values in vllm_metrics.items():
            avg_vllm[key] = sum(values) / len(values) if values else 0

        total_flops_metric = "vllm:estimated_flops_per_gpu_total"
        total_flops_values = vllm_metrics.get(total_flops_metric, [])
        gpu_count = self.config.get("vllm", {}).get("tensor_parallel", 1)
        if total_flops_values and len(snapshots) >= 2:
            window_seconds = snapshots[-1].timestamp - snapshots[0].timestamp
            if window_seconds > 0:
                total_flops = max(total_flops_values)
                actual_flops_per_second = (total_flops / window_seconds) * gpu_count
                avg_vllm[total_flops_metric] = total_flops
                avg_vllm["vllm:actual_flops_per_second"] = actual_flops_per_second
                avg_vllm["vllm:actual_tflops_per_second"] = (
                    actual_flops_per_second / 1e12
                )

        return {
            "snapshots": [
                {
                    "timestamp": s.timestamp,
                    "qps": s.qps,
                    "tps": s.tps,
                    "active_requests": s.active_requests,
                    "avg_latency": s.avg_latency,
                    "p50_latency": s.p50_latency,
                    "p90_latency": s.p90_latency,
                    "p99_latency": s.p99_latency,
                    "ttft": s.ttft,
                    "tpot": s.tpot,
                    "success_rate": s.success_rate,
                    "error_rate": s.error_rate,
                }
                for s in snapshots
            ],
            "vllm_metrics": avg_vllm,
        }

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot"""
        if self._snapshots:
            snap = self._snapshots[-1]
            return {
                "qps": snap.qps,
                "tps": snap.tps,
                "active_requests": snap.active_requests,
                "latency_p99": snap.p99_latency,
                "ttft": snap.ttft,
                "tpot": snap.tpot,
                "success_rate": snap.success_rate,
            }
        return {}
