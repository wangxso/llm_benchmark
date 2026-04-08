import time
import threading
import asyncio
from typing import Dict, Any, List, Optional
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

        self._request_count = 0
        self._success_count = 0
        self._error_count = 0
        self._total_tokens = 0

        self._latencies: List[float] = []
        self._ttfts: List[float] = []
        self._tpots: List[float] = []

        self._current_qps = 0
        self._current_tps = 0
        self._active_requests = 0

    def start(self):
        """Start metrics collection"""
        if not self.enabled:
            return

        self._running = True
        self._thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop metrics collection"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

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
        """Collect current metrics"""
        import requests

        snapshot = MetricsSnapshot(timestamp=time.time())

        snapshot.qps = self._current_qps
        snapshot.tps = self._current_tps
        snapshot.active_requests = self._active_requests

        if self._latencies:
            sorted_latencies = sorted(self._latencies)
            snapshot.avg_latency = sum(self._latencies) / len(self._latencies)
            snapshot.p50_latency = self._percentile(sorted_latencies, 50)
            snapshot.p90_latency = self._percentile(sorted_latencies, 90)
            snapshot.p99_latency = self._percentile(sorted_latencies, 99)

        if self._ttfts:
            snapshot.ttft = sum(self._ttfts) / len(self._ttfts)

        if self._tpots:
            snapshot.tpot = sum(self._tpots) / len(self._tpots)

        total = self._success_count + self._error_count
        if total > 0:
            snapshot.success_rate = self._success_count / total
            snapshot.error_rate = self._error_count / total

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
                name = parts[0]
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

    def record_request(
        self,
        latency: float,
        ttft: float = 0,
        tpot: float = 0,
        tokens: int = 0,
        success: bool = True,
    ):
        """Record a completed request"""
        self._request_count += 1

        if success:
            self._success_count += 1
            self._latencies.append(latency)
            if ttft > 0:
                self._ttfts.append(ttft)
            if tpot > 0:
                self._tpots.append(tpot)
            self._total_tokens += tokens
        else:
            self._error_count += 1

        if len(self._latencies) > 10000:
            self._latencies = self._latencies[-5000:]
        if len(self._ttfts) > 10000:
            self._ttfts = self._ttfts[-5000:]
        if len(self._tpots) > 10000:
            self._tpots = self._tpots[-5000:]

    def update_qps(self, qps: float, tps: float, active: int):
        """Update current QPS/TPS"""
        self._current_qps = qps
        self._current_tps = tps
        self._active_requests = active

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
