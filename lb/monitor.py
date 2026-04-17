from __future__ import annotations

import time
from collections import deque
from typing import Any, Deque, Dict, Iterable

from .models import InstanceState


class ProxyMonitor:
    def __init__(self, window_seconds: int = 60):
        self.window_seconds = window_seconds
        self.total_requests = 0
        self.total_success = 0
        self.total_errors = 0
        self._events: Deque[Dict[str, Any]] = deque()
        # Token tracking for throughput calculation
        self._last_token_check: Dict[str, Dict[str, float]] = {}
        self.prefill_tps: float = 0.0
        self.decode_tps: float = 0.0

    def _trim(self) -> None:
        """Remove old events outside the window"""
        cutoff = time.time() - self.window_seconds
        while self._events and self._events[0]["timestamp"] < cutoff:
            self._events.popleft()

    def record(self, instance_id: str, success: bool, latency_ms: float) -> None:
        event = {
            "timestamp": time.time(),
            "instance_id": instance_id,
            "success": success,
            "latency_ms": latency_ms,
        }
        self.total_requests += 1
        if success:
            self.total_success += 1
        else:
            self.total_errors += 1
        self._events.append(event)
        self._trim()

    def update_token_metrics(self, instances: Iterable[InstanceState]) -> None:
        """Update token throughput metrics from instance metrics"""
        now = time.time()
        total_prefill_tokens = 0
        total_decode_tokens = 0

        for instance in instances:
            if not instance.healthy:
                continue
            km = instance.key_metrics()
            inst_id = instance.config.id
            prompt_tokens = km.get("prompt_tokens_total", 0)
            gen_tokens = km.get("generation_tokens_total", 0)

            if inst_id not in self._last_token_check:
                self._last_token_check[inst_id] = {
                    "prefill": prompt_tokens,
                    "decode": gen_tokens,
                    "time": now,
                }

            last = self._last_token_check[inst_id]
            dt = now - last["time"]
            if dt > 0:
                total_prefill_tokens += prompt_tokens - last["prefill"]
                total_decode_tokens += gen_tokens - last["decode"]

            self._last_token_check[inst_id] = {
                "prefill": prompt_tokens,
                "decode": gen_tokens,
                "time": now,
            }

    def snapshot(self, instances: Iterable[InstanceState]) -> Dict[str, Any]:
        self._trim()
        self.update_token_metrics(instances)

        events = list(self._events)
        per_instance: Dict[str, Dict[str, Any]] = {}

        for instance in instances:
            recent = [e for e in events if e["instance_id"] == instance.config.id]
            latencies = [e["latency_ms"] for e in recent]
            successes = sum(1 for e in recent if e["success"])
            errors = len(recent) - successes
            km = instance.key_metrics()

            per_instance[instance.config.id] = {
                "recent_requests": len(recent),
                "recent_success": successes,
                "recent_errors": errors,
                "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
                "inflight_requests": instance.inflight_requests,
                "healthy": instance.healthy,
                "running": instance.running,
                "key_metrics": km,
                # Throughput metrics
                "prefill_tokens": km.get("prompt_tokens_total", 0),
                "decode_tokens": km.get("generation_tokens_total", 0),
            }

        overall_success_rate = (
            self.total_success / self.total_requests if self.total_requests else 0.0
        )
        recent_successes = sum(1 for event in events if event["success"])
        recent_avg_latency = (
            sum(event["latency_ms"] for event in events) / len(events) if events else 0.0
        )

        return {
            "window_seconds": self.window_seconds,
            "total_requests": self.total_requests,
            "total_success": self.total_success,
            "total_errors": self.total_errors,
            "success_rate": overall_success_rate,
            "recent_qps": len(events) / self.window_seconds if self.window_seconds else 0.0,
            "recent_success_rate": recent_successes / len(events) if events else 0.0,
            "recent_avg_latency_ms": recent_avg_latency,
            # Token throughput
            "prefill_tps": self.prefill_tps,
            "decode_tps": self.decode_tps,
            "instances": per_instance,
        }
