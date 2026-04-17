from __future__ import annotations

from typing import Iterable, Optional

from .models import InstanceState


class NoHealthyInstanceError(RuntimeError):
    pass


class LeastLoadScheduler:
    def __init__(self, queue_weight: float = 2.0, inflight_weight: float = 1.0):
        self.queue_weight = queue_weight
        self.inflight_weight = inflight_weight
        self._fallback_counter = 0

    def select_instance(
        self, instances: Iterable[InstanceState], requested_model: Optional[str] = None
    ) -> InstanceState:
        candidates = [
            item
            for item in instances
            if item.config.enabled and item.healthy and item.supports_model(requested_model)
        ]
        if not candidates:
            raise NoHealthyInstanceError("No healthy backend instance available")

        with_metrics = [item for item in candidates if self._has_metrics(item)]
        if with_metrics:
            return min(
                with_metrics,
                key=lambda item: (
                    self._score(item),
                    item.inflight_requests,
                    item.config.id,
                ),
            )

        ordered = sorted(candidates, key=lambda item: (item.inflight_requests, item.config.id))
        selected = ordered[self._fallback_counter % len(ordered)]
        self._fallback_counter = (self._fallback_counter + 1) % len(ordered)
        return selected

    def _has_metrics(self, instance: InstanceState) -> bool:
        running = instance.metrics.get("vllm:num_requests_running")
        waiting = instance.metrics.get("vllm:num_requests_waiting")
        return running is not None or waiting is not None

    def _score(self, instance: InstanceState) -> float:
        running = float(instance.metrics.get("vllm:num_requests_running") or 0.0)
        waiting = float(instance.metrics.get("vllm:num_requests_waiting") or 0.0)
        inflight = float(instance.inflight_requests)
        return running + self.queue_weight * waiting + self.inflight_weight * inflight
