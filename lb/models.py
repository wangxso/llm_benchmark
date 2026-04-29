from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from src.device.monitor import get_gpu_utilization


@dataclass
class InstanceConfig:
    id: str
    enabled: bool = True
    managed: bool = True
    host: str = "127.0.0.1"
    port: int = 8000
    model: str = ""
    tensor_parallel: int = 1
    gpu_ids: Optional[str] = None
    device: str = "nvidia"
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 4096
    enable_mfu_metrics: bool = True
    extra_args: List[str] = field(default_factory=list)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class InstanceState:
    config: InstanceConfig
    healthy: bool = False
    running: bool = False
    pid: Optional[int] = None
    inflight_requests: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    models: List[Dict[str, Any]] = field(default_factory=list)
    last_error: str = ""
    last_health_check: float = 0.0
    last_started_at: Optional[float] = None
    log_path: Optional[str] = None
    command: List[str] = field(default_factory=list)
    exit_code: Optional[int] = None

    def supports_model(self, requested_model: Optional[str]) -> bool:
        if not requested_model:
            return True

        if requested_model == self.config.model:
            return True

        if not self.models:
            return False

        for item in self.models:
            if item.get("id") == requested_model:
                return True

        return False

    def key_metrics(self) -> Dict[str, float]:
        return {
            "running_requests": self.metrics.get("vllm:num_requests_running", 0.0),
            "waiting_requests": self.metrics.get("vllm:num_requests_waiting", 0.0),
            "batch_size": self.metrics.get("vllm:batch_size", 0.0),
            "kv_cache_usage": self.metrics.get(
                "vllm:kv_cache_usage_perc", self.metrics.get("vllm:kv_cache_usage", 0.0)
            ),
            "gpu_utilization": get_gpu_utilization(),  # Multi-vendor GPU utilization
            "actual_tflops_per_second": self.metrics.get(
                "vllm:actual_tflops_per_second", 0.0
            ),
            # Token throughput
            "prompt_tokens_total": self.metrics.get("vllm:prompt_tokens_total", 0.0),
            "generation_tokens_total": self.metrics.get("vllm:generation_tokens_total", 0.0),
            "time_to_first_token": self.metrics.get("vllm:time_to_first_token_seconds", 0.0),
            "time_per_output_token": self.metrics.get("vllm:time_per_output_token_seconds", 0.0),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": asdict(self.config),
            "healthy": self.healthy,
            "running": self.running,
            "pid": self.pid,
            "inflight_requests": self.inflight_requests,
            "metrics": self.metrics,
            "key_metrics": self.key_metrics(),
            "models": self.models,
            "last_error": self.last_error,
            "last_health_check": self.last_health_check,
            "last_started_at": self.last_started_at,
            "log_path": self.log_path,
            "command": self.command,
            "exit_code": self.exit_code,
        }
