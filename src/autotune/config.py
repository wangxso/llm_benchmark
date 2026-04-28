"""Configuration classes for Auto-Tuning Agent."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class Objective(Enum):
    """Optimization objective."""
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    BALANCED = "balanced"


@dataclass
class ParameterRange:
    """Defines the search range for a single parameter."""
    name: str
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    step: Optional[float] = None  # For discretized continuous values
    values: Optional[List[Any]] = None  # For categorical/discrete values
    param_type: str = "float"  # "float", "int", "categorical"

    def __post_init__(self):
        if self.values is not None:
            self.param_type = "categorical"
        elif self.step is not None and self.step == int(self.step):
            self.param_type = "int"
        elif self.min_val is not None and self.max_val is not None:
            self.param_type = "float"

    def sample_value(self, trial) -> Any:
        """Sample a value using optuna trial."""
        if self.values is not None:
            return trial.suggest_categorical(self.name, self.values)
        elif self.param_type == "int":
            return trial.suggest_int(
                self.name,
                int(self.min_val),
                int(self.max_val),
                step=int(self.step) if self.step else None
            )
        else:
            return trial.suggest_float(
                self.name,
                self.min_val,
                self.max_val,
                step=self.step
            )


@dataclass
class SearchSpace:
    """Defines the complete search space for tuning."""
    parameters: List[ParameterRange]
    constraints: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # Constraint thresholds
    DEFAULT_CONSTRAINTS = {
        "latency_p99_ms": (0, 2000),
        "success_rate": (0.90, 1.0),
        "ttft_p99_ms": (0, 5000),
    }

    def __post_init__(self):
        if not self.constraints:
            self.constraints = self.DEFAULT_CONSTRAINTS.copy()

    def validate_result(self, metrics: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Check if metrics satisfy constraints."""
        violations = []
        for constraint_name, (min_val, max_val) in self.constraints.items():
            value = metrics.get(constraint_name, 0)
            if value < min_val or value > max_val:
                violations.append(
                    f"{constraint_name}={value:.2f} not in [{min_val}, {max_val}]"
                )
        return len(violations) == 0, violations


@dataclass
class TuningConfig:
    """Configuration for a single tuning trial."""
    # vLLM parameters
    gpu_memory_utilization: float = 0.85
    tensor_parallel: int = 1
    max_model_len: int = 4096
    max_num_seqs: int = 128
    max_num_batched_tokens: int = 2048

    # Advanced vLLM parameters (optional)
    enforce_eager: bool = False
    gpu_memory_utilization_kv_cache: Optional[float] = None

    # Load test parameters
    concurrency: int = 100
    duration: int = 60
    warmup_duration: int = 5

    def to_vllm_args(self) -> Dict[str, Any]:
        """Convert to vLLM command line arguments."""
        args = {
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "tensor_parallel_size": self.tensor_parallel,
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
        }
        if self.max_num_batched_tokens:
            args["max_num_batched_tokens"] = self.max_num_batched_tokens
        if self.enforce_eager:
            args["enforce_eager"] = True
        return args

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "tensor_parallel": self.tensor_parallel,
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "enforce_eager": self.enforce_eager,
            "concurrency": self.concurrency,
            "duration": self.duration,
        }


@dataclass
class TuningResult:
    """Result of a single tuning trial."""
    trial_id: int
    config: TuningConfig
    metrics: Dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    objective: str = "throughput"
    error: Optional[str] = None
    constraint_violations: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Key metrics shortcuts
    @property
    def tps(self) -> float:
        return self.metrics.get("tps", 0.0)

    @property
    def latency_p99(self) -> float:
        return self.metrics.get("latency_p99_ms", 0.0)

    @property
    def success_rate(self) -> float:
        return self.metrics.get("success_rate", 0.0)

    def calculate_score(self, objective: Objective) -> float:
        """Calculate optimization score based on objective."""
        if self.error:
            return float("-inf")

        if objective == Objective.THROUGHPUT:
            self.score = self.tps
        elif objective == Objective.LATENCY:
            self.score = -self.latency_p99 if self.latency_p99 > 0 else float("-inf")
        else:  # BALANCED
            if self.latency_p99 > 0:
                self.score = self.tps * 100 / self.latency_p99
            else:
                self.score = float("-inf")

        return self.score

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trial_id": self.trial_id,
            "config": self.config.to_dict(),
            "metrics": self.metrics,
            "score": self.score,
            "objective": self.objective,
            "error": self.error,
            "constraint_violations": self.constraint_violations,
            "timestamp": self.timestamp,
        }


# Predefined search spaces for common scenarios
def get_default_vllm_space(gpu_count: int = 1) -> SearchSpace:
    """Get default search space for vLLM tuning."""
    tp_values = [1]
    if gpu_count >= 2:
        tp_values = [1, 2]
    if gpu_count >= 4:
        tp_values = [1, 2, 4]
    if gpu_count >= 8:
        tp_values = [1, 2, 4, 8]

    return SearchSpace(
        parameters=[
            ParameterRange(
                name="gpu_memory_utilization",
                min_val=0.7,
                max_val=0.95,
                step=0.05,
            ),
            ParameterRange(
                name="tensor_parallel",
                values=tp_values,
            ),
            ParameterRange(
                name="max_model_len",
                values=[4096, 8192, 16384, 32768],
            ),
            ParameterRange(
                name="max_num_seqs",
                min_val=32,
                max_val=256,
                step=32,
            ),
        ],
        constraints={
            "latency_p99_ms": (0, 2000),
            "success_rate": (0.90, 1.0),
        }
    )


def get_high_throughput_space(gpu_count: int = 1) -> SearchSpace:
    """Search space optimized for high throughput scenarios."""
    return SearchSpace(
        parameters=[
            ParameterRange(
                name="gpu_memory_utilization",
                min_val=0.85,
                max_val=0.98,
                step=0.05,
            ),
            ParameterRange(
                name="tensor_parallel",
                values=[max(1, gpu_count // 2), gpu_count] if gpu_count > 1 else [1],
            ),
            ParameterRange(
                name="max_model_len",
                values=[2048, 4096, 8192],
            ),
            ParameterRange(
                name="max_num_seqs",
                min_val=128,
                max_val=512,
                step=64,
            ),
        ],
        constraints={
            "latency_p99_ms": (0, 3000),
            "success_rate": (0.85, 1.0),
        }
    )


def get_low_latency_space(gpu_count: int = 1) -> SearchSpace:
    """Search space optimized for low latency scenarios."""
    return SearchSpace(
        parameters=[
            ParameterRange(
                name="gpu_memory_utilization",
                min_val=0.7,
                max_val=0.9,
                step=0.05,
            ),
            ParameterRange(
                name="tensor_parallel",
                values=[gpu_count] if gpu_count > 0 else [1],
            ),
            ParameterRange(
                name="max_model_len",
                values=[2048, 4096, 8192],
            ),
            ParameterRange(
                name="max_num_seqs",
                min_val=16,
                max_val=64,
                step=16,
            ),
        ],
        constraints={
            "latency_p99_ms": (0, 500),
            "success_rate": (0.95, 1.0),
        }
    )
