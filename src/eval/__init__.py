"""LLM Benchmark Evaluation Module"""

from .registry import register_benchmark, get_benchmark, list_benchmarks
from .runner import EvalRunner
from .base import BaseBenchmark

# Import datasets to trigger registration
from . import datasets

__all__ = [
    "register_benchmark",
    "get_benchmark",
    "list_benchmarks",
    "EvalRunner",
    "BaseBenchmark",
]
