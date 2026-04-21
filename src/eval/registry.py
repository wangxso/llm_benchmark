"""Benchmark registry for managing evaluation datasets"""

from typing import Dict, Type, List

_BENCHMARKS: Dict[str, Type] = {}


def register_benchmark(name: str):
    """Decorator to register a benchmark class"""
    def decorator(cls):
        _BENCHMARKS[name] = cls
        return cls
    return decorator


def get_benchmark(name: str):
    """Get a benchmark class by name"""
    if name not in _BENCHMARKS:
        available = ", ".join(list_benchmarks())
        raise ValueError(f"Unknown benchmark: {name}. Available: {available}")
    return _BENCHMARKS[name]


def list_benchmarks() -> List[str]:
    """List all registered benchmarks"""
    return sorted(_BENCHMARKS.keys())
