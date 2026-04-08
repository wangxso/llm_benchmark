import yaml
from pathlib import Path
from typing import Any, Dict


DEFAULT_CONFIG = {
    "vllm": {
        "host": "localhost",
        "port": 8000,
        "model": "./models/Qwen3.5-0.8B",
        "tensor_parallel": 1,
        "gpu_memory_utilization": 0.8,
    },
    "scenario": {
        "name": "default-benchmark",
    },
    "load": {
        "type": "fixed",
        "base_concurrency": 100,
        "duration": 300,
        "warmup_duration": 10,
    },
    "dataset": {
        "mode": "generate",
        "generate": {
            "short_ratio": 0.7,
            "long_ratio": 0.3,
            "max_input_len": 4096,
            "max_output_len": 2048,
        },
    },
    "request": {
        "stream": False,
        "temperature": 0.7,
        "max_tokens": 1024,
        "timeout": 120,
    },
    "metrics": {
        "enabled": True,
        "collection_interval": 1,
        "percentiles": [50, 90, 99],
    },
    "output": {
        "path": "./results",
    },
}


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from file or use defaults"""
    config = DEFAULT_CONFIG.copy()

    if config_path:
        path = Path(config_path)
        if path.exists():
            with open(path, "r") as f:
                user_config = yaml.safe_load(f)
                config = deep_merge(config, user_config)
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")

    return config


def deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries"""
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def validate_config(config: Dict) -> bool:
    """Validate configuration"""
    required_keys = ["vllm", "load"]

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    if config["load"].get("type") == "step":
        if not config["load"].get("step_increment"):
            raise ValueError("step_increment required for step load")

    return True
