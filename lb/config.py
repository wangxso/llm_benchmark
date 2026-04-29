from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import yaml

from src.config import deep_merge

from .models import InstanceConfig
from src.device import get_device_profile


INSTANCE_DEFAULTS = {
    "enabled": True,
    "managed": True,
    "host": "127.0.0.1",
    "tensor_parallel": 1,
    "gpu_ids": None,
    "device": "nvidia",
    "gpu_memory_utilization": 0.85,
    "max_model_len": 4096,
    "enable_mfu_metrics": True,
    "extra_args": [],
}

DEFAULT_CONFIG = {
    "server": {
        "host": "0.0.0.0",
        "port": 9000,
        "request_timeout": 180,
    },
    "scheduler": {
        "strategy": "least_load",
        "refresh_interval": 2,
        "queue_weight": 2.0,
        "inflight_weight": 1.0,
    },
    "instances": [],
    "ui": {
        "enabled": True,
        "title": "vLLM Load Balancer",
    },
    "runtime": {
        "log_dir": "./lb/runtime/logs",
        "state_path": "./lb/runtime/state.json",
    },
}


def default_config_path() -> Path:
    return Path(__file__).resolve().parent / "config" / "default.yaml"


def load_config(config_path: str | None = None) -> Dict[str, Any]:
    path = Path(config_path) if config_path else default_config_path()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    if not isinstance(raw_config, dict):
        raise ValueError("Config root must be a mapping")

    config = deep_merge(deepcopy(DEFAULT_CONFIG), raw_config)
    return normalize_config(config)


def parse_config_text(text: str) -> Dict[str, Any]:
    raw_config = yaml.safe_load(text) or {}
    if not isinstance(raw_config, dict):
        raise ValueError("Config root must be a mapping")

    config = deep_merge(deepcopy(DEFAULT_CONFIG), raw_config)
    return normalize_config(config)


def save_config_text(text: str, config_path: str | Path) -> Dict[str, Any]:
    config = parse_config_text(text)
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return config


def dump_config(config: Dict[str, Any]) -> str:
    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True)


def build_instance_configs(config: Dict[str, Any]) -> List[InstanceConfig]:
    return [InstanceConfig(**item) for item in config.get("instances", [])]


def normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    normalized = deepcopy(config)
    instances = []

    for index, item in enumerate(normalized.get("instances", [])):
        if not isinstance(item, dict):
            raise ValueError(f"Instance at index {index} must be a mapping")

        merged = deep_merge(deepcopy(INSTANCE_DEFAULTS), item)
        instances.append(merged)

    normalized["instances"] = instances
    validate_config(normalized)
    return normalized


def validate_config(config: Dict[str, Any]) -> bool:
    server = config.get("server", {})
    scheduler = config.get("scheduler", {})
    instances = config.get("instances", [])

    port = server.get("port")
    if not isinstance(port, int) or port <= 0 or port > 65535:
        raise ValueError("server.port must be an integer between 1 and 65535")

    timeout = server.get("request_timeout")
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError("server.request_timeout must be positive")

    if scheduler.get("strategy") != "least_load":
        raise ValueError("scheduler.strategy must be 'least_load'")

    refresh_interval = scheduler.get("refresh_interval")
    if not isinstance(refresh_interval, (int, float)) or refresh_interval <= 0:
        raise ValueError("scheduler.refresh_interval must be positive")

    if not instances:
        raise ValueError("At least one instance must be configured")

    seen_ids = set()
    seen_bindings = set()

    for item in instances:
        instance_id = item.get("id")
        if not instance_id or not isinstance(instance_id, str):
            raise ValueError("Each instance requires a non-empty string id")
        if instance_id in seen_ids:
            raise ValueError(f"Duplicate instance id: {instance_id}")
        seen_ids.add(instance_id)

        host = item.get("host")
        if not host or not isinstance(host, str):
            raise ValueError(f"Instance {instance_id} requires a host")

        instance_port = item.get("port")
        if not isinstance(instance_port, int) or instance_port <= 0 or instance_port > 65535:
            raise ValueError(f"Instance {instance_id} has invalid port")

        binding = (host, instance_port)
        if binding in seen_bindings:
            raise ValueError(f"Duplicate instance host/port: {host}:{instance_port}")
        seen_bindings.add(binding)

        model = item.get("model")
        if not model or not isinstance(model, str):
            raise ValueError(f"Instance {instance_id} requires a model")

        tensor_parallel = item.get("tensor_parallel")
        if not isinstance(tensor_parallel, int) or tensor_parallel <= 0:
            raise ValueError(f"Instance {instance_id} has invalid tensor_parallel")

        gpu_memory = item.get("gpu_memory_utilization")
        device = item.get("device", "nvidia")
        profile = get_device_profile(device)
        if profile.supports_gpu_mem_util:
            if not isinstance(gpu_memory, (int, float)) or not 0 < gpu_memory <= 1:
                raise ValueError(
                    f"Instance {instance_id} has invalid gpu_memory_utilization"
                )

        max_model_len = item.get("max_model_len")
        if not isinstance(max_model_len, int) or max_model_len <= 0:
            raise ValueError(f"Instance {instance_id} has invalid max_model_len")

        extra_args = item.get("extra_args")
        if not isinstance(extra_args, list) or not all(
            isinstance(value, str) for value in extra_args
        ):
            raise ValueError(f"Instance {instance_id} extra_args must be a list of strings")

    return True
