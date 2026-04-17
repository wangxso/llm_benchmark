"""Tests for lb.config module."""
import pytest
from lb.config import (
    normalize_config,
    validate_config,
    build_instance_configs,
    load_config,
    parse_config_text,
    DEFAULT_CONFIG,
)


def test_normalize_config_empty_instances():
    """Test that empty instances list raises error."""
    config = {"instances": []}
    with pytest.raises(ValueError):
        normalize_config(config)


def test_normalize_config_missing_instance_id():
    """Test that instance without id raises error."""
    config = {"instances": [{"port": 8001, "model": "test"}]}
    with pytest.raises(ValueError):
        normalize_config(config)


def test_normalize_config_duplicate_ids():
    """Test that duplicate instance ids raise error."""
    config = {
        "instances": [
            {"id": "gpu0", "port": 8001, "model": "test"},
            {"id": "gpu0", "port": 8002, "model": "test"},
        ]
    }
    with pytest.raises(ValueError):
        normalize_config(config)


def test_normalize_config_duplicate_ports():
    """Test that duplicate host/port combinations raise error."""
    config = {
        "instances": [
            {"id": "gpu0", "host": "127.0.0.1", "port": 8001, "model": "test"},
            {"id": "gpu1", "host": "127.0.0.1", "port": 8001, "model": "test"},
        ]
    }
    with pytest.raises(ValueError):
        normalize_config(config)


def test_normalize_config_missing_model():
    """Test that instance without model raises error."""
    config = {"instances": [{"id": "gpu0", "port": 8001}]}
    with pytest.raises(ValueError):
        normalize_config(config)


def test_normalize_config_invalid_port():
    """Test that invalid port raises error."""
    config = {
        "instances": [{"id": "gpu0", "port": 99999, "model": "test"}]
    }
    with pytest.raises(ValueError):
        normalize_config(config)


def test_validate_config_invalid_server_port():
    """Test that invalid server port raises error."""
    config = {"server": {"port": 99999}, "instances": [{"id": "t", "port": 1, "model": "m"}]}
    with pytest.raises(ValueError, match="server.port"):
        validate_config(config)


def test_validate_config_invalid_scheduler_strategy():
    """Test that invalid scheduler strategy raises error."""
    config = {
        "server": {"port": 9000},
        "scheduler": {"strategy": "round_robin"},
        "instances": [{"id": "t", "port": 1, "model": "m"}],
    }
    with pytest.raises(ValueError):
        validate_config(config)


def test_build_instance_configs():
    """Test building instance configs from normalized config."""
    config = {
        "instances": [
            {
                "id": "gpu0",
                "enabled": True,
                "managed": True,
                "host": "127.0.0.1",
                "port": 8001,
                "model": "test-model",
                "tensor_parallel": 1,
                "gpu_ids": "0",
                "gpu_memory_utilization": 0.9,
                "max_model_len": 2048,
                "enable_mfu_metrics": True,
                "extra_args": [],
            }
        ]
    }
    instances = build_instance_configs(config)
    assert len(instances) == 1
    assert instances[0].id == "gpu0"
    assert instances[0].base_url == "http://127.0.0.1:8001"


def test_parse_config_text_valid():
    """Test parsing valid config text."""
    text = """
instances:
  - id: gpu0
    port: 8001
    model: test-model
"""
    config = parse_config_text(text)
    assert len(config["instances"]) == 1
    assert config["instances"][0]["id"] == "gpu0"


def test_parse_config_text_invalid_yaml():
    """Test parsing invalid YAML raises error."""
    text = "invalid: [yaml"
    with pytest.raises(Exception):  # yaml.parser.ParserError or ValueError
        parse_config_text(text)
