"""Tests for lb.scheduler module."""
import pytest
from lb.scheduler import LeastLoadScheduler, NoHealthyInstanceError
from lb.models import InstanceConfig, InstanceState


def make_instance(
    instance_id: str,
    healthy: bool = True,
    enabled: bool = True,
    model: str = "test-model",
    running: int = 0,
    waiting: int = 0,
    inflight: int = 0,
) -> InstanceState:
    """Create a test instance state."""
    config = InstanceConfig(
        id=instance_id,
        enabled=enabled,
        model=model,
        port=8001,
    )
    state = InstanceState(
        config=config,
        healthy=healthy,
        running=True,
        inflight_requests=inflight,
        metrics={
            "vllm:num_requests_running": float(running),
            "vllm:num_requests_waiting": float(waiting),
        },
    )
    return state


def test_select_instance_no_healthy():
    """Test that no healthy instances raises error."""
    scheduler = LeastLoadScheduler()
    instances = [
        make_instance("gpu0", healthy=False),
        make_instance("gpu1", healthy=False),
    ]
    with pytest.raises(NoHealthyInstanceError):
        scheduler.select_instance(instances)


def test_select_instance_all_disabled():
    """Test that all disabled instances raises error."""
    scheduler = LeastLoadScheduler()
    instances = [
        make_instance("gpu0", enabled=False, healthy=True),
        make_instance("gpu1", enabled=False, healthy=True),
    ]
    with pytest.raises(NoHealthyInstanceError):
        scheduler.select_instance(instances)


def test_select_instance_single_healthy():
    """Test selection with single healthy instance."""
    scheduler = LeastLoadScheduler()
    instances = [
        make_instance("gpu0", healthy=True, running=5, waiting=2),
        make_instance("gpu1", healthy=False),
    ]
    selected = scheduler.select_instance(instances)
    assert selected.config.id == "gpu0"


def test_select_instance_least_load():
    """Test that least loaded instance is selected based on metrics."""
    scheduler = LeastLoadScheduler(queue_weight=2.0, inflight_weight=1.0)
    instances = [
        make_instance("gpu0", healthy=True, running=10, waiting=5, inflight=2),
        make_instance("gpu1", healthy=True, running=2, waiting=1, inflight=0),
    ]
    selected = scheduler.select_instance(instances)
    assert selected.config.id == "gpu1"


def test_select_instance_fallback_without_metrics():
    """Test selection falls back to round-robin when metrics unavailable."""
    scheduler = LeastLoadScheduler()
    instances = [
        make_instance("gpu0", healthy=True, inflight=3),
        make_instance("gpu1", healthy=True, inflight=1),
    ]
    # Clear metrics to simulate unavailable
    instances[0].metrics = {}
    instances[1].metrics = {}

    # Round-robin should cycle through instances
    first = scheduler.select_instance(instances)
    second = scheduler.select_instance(instances)
    assert first.config.id != second.config.id


def test_select_instance_model_filtering():
    """Test that instance is filtered by model when specified."""
    scheduler = LeastLoadScheduler()
    instances = [
        make_instance("gpu0", healthy=True, model="model-a"),
        make_instance("gpu1", healthy=True, model="model-b"),
    ]
    instances[0].models = [{"id": "model-a"}]
    instances[1].models = [{"id": "model-b"}]

    selected = scheduler.select_instance(instances, requested_model="model-b")
    assert selected.config.id == "gpu1"


def test_select_instance_model_not_found():
    """Test that error raised when requested model not found."""
    scheduler = LeastLoadScheduler()
    instances = [
        make_instance("gpu0", healthy=True, model="model-a"),
    ]
    instances[0].models = [{"id": "model-a"}]

    with pytest.raises(NoHealthyInstanceError):
        scheduler.select_instance(instances, requested_model="model-b")


def test_select_instance_empty_iterable():
    """Test that empty instances iterable raises error."""
    scheduler = LeastLoadScheduler()
    with pytest.raises(NoHealthyInstanceError):
        scheduler.select_instance([])
