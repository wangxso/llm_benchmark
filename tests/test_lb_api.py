"""Tests for lb API endpoints."""
import json
import pytest
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop

from lb.app import create_app
from lb.models import InstanceState, InstanceConfig


class TestBalancerAPI(AioHTTPTestCase):
    async def get_application(self):
        """Create test application."""
        app = create_app(config_path=None)
        return app

    @unittest_run_loop
    async def test_health_endpoint(self):
        """Test /health endpoint returns expected structure."""
        resp = await self.client.get("/health")
        assert resp.status in (200, 503)
        data = await resp.json()
        assert "status" in data
        assert "healthy_instances" in data
        assert "total_instances" in data

    @unittest_run_loop
    async def test_models_endpoint_returns_list(self):
        """Test /v1/models returns list structure."""
        resp = await self.client.get("/v1/models")
        data = await resp.json()
        assert "object" in data
        assert data["object"] == "list"
        assert "data" in data
        assert isinstance(data["data"], list)

    @unittest_run_loop
    async def test_chat_completions_no_healthy_instances(self):
        """Test chat completions returns 503 when no healthy instances."""
        resp = await self.client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
        )
        # Should be 503 when no instances are healthy
        assert resp.status == 503

    @unittest_run_loop
    async def test_admin_state_requires_no_token_by_default(self):
        """Test admin endpoints work without token when not configured."""
        resp = await self.client.get("/admin/state")
        assert resp.status == 200
        data = await resp.json()
        assert "config" in data
        assert "instances" in data
        assert "monitor" in data

    @unittest_run_loop
    async def test_admin_metrics_structure(self):
        """Test /admin/metrics returns expected structure."""
        resp = await self.client.get("/admin/metrics")
        assert resp.status == 200
        data = await resp.json()
        assert "total_requests" in data
        assert "instances" in data

    @unittest_run_loop
    async def test_admin_config_get(self):
        """Test GET /admin/config returns config text."""
        resp = await self.client.get("/admin/config")
        assert resp.status == 200
        data = await resp.json()
        assert "text" in data

    @unittest_run_loop
    async def test_ui_disabled(self):
        """Test UI can be disabled via config."""
        service = self.app["service"]
        service.config["ui"]["enabled"] = False
        resp = await self.client.get("/")
        # Without healthy instances, should return 404 (UI disabled)
        assert resp.status in (404, 503)


class TestBackendClient:
    """Tests for BackendClient."""

    def test_aggregate_models_filters_unhealthy(self):
        """Test that aggregate_models only includes healthy instances."""
        from lb.backend_client import BackendClient
        from lb.models import InstanceState, InstanceConfig

        client = BackendClient()

        config1 = InstanceConfig(id="gpu0", model="model-a", port=8001)
        state1 = InstanceState(config=config1, healthy=True)
        state1.models = [{"id": "model-a"}]

        config2 = InstanceConfig(id="gpu1", model="model-b", port=8002)
        state2 = InstanceState(config=config2, healthy=False)
        state2.models = [{"id": "model-b"}]

        result = client.aggregate_models([state1, state2])
        model_ids = [m["id"] for m in result["data"]]
        assert "model-a" in model_ids
        # unhealthy instance should not contribute its models
        assert "model-b" not in model_ids

    def test_parse_prometheus_metrics(self):
        """Test Prometheus text parsing."""
        from lb.backend_client import BackendClient

        text = """
# HELP vllm_num_requests_running Number of requests running
vllm:num_requests_running{model="test"} 5.0
# TYPE vllm_batch_size gauge
vllm:batch_size 3
"""

        client = BackendClient()
        metrics = client._parse_prometheus(text)
        assert metrics.get("vllm:num_requests_running") == 5.0
        assert metrics.get("vllm:batch_size") == 3.0
