from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from aiohttp import web

from .backend_client import BackendClient, BackendRequestError
from .config import build_instance_configs, dump_config, load_config, save_config_text
from .models import InstanceState
from .monitor import ProxyMonitor
from .process_manager import ProcessManager
from .scheduler import LeastLoadScheduler, NoHealthyInstanceError


class BalancerService:
    def __init__(self, config_path: str | None = None):
        self.config_path = Path(config_path) if config_path else None
        self.config = load_config(str(self.config_path) if self.config_path else None)
        self.backend_client = BackendClient(
            timeout=self.config.get("server", {}).get("request_timeout", 180)
        )
        runtime = self.config.get("runtime", {})
        self.process_manager = ProcessManager(
            log_dir=runtime.get("log_dir", "./lb/runtime/logs"),
            verbose=True
        )
        scheduler = self.config.get("scheduler", {})
        self.scheduler = LeastLoadScheduler(
            queue_weight=scheduler.get("queue_weight", 2.0),
            inflight_weight=scheduler.get("inflight_weight", 1.0),
        )
        self.monitor = ProxyMonitor()
        self.instances: Dict[str, InstanceState] = {}
        self._refresh_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        await self.apply_config(self.config)
        self._refresh_task = asyncio.create_task(self._refresh_loop())

    async def close(self) -> None:
        if self._refresh_task:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
        for state in list(self.instances.values()):
            await self.process_manager.stop_instance(state)
        await self.backend_client.close()

    async def apply_config(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.backend_client.timeout = self.config.get("server", {}).get(
            "request_timeout", 180
        )
        scheduler = self.config.get("scheduler", {})
        self.scheduler = LeastLoadScheduler(
            queue_weight=scheduler.get("queue_weight", 2.0),
            inflight_weight=scheduler.get("inflight_weight", 1.0),
        )
        self.instances = await self.process_manager.sync_instances(
            self.instances,
            build_instance_configs(self.config),
            autostart=True,
        )
        await self.refresh_instances()

    async def reload_from_disk(self) -> Dict[str, Any]:
        config = load_config(str(self.config_path) if self.config_path else None)
        await self.apply_config(config)
        return config

    async def update_config_text(self, text: str) -> Dict[str, Any]:
        if self.config_path is None:
            raise ValueError("Cannot update config without a config path")
        config = save_config_text(text, self.config_path)
        await self.apply_config(config)
        return config

    def current_config_text(self) -> str:
        if self.config_path and self.config_path.exists():
            return self.config_path.read_text(encoding="utf-8")
        return dump_config(self.config)

    async def refresh_instances(self) -> None:
        tasks = [self._refresh_instance(state) for state in self.instances.values()]
        if tasks:
            await asyncio.gather(*tasks)

    async def _refresh_instance(self, state: InstanceState) -> None:
        self.process_manager.refresh_process_state(state)

        if not state.config.enabled:
            state.running = False
            state.healthy = False
            state.metrics = {}
            state.models = []
            return

        if not state.config.managed:
            state.running = True

        healthy = await self.backend_client.check_health(state)
        state.last_health_check = time.time()
        state.healthy = healthy
        if not state.config.managed:
            state.running = healthy

        if not healthy:
            state.metrics = {}
            state.models = []
            state.last_error = "health check failed"
            return

        state.last_error = ""
        state.models = await self.backend_client.fetch_models(state)
        state.metrics = await self.backend_client.fetch_metrics(state)

    async def _refresh_loop(self) -> None:
        while True:
            interval = self.config.get("scheduler", {}).get("refresh_interval", 2)
            await asyncio.sleep(interval)
            await self.refresh_instances()

    def require_admin(self, request: web.Request) -> None:
        token = self.config.get("server", {}).get("admin_token")
        if token and request.headers.get("x-admin-token") != token:
            raise web.HTTPUnauthorized(text="invalid admin token")

    def select_instance(self, requested_model: Optional[str]) -> InstanceState:
        return self.scheduler.select_instance(self.instances.values(), requested_model)

    def state_payload(self) -> Dict[str, Any]:
        return {
            "config": self.config,
            "instances": [state.to_dict() for state in self.instances.values()],
            "monitor": self.monitor.snapshot(self.instances.values()),
        }


async def on_startup(app: web.Application) -> None:
    await app["service"].start()


async def on_cleanup(app: web.Application) -> None:
    await app["service"].close()


@web.middleware
async def error_middleware(request: web.Request, handler):
    try:
        return await handler(request)
    except ValueError as exc:
        raise web.HTTPBadRequest(text=str(exc)) from exc


async def handle_index(request: web.Request) -> web.StreamResponse:
    service: BalancerService = request.app["service"]
    if not service.config.get("ui", {}).get("enabled", True):
        raise web.HTTPNotFound(text="UI disabled")
    static_dir: Path = request.app["static_dir"]
    return web.FileResponse(static_dir / "index.html")


async def handle_health(request: web.Request) -> web.Response:
    service: BalancerService = request.app["service"]
    healthy = sum(1 for state in service.instances.values() if state.healthy)
    payload = {
        "status": "ok" if healthy > 0 else "degraded",
        "healthy_instances": healthy,
        "total_instances": len(service.instances),
    }
    status = 200 if healthy > 0 else 503
    return web.json_response(payload, status=status)


async def handle_models(request: web.Request) -> web.Response:
    service: BalancerService = request.app["service"]
    payload = service.backend_client.aggregate_models(service.instances.values())
    return web.json_response(payload)


async def handle_chat_completions(request: web.Request) -> web.StreamResponse:
    service: BalancerService = request.app["service"]
    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:
        raise web.HTTPBadRequest(text=f"Invalid JSON body: {exc}") from exc

    requested_model = payload.get("model")
    try:
        instance = service.select_instance(requested_model)
    except NoHealthyInstanceError as exc:
        raise web.HTTPServiceUnavailable(text=str(exc)) from exc

    start_time = time.time()
    instance.inflight_requests += 1
    success = False
    stream_response = None
    backend_response = None

    try:
        if payload.get("stream"):
            backend_response = await service.backend_client.stream_completion(
                instance, payload
            )
            headers = {}
            content_type = backend_response.headers.get(
                "Content-Type", "text/event-stream"
            )
            headers["Content-Type"] = content_type
            cache_control = backend_response.headers.get("Cache-Control")
            if cache_control:
                headers["Cache-Control"] = cache_control

            stream_response = web.StreamResponse(
                status=backend_response.status, headers=headers
            )
            await stream_response.prepare(request)
            async for chunk in backend_response.content.iter_any():
                if chunk:
                    await stream_response.write(chunk)
            await stream_response.write_eof()
            success = backend_response.status < 400
            return stream_response

        status, headers, body = await service.backend_client.create_completion(
            instance, payload
        )
        success = status < 400
        return web.Response(status=status, headers=headers, body=body)
    except BackendRequestError as exc:
        return web.Response(status=exc.status, headers=exc.headers, text=exc.text)
    except web.HTTPException:
        raise
    except Exception as exc:
        raise web.HTTPBadGateway(text=str(exc)) from exc
    finally:
        instance.inflight_requests = max(0, instance.inflight_requests - 1)
        latency_ms = (time.time() - start_time) * 1000
        service.monitor.record(instance.config.id, success, latency_ms)
        if backend_response is not None:
            backend_response.release()


async def handle_admin_state(request: web.Request) -> web.Response:
    service: BalancerService = request.app["service"]
    service.require_admin(request)
    return web.json_response(service.state_payload())


async def handle_admin_metrics(request: web.Request) -> web.Response:
    service: BalancerService = request.app["service"]
    service.require_admin(request)
    return web.json_response(service.monitor.snapshot(service.instances.values()))


async def handle_admin_get_config(request: web.Request) -> web.Response:
    service: BalancerService = request.app["service"]
    service.require_admin(request)
    return web.json_response(
        {
            "path": str(service.config_path) if service.config_path else None,
            "text": service.current_config_text(),
        }
    )


async def handle_admin_put_config(request: web.Request) -> web.Response:
    service: BalancerService = request.app["service"]
    service.require_admin(request)
    if request.content_type == "application/json":
        payload = await request.json()
        text = payload.get("text", "")
    else:
        text = await request.text()
    config = await service.update_config_text(text)
    return web.json_response({"saved": True, "config": config})


async def handle_admin_reload(request: web.Request) -> web.Response:
    service: BalancerService = request.app["service"]
    service.require_admin(request)
    config = await service.reload_from_disk()
    return web.json_response({"reloaded": True, "config": config})


async def handle_admin_start_instance(request: web.Request) -> web.Response:
    service: BalancerService = request.app["service"]
    service.require_admin(request)
    instance_id = request.match_info["instance_id"]
    state = service.instances.get(instance_id)
    if state is None:
        raise web.HTTPNotFound(text=f"Unknown instance: {instance_id}")
    if not state.config.enabled:
        raise web.HTTPBadRequest(text="Instance is disabled in config")
    await service.process_manager.start_instance(state)
    await service.refresh_instances()
    return web.json_response(state.to_dict())


async def handle_admin_stop_instance(request: web.Request) -> web.Response:
    service: BalancerService = request.app["service"]
    service.require_admin(request)
    instance_id = request.match_info["instance_id"]
    state = service.instances.get(instance_id)
    if state is None:
        raise web.HTTPNotFound(text=f"Unknown instance: {instance_id}")
    await service.process_manager.stop_instance(state)
    return web.json_response(state.to_dict())


def create_app(config_path: str | None = None) -> web.Application:
    app = web.Application(
        client_max_size=20 * 1024**2,
        middlewares=[error_middleware],
    )
    service = BalancerService(config_path=config_path)
    static_dir = Path(__file__).resolve().parent / "static"

    app["service"] = service
    app["static_dir"] = static_dir
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    app.router.add_get("/", handle_index)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_post("/v1/chat/completions", handle_chat_completions)

    app.router.add_get("/admin/state", handle_admin_state)
    app.router.add_get("/admin/metrics", handle_admin_metrics)
    app.router.add_get("/admin/config", handle_admin_get_config)
    app.router.add_put("/admin/config", handle_admin_put_config)
    app.router.add_post("/admin/reload", handle_admin_reload)
    app.router.add_post(
        "/admin/instances/{instance_id}/start", handle_admin_start_instance
    )
    app.router.add_post(
        "/admin/instances/{instance_id}/stop", handle_admin_stop_instance
    )

    app.router.add_static("/static/", static_dir)
    return app
