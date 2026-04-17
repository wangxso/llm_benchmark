from __future__ import annotations

import aiohttp
import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from .models import InstanceState


@dataclass
class BackendRequestError(Exception):
    status: int
    text: str
    headers: Dict[str, str]


class BackendClient:
    def __init__(self, timeout: int = 180):
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def check_health(self, instance: InstanceState) -> bool:
        session = await self._get_session()
        try:
            async with session.get(f"{instance.config.base_url}/health") as response:
                return response.status == 200
        except Exception:
            return False

    async def fetch_models(self, instance: InstanceState) -> List[Dict[str, Any]]:
        session = await self._get_session()
        try:
            async with session.get(f"{instance.config.base_url}/v1/models") as response:
                if response.status != 200:
                    return []
                payload = await response.json()
                return payload.get("data", []) if isinstance(payload, dict) else []
        except Exception:
            return []

    async def fetch_metrics(self, instance: InstanceState) -> Dict[str, float]:
        session = await self._get_session()
        try:
            async with session.get(f"{instance.config.base_url}/metrics") as response:
                if response.status != 200:
                    return {}
                text = await response.text()
                return self._parse_prometheus(text)
        except Exception:
            return {}

    async def create_completion(
        self, instance: InstanceState, payload: Dict[str, Any]
    ) -> tuple[int, Dict[str, str], bytes]:
        session = await self._get_session()
        async with session.post(
            f"{instance.config.base_url}/v1/chat/completions", json=payload
        ) as response:
            body = await response.read()
            headers = self._copy_response_headers(response.headers)
            return response.status, headers, body

    async def stream_completion(self, instance: InstanceState, payload: Dict[str, Any]):
        session = await self._get_session()
        response = await session.post(
            f"{instance.config.base_url}/v1/chat/completions", json=payload
        )
        if response.status >= 400:
            headers = self._copy_response_headers(response.headers)
            try:
                text = await response.text()
            finally:
                response.release()
            raise BackendRequestError(response.status, text, headers)
        return response

    def aggregate_models(self, instances: Iterable[InstanceState]) -> Dict[str, Any]:
        deduped: Dict[str, Dict[str, Any]] = {}
        for instance in instances:
            if not instance.healthy:
                continue
            for item in instance.models:
                item_id = item.get("id")
                if item_id and item_id not in deduped:
                    deduped[item_id] = item
            if instance.config.model and instance.config.model not in deduped:
                deduped[instance.config.model] = {
                    "id": instance.config.model,
                    "object": "model",
                    "owned_by": "vllm-lb",
                }
        return {"object": "list", "data": list(deduped.values())}

    def _copy_response_headers(self, headers: aiohttp.typedefs.LooseHeaders) -> Dict[str, str]:
        allowed = {"content-type", "cache-control"}
        result = {}
        for key, value in headers.items():
            if key.lower() in allowed:
                result[key] = value
        return result

    def _parse_prometheus(self, text: str) -> Dict[str, float]:
        metrics = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or " " not in line:
                continue
            name, raw_value = line.split(" ", 1)
            metric_name = name.split("{", 1)[0]
            try:
                metrics[metric_name] = float(raw_value)
            except ValueError:
                continue
        return metrics
