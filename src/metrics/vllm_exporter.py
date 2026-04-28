import asyncio
import aiohttp
from typing import Dict, Any, Optional

# GPU monitoring via pynvml (optional)
_pynvml_available = False
try:
    import pynvml
    pynvml.nvmlInit()
    _pynvml_available = True
except Exception:
    pass


def _get_gpu_utilization() -> float:
    """Get GPU utilization via nvidia-smi/pynvml"""
    if not _pynvml_available:
        return 0.0

    try:
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count == 0:
            return 0.0

        total_util = 0.0
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            total_util += util.gpu

        return total_util / device_count / 100.0  # Return as 0-1 range
    except Exception:
        return 0.0


class VLLMExporter:
    """vLLM metrics exporter"""

    def __init__(self, host: str = "localhost", port: int = 8000):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"

    async def fetch_metrics(self) -> Dict[str, float]:
        """Fetch metrics from vLLM"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/metrics") as response:
                    if response.status == 200:
                        text = await response.text()
                        return self._parse_metrics(text)
                    return {}
        except Exception:
            return {}

    def fetch_metrics_sync(self) -> Dict[str, float]:
        """Fetch metrics synchronously"""
        try:
            import requests

            response = requests.get(f"{self.base_url}/metrics", timeout=5)
            if response.status_code == 200:
                return self._parse_metrics(response.text)
            return {}
        except Exception:
            return {}

    def _parse_metrics(self, text: str) -> Dict[str, float]:
        """Parse Prometheus text format"""
        metrics = {}

        for line in text.strip().split("\n"):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) >= 2:
                raw_name = parts[0]
                name = raw_name.split("{", 1)[0]
                try:
                    value = float(parts[1])
                    metrics[name] = value
                except ValueError:
                    continue

        return metrics

    def get_key_metrics(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Extract key metrics"""
        return {
            "num_requests_running": metrics.get("vllm:num_requests_running", 0),
            "num_requests_waiting": metrics.get("vllm:num_requests_waiting", 0),
            "batch_size": metrics.get("vllm:batch_size", 0),
            "kv_cache_usage": metrics.get("vllm:kv_cache_usage", 0),
            "prefill_latency": metrics.get("vllm:prefill_latency", 0),
            "decode_latency": metrics.get("vllm:decode_latency", 0),
            "gpu_utilization": _get_gpu_utilization(),  # Real GPU utilization via pynvml
        }

    async def check_health(self) -> bool:
        """Check vLLM health"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/health") as response:
                    return response.status == 200
        except Exception:
            return False

    async def get_server_info(self) -> Dict[str, Any]:
        """Get vLLM server info"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/v1/models") as response:
                    if response.status == 200:
                        return await response.json()
                    return {}
        except Exception:
            return {}
