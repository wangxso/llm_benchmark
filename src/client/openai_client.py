import asyncio
import aiohttp
import time
from typing import Dict, Any, Optional, AsyncIterator


class OpenAIClient:
    """OpenAI-compatible client for vLLM"""

    def __init__(self, config: Dict):
        self.config = config

        # Support direct base_url or fall back to host/port
        self.base_url = config.get("vllm", {}).get("base_url")
        if self.base_url:
            # Remove trailing /v1 if present (we'll add it back)
            self.base_url = self.base_url.rstrip("/").rstrip("/v1")
            # Extract host and port for health checks
            parts = self.base_url.replace("http://", "").replace("https://", "").split(":")
            self.host = parts[0]
            self.port = int(parts[1]) if len(parts) > 1 else (443 if self.base_url.startswith("https") else 80)
        else:
            self.host = config.get("vllm", {}).get("host", "localhost")
            self.port = config.get("vllm", {}).get("port", 8000)
            self.base_url = f"http://{self.host}:{self.port}"

        self.timeout = config.get("request", {}).get("timeout", 120)

        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        """Close the client session"""
        if self._session and not self._session.closed:
            await self._session.close()

    async def send_request(self, request: Dict, timeout: int = None) -> Dict[str, Any]:
        """Send a non-streaming request"""
        session = await self._get_session()

        url = f"{self.base_url}/v1/chat/completions"

        timeout_val = timeout or self.timeout

        try:
            async with session.post(url, json=request) as response:
                if response.status != 200:
                    text = await response.text()
                    raise Exception(
                        f"Request failed with status {response.status}: {text}"
                    )

                result = await response.json()

                return self._parse_response(result, stream=False)

        except asyncio.TimeoutError:
            raise Exception(f"Request timeout after {timeout_val}s")
        except aiohttp.ClientError as e:
            raise Exception(f"Client error: {e}")
        except Exception as e:
            raise Exception(f"Request error: {e}")

    async def send_request_stream(self, request: Dict) -> AsyncIterator[Dict[str, Any]]:
        """Send a streaming request"""
        session = await self._get_session()

        url = f"{self.base_url}/chat/completions"
        request["stream"] = True

        try:
            async with session.post(url, json=request) as response:
                if response.status != 200:
                    text = await response.text()
                    raise Exception(
                        f"Request failed with status {response.status}: {text}"
                    )

                first_token_time = None
                token_count = 0

                async for line in response.content:
                    line = line.decode("utf-8").strip()

                    if not line or not line.startswith("data:"):
                        continue

                    if line.startswith("data: "):
                        data = line[6:]
                    else:
                        data = line

                    if data == "[DONE]":
                        break

                    try:
                        import json

                        chunk = json.loads(data)

                        if "choices" in chunk and chunk["choices"]:
                            delta = chunk["choices"][0].get("delta", {})

                            if delta.get("content"):
                                if first_token_time is None:
                                    first_token_time = time.time()

                                token_count += 1

                                yield {
                                    "content": delta["content"],
                                    "first_token_time": first_token_time,
                                    "tokens": token_count,
                                    "finish_reason": chunk["choices"][0].get(
                                        "finish_reason"
                                    ),
                                }
                    except json.JSONDecodeError:
                        continue

        except asyncio.TimeoutError:
            raise Exception(f"Request timeout after {self.timeout}s")
        except aiohttp.ClientError as e:
            raise Exception(f"Client error: {e}")

    def _parse_response(self, result: Dict, stream: bool = False) -> Dict[str, Any]:
        """Parse API response"""
        if "choices" not in result or not result["choices"]:
            raise Exception("Invalid response: no choices")

        choice = result["choices"][0]

        if stream:
            content = ""
            for _ in self.send_request_stream(result):
                content += _.get("content", "")
            return {
                "content": content,
                "tokens": len(content),
                "finish_reason": choice.get("finish_reason"),
            }

        message = choice.get("message", {})
        content = message.get("content", "")

        usage = result.get("usage", {})

        return {
            "content": content,
            "tokens": usage.get("completion_tokens", len(content)),
            "input_tokens": usage.get("prompt_tokens", 0),
            "finish_reason": choice.get("finish_reason"),
        }

    async def check_health(self) -> bool:
        """Check if vLLM server is healthy"""
        try:
            session = await self._get_session()
            async with session.get(
                f"http://{self.host}:{self.port}/health"
            ) as response:
                return response.status == 200
        except Exception:
            return False

    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/models") as response:
                if response.status == 200:
                    return await response.json()
                return {}
        except Exception:
            return {}

    async def get_vllm_metrics(self) -> Dict[str, Any]:
        """Get vLLM internal metrics"""
        try:
            session = await self._get_session()
            async with session.get(
                f"http://{self.host}:{self.port}/metrics"
            ) as response:
                if response.status == 200:
                    text = await response.text()
                    return self._parse_prometheus(text)
                return {}
        except Exception:
            return {}

    def _parse_prometheus(self, text: str) -> Dict[str, float]:
        """Parse Prometheus metrics text format"""
        metrics = {}

        for line in text.split("\n"):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            if " " in line:
                parts = line.split(" ", 1)
                raw_name = parts[0]
                metric_name = raw_name.split("{", 1)[0]
                metric_value = parts[1] if len(parts) > 1 else ""

                try:
                    value = float(metric_value)
                    metrics[metric_name] = value
                except ValueError:
                    continue

        return metrics
