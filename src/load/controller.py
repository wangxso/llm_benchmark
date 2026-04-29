import asyncio
import time
import random
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from .generator import LoadScenario, LoadType


@dataclass
class RequestResult:
    """Result of a single request"""

    request_id: str
    start_time: float
    end_time: float = 0
    success: bool = False
    error: str = ""
    ttft: float = 0
    tpot: float = 0
    input_tokens: int = 0
    output_tokens: int = 0
    first_token_time: float = 0
    total_latency: float = 0


class TrafficController:
    """Traffic controller for different load patterns"""

    def __init__(self, config: Dict):
        self.config = config
        self.vllm_host = config.get("vllm", {}).get("host", "localhost")
        self.vllm_port = config.get("vllm", {}).get("port", 8000)
        self.timeout = config.get("request", {}).get("timeout", 120)

        self._results: List[RequestResult] = []
        self._active_requests = 0
        self._request_count = 0
        self._error_count = 0

    def run(self, scenario: LoadScenario, generator, client, stop_event=None) -> Dict[str, Any]:
        """Run load test based on scenario type"""
        if scenario.scenario_type == LoadType.FIXED:
            return asyncio.run(self._run_fixed(scenario, generator, client, stop_event))
        elif scenario.scenario_type == LoadType.STEP:
            return asyncio.run(self._run_step(scenario, generator, client, stop_event))
        elif scenario.scenario_type == LoadType.BURST:
            return asyncio.run(self._run_burst(scenario, generator, client, stop_event))
        elif scenario.scenario_type == LoadType.STREAMING:
            return asyncio.run(self._run_streaming(scenario, generator, client, stop_event))
        elif scenario.scenario_type == LoadType.LONG_CONTEXT:
            return asyncio.run(self._run_long_context(scenario, generator, client, stop_event))
        else:
            raise ValueError(f"Unknown load type: {scenario.scenario_type}")

    async def _send_request(self, client, request: Dict) -> RequestResult:
        """Send a single request and collect metrics"""
        result = RequestResult(
            request_id=str(random.randint(100000, 999999)), start_time=time.time()
        )

        try:
            if request.get("stream"):
                response = await self._send_streaming_request(client, request)
            else:
                response = await client.send_request(request, timeout=self.timeout)

            result.end_time = time.time()
            result.success = True
            result.total_latency = (result.end_time - result.start_time) * 1000

            if request.get("stream"):
                result.first_token_time = response.get("first_token_time", 0)
                if result.first_token_time:
                    result.ttft = (result.first_token_time - result.start_time) * 1000
                result.output_tokens = response.get("tokens", 0)

                if result.output_tokens > 1:
                    result.tpot = (result.total_latency - result.ttft) / (
                        result.output_tokens - 1
                    )
                else:
                    result.tpot = result.total_latency
            else:
                result.output_tokens = response.get("tokens", 0)
                if result.output_tokens > 0:
                    result.ttft = result.total_latency * 0.3
                    result.tpot = (
                        result.total_latency - result.ttft
                    ) / result.output_tokens

            result.input_tokens = response.get("input_tokens", 0)

        except Exception as e:
            result.end_time = time.time()
            result.success = False
            result.error = str(e)
            result.total_latency = (result.end_time - result.start_time) * 1000
            self._error_count += 1

        self._results.append(result)
        return result

    async def _send_streaming_request(self, client, request: Dict) -> Dict[str, Any]:
        """Send a streaming request and aggregate stream metrics"""
        first_token_time = 0
        token_count = 0
        content_parts = []

        async for chunk in client.send_request_stream(request.copy()):
            if not first_token_time and chunk.get("first_token_time"):
                first_token_time = chunk["first_token_time"]
            token_count = chunk.get("tokens", token_count)
            if chunk.get("content"):
                content_parts.append(chunk["content"])

        return {
            "content": "".join(content_parts),
            "tokens": token_count,
            "input_tokens": 0,
            "first_token_time": first_token_time,
        }

    async def _run_fixed(
        self, scenario: LoadScenario, generator, client, stop_event=None
    ) -> Dict[str, Any]:
        """Fixed concurrency load test"""
        print(f"[TrafficController] Running fixed concurrency: {scenario.concurrency}")

        warmup = scenario.warmup_duration
        duration = scenario.duration

        if warmup > 0:
            await asyncio.sleep(warmup)

        start_time = time.time()
        active_tasks = set()

        while time.time() - start_time < duration and not (stop_event and stop_event.is_set()):
            while len(active_tasks) < scenario.concurrency and not (stop_event and stop_event.is_set()):
                request = generator.generate_request()
                task = asyncio.create_task(self._send_request(client, request))
                active_tasks.add(task)
                self._request_count += 1

            done, active_tasks = await asyncio.wait(
                active_tasks, return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                await task

        if active_tasks:
            for t in active_tasks:
                t.cancel()
            await asyncio.gather(*active_tasks, return_exceptions=True)

        return self._aggregate_results()

    async def _run_step(
        self, scenario: LoadScenario, generator, client, stop_event=None
    ) -> Dict[str, Any]:
        """Step/incremental concurrency load test"""
        print(
            f"[TrafficController] Running step load: {scenario.concurrency} -> {scenario.max_concurrency}"
        )

        warmup = scenario.warmup_duration
        if warmup > 0:
            await asyncio.sleep(warmup)

        current_concurrency = scenario.concurrency
        step_count = 0

        while current_concurrency <= scenario.max_concurrency and not (stop_event and stop_event.is_set()):
            print(f"[Step] Level {step_count + 1}: concurrency={current_concurrency}")

            start_time = time.time()
            active_tasks = set()

            while time.time() - start_time < scenario.step_duration and not (stop_event and stop_event.is_set()):
                while len(active_tasks) < current_concurrency and not (stop_event and stop_event.is_set()):
                    request = generator.generate_request()
                    task = asyncio.create_task(self._send_request(client, request))
                    active_tasks.add(task)
                    self._request_count += 1

                done, active_tasks = await asyncio.wait(
                    active_tasks, return_when=asyncio.FIRST_COMPLETED
                )

                for task in done:
                    await task

            if active_tasks:
                for t in active_tasks:
                    t.cancel()
                await asyncio.gather(*active_tasks, return_exceptions=True)

            current_concurrency += scenario.step_increment
            step_count += 1

        return self._aggregate_results()

    async def _run_burst(
        self, scenario: LoadScenario, generator, client, stop_event=None
    ) -> Dict[str, Any]:
        """Burst/peak load test"""
        print(
            f"[TrafficController] Running burst load: peak={scenario.peak_concurrency}"
        )

        warmup = scenario.warmup_duration
        if warmup > 0:
            await asyncio.sleep(warmup)

        active_tasks = set()

        print(f"[Burst] Launching {scenario.peak_concurrency} concurrent requests")

        for _ in range(scenario.peak_concurrency):
            if stop_event and stop_event.is_set():
                break
            request = generator.generate_request()
            task = asyncio.create_task(self._send_request(client, request))
            active_tasks.add(task)
            self._request_count += 1

        print(f"[Burst] Waiting for completion...")
        await asyncio.gather(*active_tasks, return_exceptions=True)

        return self._aggregate_results()

    async def _run_streaming(
        self, scenario: LoadScenario, generator, client, stop_event=None
    ) -> Dict[str, Any]:
        """Streaming response load test"""
        print(
            f"[TrafficController] Running streaming load: concurrency={scenario.concurrency}"
        )

        old_stream = generator.stream
        generator.stream = True

        result = await self._run_fixed(scenario, generator, client, stop_event)

        generator.stream = old_stream
        return result

    async def _run_long_context(
        self, scenario: LoadScenario, generator, client, stop_event=None
    ) -> Dict[str, Any]:
        """Long context load test"""
        print(
            f"[TrafficController] Running long context: input_len={scenario.max_input_len}"
        )

        old_max_input = generator.dataset.max_input_len
        generator.dataset.max_input_len = scenario.max_input_len

        result = await self._run_fixed(scenario, generator, client, stop_event)

        generator.dataset.max_input_len = old_max_input
        return result

    def _aggregate_results(self) -> Dict[str, Any]:
        """Aggregate results into metrics"""
        if not self._results:
            return {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "qps": 0,
                "tps": 0,
            }

        successful = [r for r in self._results if r.success]
        failed = [r for r in self._results if not r.success]

        total_latencies = sorted([r.total_latency for r in successful])
        ttfts = sorted([r.ttft for r in successful if r.ttft > 0])
        tpots = sorted([r.tpot for r in successful if r.tpot > 0])

        total_tokens = sum(r.output_tokens for r in successful)

        duration = (
            (self._results[-1].end_time - self._results[0].start_time)
            if self._results
            else 1
        )

        metrics = {
            "total_requests": len(self._results),
            "successful_requests": len(successful),
            "failed_requests": len(failed),
            "qps": len(successful) / duration if duration > 0 else 0,
            "tps": total_tokens / duration if duration > 0 else 0,
            "latency_p50": self._percentile(total_latencies, 50)
            if total_latencies
            else 0,
            "latency_p90": self._percentile(total_latencies, 90)
            if total_latencies
            else 0,
            "latency_p99": self._percentile(total_latencies, 99)
            if total_latencies
            else 0,
            "ttft_p50": self._percentile(ttfts, 50) if ttfts else 0,
            "ttft_p90": self._percentile(ttfts, 90) if ttfts else 0,
            "ttft_p99": self._percentile(ttfts, 99) if ttfts else 0,
            "tpot_p50": self._percentile(tpots, 50) if tpots else 0,
            "tpot_p90": self._percentile(tpots, 90) if tpots else 0,
            "tpot_p99": self._percentile(tpots, 99) if tpots else 0,
            "avg_output_tokens": total_tokens / len(successful) if successful else 0,
            "success_rate": len(successful) / len(self._results)
            if self._results
            else 0,
            "error_rate": len(failed) / len(self._results) if self._results else 0,
        }

        return metrics

    def _percentile(self, sorted_list: List[float], percentile: int) -> float:
        """Calculate percentile from sorted list"""
        if not sorted_list:
            return 0

        index = int(len(sorted_list) * percentile / 100)
        index = min(index, len(sorted_list) - 1)
        return sorted_list[index]

    def _result_to_dict(self, result: RequestResult) -> Dict[str, Any]:
        return {
            "request_id": result.request_id,
            "start_time": result.start_time,
            "end_time": result.end_time,
            "success": result.success,
            "error": result.error,
            "ttft": result.ttft,
            "tpot": result.tpot,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "first_token_time": result.first_token_time,
            "total_latency": result.total_latency,
        }

    def get_results_live(self) -> List[Dict]:
        """Get live results view for metrics collection"""
        return [self._result_to_dict(r) for r in self._results]

    def get_results(self) -> List[Dict]:
        """Get all results as dicts"""
        return [self._result_to_dict(r) for r in self._results]

    def reset(self):
        """Reset controller state"""
        self._results = []
        self._request_count = 0
        self._error_count = 0
