"""Configuration evaluator for Auto-Tuning Agent."""

import asyncio
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .config import TuningConfig, TuningResult
from src.device import get_device_profile


@dataclass
class InstanceHandle:
    """Handle to a running vLLM instance."""
    process: asyncio.subprocess.Process
    config: TuningConfig
    port: int
    gpu_ids: str
    model_path: str
    log_path: str
    start_time: float = field(default_factory=time.time)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


class ConfigEvaluator:
    """Evaluates vLLM configurations by running benchmarks."""

    def __init__(
        self,
        model_path: str,
        gpu_ids: str,
        device: str = "nvidia",
        base_port: int = 8100,
        log_dir: str = "./results/autotune/logs",
        startup_timeout: int = 300,
        health_check_interval: float = 2.0,
        verbose: bool = True,
    ):
        self.model_path = model_path
        self.gpu_ids = gpu_ids
        self.device = device
        self.base_port = base_port
        self.log_dir = Path(log_dir)
        self.startup_timeout = startup_timeout
        self.health_check_interval = health_check_interval
        self.verbose = verbose

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._current_instance: Optional[InstanceHandle] = None
        self._trial_count = 0  # Track trials to increment port

    async def evaluate(
        self,
        config: TuningConfig,
        trial_id: int,
        objective: str = "throughput",
    ) -> TuningResult:
        """Evaluate a single configuration."""
        result = TuningResult(
            trial_id=trial_id,
            config=config,
            objective=objective,
        )

        try:
            # Start vLLM instance
            instance = await self._start_instance(config)
            self._current_instance = instance

            # Run load test
            metrics = await self._run_load_test(instance, config)
            result.metrics = metrics

            # Calculate score
            from .config import Objective
            result.calculate_score(Objective(objective))

        except Exception as e:
            result.error = str(e)
            result.score = float("-inf")
            if self.verbose:
                print(f"[Trial {trial_id}] Error: {e}")

        finally:
            # Stop instance
            await self._stop_instance()

        return result

    async def _start_instance(self, config: TuningConfig) -> InstanceHandle:
        """Start a vLLM instance with the given configuration."""
        # Validate model path first
        if not self.model_path:
            raise ValueError("Model path is not specified")

        # Check if model exists (for local paths)
        model_path_obj = Path(self.model_path)
        if not self.model_path.startswith("/") and "/" in self.model_path:
            # Relative path like "models/xxx" - check if it exists
            if not model_path_obj.exists() and not model_path_obj.is_dir():
                # Check if it might be a HuggingFace repo ID (contains "/" but doesn't exist locally)
                pass  # Assume it's a HF repo ID
        elif model_path_obj.exists() and model_path_obj.is_dir():
            # Local directory - check for config.json
            config_json = model_path_obj / "config.json"
            if not config_json.exists():
                raise ValueError(
                    f"Model directory '{self.model_path}' exists but missing 'config.json'. "
                    f"Please verify the model files are complete."
                )

        port = self.base_port
        log_path = str(self.log_dir / f"vllm_trial_{int(time.time())}.log")

        # Ensure previous instance is fully stopped before starting new one
        if self._current_instance is not None:
            await self._stop_instance()
            # Wait a moment for port to be released and GPU memory to free
            await asyncio.sleep(3)
        else:
            # Kill any orphan vLLM processes from prior crashed trials
            self._kill_orphan_vllm(port)
            await asyncio.sleep(1)

        command = self._build_command(config, port)

        if self.verbose:
            print(f"\n[ConfigEvaluator] Starting vLLM instance...")
            print(f"  GPU: {self.gpu_ids}")
            print(f"  Port: {port}")
            print(f"  gpu_memory_utilization: {config.gpu_memory_utilization}")
            print(f"  tensor_parallel: {config.tensor_parallel}")
            print(f"  max_model_len: {config.max_model_len}")
            print(f"  max_num_seqs: {config.max_num_seqs}")
            print(f"  Log: {log_path}")

        env = os.environ.copy()
        profile = get_device_profile(config.device)
        env[profile.visible_devices_env] = self.gpu_ids

        log_file = open(log_path, "ab")

        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=log_file,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )

        instance = InstanceHandle(
            process=process,
            config=config,
            port=port,
            gpu_ids=self.gpu_ids,
            model_path=self.model_path,
            log_path=log_path,
        )

        # Wait for health
        await self._wait_for_health(instance)

        return instance

    def _build_command(self, config: TuningConfig, port: int) -> list:
        """Build vLLM command."""
        profile = get_device_profile(config.device)

        command = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--host", "127.0.0.1",
            "--port", str(port),
            "--model", self.model_path,
            "--tensor-parallel-size", str(config.tensor_parallel),
        ]

        # Add --gpu-memory-utilization only if supported by the device
        if profile.supports_gpu_mem_util:
            command.extend([
                "--gpu-memory-utilization", str(config.gpu_memory_utilization),
            ])

        command.extend([
            "--max-model-len", str(config.max_model_len),
            "--max-num-seqs", str(config.max_num_seqs),
        ])

        if config.max_num_batched_tokens:
            command.extend([
                "--max-num-batched-tokens",
                str(config.max_num_batched_tokens)
            ])

        if config.enforce_eager:
            command.append("--enforce-eager")

        return command

    def _read_log_error(self, log_path: str) -> str:
        """Read log file to extract the root cause error message."""
        try:
            with open(log_path, "r") as f:
                content = f.read()
            # Look for common fatal errors
            for pattern in [
                "ValueError:",
                "RuntimeError:",
                "No available memory for the cache blocks",
                "OutOfMemoryError",
                "CUDA out of memory",
            ]:
                idx = content.rfind(pattern)
                if idx >= 0:
                    # Extract the line containing the error
                    line_start = content.rfind("\n", 0, idx) + 1
                    line_end = content.find("\n", idx)
                    if line_end < 0:
                        line_end = len(content)
                    return content[line_start:line_end].strip()
        except Exception:
            pass
        return ""

    async def _wait_for_health(self, instance: InstanceHandle, max_retries: int = 150):
        """Wait for vLLM instance to become healthy."""
        import aiohttp

        url = f"{instance.base_url}/health"
        start_time = time.time()
        log_checked = False

        while time.time() - start_time < self.startup_timeout:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        if resp.status == 200:
                            if self.verbose:
                                elapsed = time.time() - start_time
                                print(f"[ConfigEvaluator] Instance ready in {elapsed:.1f}s")
                            return
            except Exception:
                pass

            # Check if process died
            if instance.process.returncode is not None:
                log_error = self._read_log_error(instance.log_path)
                if log_error:
                    raise RuntimeError(
                        f"vLLM startup failed (exit {instance.process.returncode}): {log_error}"
                    )
                raise RuntimeError(
                    f"vLLM process exited with code {instance.process.returncode}"
                )

            # After 15s, check log for fatal errors that may cause a hang
            # (vLLM logs the error but may not exit immediately)
            if not log_checked and (time.time() - start_time > 15):
                log_checked = True
                try:
                    with open(instance.log_path, "r") as f:
                        content = f.read()
                    for fatal_msg in [
                        "No available memory for the cache blocks",
                        "CUDA out of memory",
                        "EngineCore failed to start",
                    ]:
                        if fatal_msg in content:
                            instance.process.kill()
                            await instance.process.wait()
                            log_error = self._read_log_error(instance.log_path)
                            raise RuntimeError(
                                f"vLLM startup failed: {log_error or fatal_msg}"
                            )
                except (IOError, OSError):
                    pass

            await asyncio.sleep(self.health_check_interval)

        raise TimeoutError(f"vLLM instance not ready after {self.startup_timeout}s")

    async def _run_load_test(
        self,
        instance: InstanceHandle,
        config: TuningConfig,
    ) -> Dict[str, float]:
        """Run load test against the instance."""
        import concurrent.futures

        from ..load.generator import LoadGenerator
        from ..load.controller import TrafficController
        from ..client.openai_client import OpenAIClient

        # Build config for load test
        load_config = {
            "vllm": {
                "host": "127.0.0.1",
                "port": instance.port,
                "model": self.model_path,
            },
            "load": {
                "type": "fixed",
                "base_concurrency": config.concurrency,
                "duration": config.duration,
                "warmup_duration": config.warmup_duration,
            },
            "dataset": {
                "mode": "generate",
                "generate": {
                    "short_ratio": 0.7,
                    "long_ratio": 0.3,
                    "max_input_len": min(2048, config.max_model_len // 2),
                    "max_output_len": 512,
                },
            },
            "request": {
                "stream": False,
                "temperature": 0.7,
                "max_tokens": 512,
                "timeout": 120,
            },
        }

        if self.verbose:
            print(f"[ConfigEvaluator] Running load test...")
            print(f"  Concurrency: {config.concurrency}")
            print(f"  Duration: {config.duration}s")

        # Run load test in a separate thread to avoid asyncio.run() conflict
        def _run_sync_load_test():
            generator = LoadGenerator(load_config)
            controller = TrafficController(load_config)
            client = OpenAIClient(load_config)

            scenario = generator.create_scenario()
            results = controller.run(scenario, generator, client)

            # Close client synchronously
            import asyncio
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is None:
                asyncio.run(client.close())
            # If there's a running loop, we'll just let it clean up on its own

            return results

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(_run_sync_load_test)
            try:
                results = future.result(timeout=config.duration + 120)
            except concurrent.futures.TimeoutError:
                if self.verbose:
                    print(f"[ConfigEvaluator] Load test timed out")
                results = {"tps": 0, "qps": 0, "latency_p99_ms": 0, "success_rate": 0}

        if self.verbose:
            print(f"[ConfigEvaluator] Load test completed")
            print(f"  TPS: {results.get('tps', 0):.2f} tokens/s")
            print(f"  QPS: {results.get('qps', 0):.2f} req/s")
            print(f"  Latency P99: {results.get('latency_p99', 0):.2f} ms")
            print(f"  Success Rate: {results.get('success_rate', 0) * 100:.1f}%")

        return results

    def _kill_orphan_vllm(self, port: int):
        """Kill any orphan vLLM processes on the given port or GPU."""
        killed = 0
        try:
            # Find vLLM processes listening on the port
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True, text=True, timeout=5,
            )
            for pid_str in result.stdout.strip().split("\n"):
                pid_str = pid_str.strip()
                if pid_str and pid_str.isdigit():
                    pid = int(pid_str)
                    try:
                        os.kill(pid, signal.SIGKILL)
                        killed += 1
                    except (ProcessLookupError, PermissionError):
                        pass
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Also kill any lingering vllm processes targeting our GPU IDs
        try:
            result = subprocess.run(
                ["pgrep", "-f", f"CUDA_VISIBLE_DEVICES={self.gpu_ids}.*vllm"],
                capture_output=True, text=True, timeout=5,
            )
            for pid_str in result.stdout.strip().split("\n"):
                pid_str = pid_str.strip()
                if pid_str and pid_str.isdigit():
                    pid = int(pid_str)
                    try:
                        os.kill(pid, signal.SIGKILL)
                        killed += 1
                    except (ProcessLookupError, PermissionError):
                        pass
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        if killed > 0 and self.verbose:
            print(f"[ConfigEvaluator] Killed {killed} orphan vLLM process(es)")

    async def _stop_instance(self):
        """Stop the current vLLM instance."""
        if self._current_instance is None:
            return

        instance = self._current_instance
        process = instance.process

        if process.returncode is None:
            if self.verbose:
                print(f"[ConfigEvaluator] Stopping vLLM instance...")

            process.send_signal(signal.SIGTERM)

            try:
                await asyncio.wait_for(process.wait(), timeout=30)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()

        self._current_instance = None

        # Also kill any orphan processes
        self._kill_orphan_vllm(instance.port)

        if self.verbose:
            print(f"[ConfigEvaluator] Instance stopped")

    async def cleanup(self):
        """Cleanup resources."""
        await self._stop_instance()
