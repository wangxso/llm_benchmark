from __future__ import annotations

import asyncio
import os
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .models import InstanceConfig, InstanceState


class ProcessManager:
    def __init__(self, log_dir: str | Path, verbose: bool = True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self._processes: Dict[str, asyncio.subprocess.Process] = {}

    async def start_instance(self, state: InstanceState) -> InstanceState:
        if not state.config.managed:
            state.running = True
            state.last_error = ""
            return state

        existing = self._processes.get(state.config.id)
        if existing and existing.returncode is None:
            state.running = True
            state.pid = existing.pid
            return state

        command = self._build_command(state.config)
        log_path = self.log_dir / f"{state.config.id}.log"

        if self.verbose:
            print(f"\n[{state.config.id}] Starting vLLM instance...")
            print(f"[{state.config.id}] Model: {state.config.model}")
            print(f"[{state.config.id}] Port: {state.config.port}")
            print(f"[{state.config.id}] GPU: {state.config.gpu_ids or 'auto'}")
            print(f"[{state.config.id}] Command: {' '.join(command)}")
            print(f"[{state.config.id}] Log: {log_path}")

        log_handle = log_path.open("ab")
        env = os.environ.copy()
        if state.config.gpu_ids is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(state.config.gpu_ids)

        # 输出到日志文件和控制台
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )
        self._processes[state.config.id] = process

        # 启动后台任务转发输出
        if self.verbose:
            asyncio.create_task(
                self._stream_output(process, state.config.id, log_handle)
            )
        else:
            asyncio.create_task(self._write_to_file_only(process, log_handle))

        state.running = True
        state.pid = process.pid
        state.last_started_at = time.time()
        state.last_error = ""
        state.log_path = str(log_path)
        state.command = command
        state.exit_code = None
        return state

    async def _stream_output(
        self,
        process: asyncio.subprocess.Process,
        instance_id: str,
        log_handle
    ):
        """流式输出到控制台和日志文件"""
        try:
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                decoded = line.decode("utf-8", errors="replace")
                # 写入日志文件
                log_handle.write(line)
                log_handle.flush()
                # 打印到控制台
                print(f"[{instance_id}] {decoded}", end="")
        except Exception as e:
            print(f"[{instance_id}] Output stream error: {e}")
        finally:
            log_handle.close()

    async def _write_to_file_only(
        self,
        process: asyncio.subprocess.Process,
        log_handle
    ):
        """仅写入日志文件"""
        try:
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                log_handle.write(line)
                log_handle.flush()
        finally:
            log_handle.close()

    async def stop_instance(self, state: InstanceState) -> InstanceState:
        process = self._processes.get(state.config.id)
        if process and process.returncode is None:
            if self.verbose:
                print(f"\n[{state.config.id}] Stopping vLLM instance...")
            process.send_signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(process.wait(), timeout=10)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        if process:
            state.exit_code = process.returncode
            self._processes.pop(state.config.id, None)
        state.running = False
        state.pid = None
        state.healthy = False
        return state

    async def sync_instances(
        self,
        current: Dict[str, InstanceState],
        desired: Iterable[InstanceConfig],
        autostart: bool = True,
    ) -> Dict[str, InstanceState]:
        desired_map = {item.id: item for item in desired}

        for instance_id in list(current.keys()):
            if instance_id not in desired_map:
                await self.stop_instance(current[instance_id])
                current.pop(instance_id, None)

        for instance_id, config in desired_map.items():
            state = current.get(instance_id)
            if state is None:
                state = InstanceState(config=config)
                current[instance_id] = state
            else:
                old_config = state.config
                config_changed = asdict(old_config) != asdict(config)
                if config_changed and old_config.managed:
                    await self.stop_instance(state)
                state.config = config
            if autostart and config.enabled:
                await self.start_instance(state)

        return current

    def refresh_process_state(self, state: InstanceState) -> InstanceState:
        process = self._processes.get(state.config.id)
        if process is None:
            if state.config.managed:
                state.running = False
                state.pid = None
            return state

        if process.returncode is not None:
            state.running = False
            state.pid = None
            state.healthy = False
            state.exit_code = process.returncode
            self._processes.pop(state.config.id, None)
            if self.verbose:
                print(f"\n[{state.config.id}] Process exited with code {state.exit_code}")
        else:
            state.running = True
            state.pid = process.pid
        return state

    def _build_command(self, config: InstanceConfig) -> List[str]:
        command = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--host",
            config.host,
            "--port",
            str(config.port),
            "--model",
            config.model,
            "--tensor-parallel-size",
            str(config.tensor_parallel),
            "--gpu-memory-utilization",
            str(config.gpu_memory_utilization),
            "--max-model-len",
            str(config.max_model_len),
        ]
        # MFU metrics 需要 vLLM >= 0.18.0，默认不启用
        # 如需启用，请在 extra_args 中添加 "--enable-mfu-metrics"
        # if config.enable_mfu_metrics:
        #     command.append("--enable-mfu-metrics")
        command.extend(config.extra_args)
        return command
