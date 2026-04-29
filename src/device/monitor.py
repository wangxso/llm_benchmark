"""Unified GPU utilization monitor for multi-vendor support."""

import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Optional

from .profile import get_device_profile, DeviceProfile


@dataclass
class GpuInfo:
    """Per-GPU detail information."""
    index: int
    name: str
    uuid: str = ""
    gpu_util: float = 0.0        # 0-100
    mem_util: float = 0.0        # 0-100
    mem_used_mb: float = 0.0
    mem_total_mb: float = 0.0
    mem_free_mb: float = 0.0
    temperature_c: float = 0.0
    power_draw_w: float = 0.0
    power_limit_w: float = 0.0
    fan_speed: float = 0.0       # 0-100


@dataclass
class GpuProcess:
    """Process using a GPU."""
    gpu_index: int
    pid: int
    process_name: str
    used_memory_mb: float


def get_gpu_utilization(device: Optional[str] = None) -> float:
    """Get GPU utilization across all visible GPUs.

    Returns a float in [0.0, 1.0] representing average GPU utilization.
    Falls back to 0.0 if monitoring is unavailable.
    """
    profile = get_device_profile(device)

    monitor_funcs = {
        "pynvml": _get_nvidia_utilization,
        "rocm_smi": _get_rocm_utilization,
        "npu_smi": _get_ascend_utilization,
        "cnmon": _get_cambricon_utilization,
        "biren_smi": _get_biren_utilization,
        "metax_smi": _get_metax_utilization,
        "mthreads_smi": _get_moorethreads_utilization,
    }

    func = monitor_funcs.get(profile.monitor_tool)
    if func:
        try:
            return func()
        except Exception:
            return 0.0

    return 0.0


def _get_nvidia_utilization() -> float:
    """Get NVIDIA GPU utilization via pynvml."""
    try:
        import pynvml
        pynvml.nvmlInit()
    except Exception:
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

        return total_util / device_count / 100.0
    except Exception:
        return 0.0
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _get_rocm_utilization() -> float:
    """Get AMD ROCm GPU utilization via rocm-smi."""
    tool = shutil.which("rocm-smi")
    if not tool:
        return 0.0

    try:
        result = subprocess.run(
            ["rocm-smi", "--showuse", "--csv"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return 0.0

        # Parse CSV output: gpu,GPU use (%)
        util_values = []
        for line in result.stdout.strip().split("\n"):
            if "GPU use" in line or "gpu" in line.lower():
                continue
            parts = line.split(",")
            if len(parts) >= 2:
                try:
                    val = float(parts[1].strip().rstrip("%"))
                    util_values.append(val)
                except ValueError:
                    continue

        if not util_values:
            return 0.0

        return sum(util_values) / len(util_values) / 100.0
    except Exception:
        return 0.0


def _get_ascend_utilization() -> float:
    """Get Huawei Ascend NPU utilization via npu-smi."""
    tool = shutil.which("npu-smi")
    if not tool:
        return 0.0

    try:
        result = subprocess.run(
            ["npu-smi", "info"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return 0.0

        # Parse npu-smi info output
        # Look for utilization percentage lines
        util_values = []
        for line in result.stdout.split("\n"):
            line = line.strip()
            if "utilization" in line.lower() or "Usage" in line:
                # Try to extract percentage
                parts = line.split(":")
                if len(parts) >= 2:
                    val_str = parts[-1].strip().rstrip("%")
                    try:
                        val = float(val_str)
                        util_values.append(val)
                    except ValueError:
                        continue

        if not util_values:
            return 0.0

        return sum(util_values) / len(util_values) / 100.0
    except Exception:
        return 0.0


def _get_cambricon_utilization() -> float:
    """Get Cambricon MLU utilization via cnmon."""
    tool = shutil.which("cnmon")
    if not tool:
        return 0.0

    try:
        result = subprocess.run(
            ["cnmon", "info", "-s"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return 0.0

        # Parse cnmon output for utilization
        util_values = []
        for line in result.stdout.split("\n"):
            line = line.strip()
            if "util" in line.lower():
                parts = line.split()
                for part in parts:
                    part = part.rstrip("%")
                    try:
                        val = float(part)
                        if 0 <= val <= 100:
                            util_values.append(val)
                            break
                    except ValueError:
                        continue

        if not util_values:
            return 0.0

        return sum(util_values) / len(util_values) / 100.0
    except Exception:
        return 0.0


def _get_biren_utilization() -> float:
    """Get Biren GPU utilization via biren-smi or brsmi."""
    tool = shutil.which("biren-smi") or shutil.which("brsmi")
    if not tool:
        return 0.0

    try:
        result = subprocess.run(
            [tool, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            # Try alternative format
            result = subprocess.run(
                [tool],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return 0.0

            util_values = []
            for line in result.stdout.split("\n"):
                if "util" in line.lower():
                    parts = line.split()
                    for part in parts:
                        part = part.rstrip("%")
                        try:
                            val = float(part)
                            if 0 <= val <= 100:
                                util_values.append(val)
                                break
                        except ValueError:
                            continue

            if not util_values:
                return 0.0
            return sum(util_values) / len(util_values) / 100.0

        # Parse CSV output
        util_values = []
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                val = float(line)
                util_values.append(val)
            except ValueError:
                continue

        if not util_values:
            return 0.0

        return sum(util_values) / len(util_values) / 100.0
    except Exception:
        return 0.0


def _get_metax_utilization() -> float:
    """Get Metax GPU utilization via metax-smi."""
    tool = shutil.which("metax-smi")
    if not tool:
        return 0.0

    try:
        result = subprocess.run(
            [tool, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return 0.0

        util_values = []
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                val = float(line)
                util_values.append(val)
            except ValueError:
                continue

        if not util_values:
            return 0.0

        return sum(util_values) / len(util_values) / 100.0
    except Exception:
        return 0.0


def _get_moorethreads_utilization() -> float:
    """Get Moore Threads GPU utilization via mthreads-gmi or mthreads-smi."""
    tool = shutil.which("mthreads-gmi") or shutil.which("mthreads-smi")
    if not tool:
        return 0.0

    try:
        # Try query format first (mthreads-gmi supports nvidia-smi-like queries)
        result = subprocess.run(
            [tool, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            # Fallback: try plain output
            result = subprocess.run(
                [tool],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return 0.0

            util_values = []
            for line in result.stdout.split("\n"):
                if "util" in line.lower():
                    parts = line.split()
                    for part in parts:
                        part = part.rstrip("%")
                        try:
                            val = float(part)
                            if 0 <= val <= 100:
                                util_values.append(val)
                                break
                        except ValueError:
                            continue

            if not util_values:
                return 0.0
            return sum(util_values) / len(util_values) / 100.0

        util_values = []
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                val = float(line)
                util_values.append(val)
            except ValueError:
                continue

        if not util_values:
            return 0.0

        return sum(util_values) / len(util_values) / 100.0
    except Exception:
        return 0.0


def get_gpu_details(device: Optional[str] = None) -> List[GpuInfo]:
    """Get per-GPU detail information.

    Returns a list of GpuInfo, one per visible GPU.
    Falls back to [] if monitoring is unavailable.
    """
    profile = get_device_profile(device)

    detail_funcs = {
        "pynvml": _get_nvidia_details,
        "rocm_smi": _get_rocm_details,
        "npu_smi": _get_ascend_details,
    }

    func = detail_funcs.get(profile.monitor_tool)
    if func:
        try:
            return func()
        except Exception:
            return []

    return []


def get_gpu_processes(device: Optional[str] = None) -> List[GpuProcess]:
    """Get processes running on GPUs.

    Returns a list of GpuProcess. Only supported on NVIDIA (pynvml).
    Falls back to [] if unavailable.
    """
    profile = get_device_profile(device)
    if profile.monitor_tool == "pynvml":
        try:
            return _get_nvidia_processes()
        except Exception:
            return []
    return []


def _get_nvidia_details() -> List[GpuInfo]:
    """Get per-GPU details via pynvml."""
    try:
        import pynvml
        pynvml.nvmlInit()
    except Exception:
        return []

    try:
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count == 0:
            return []

        gpus = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            name = ""
            try:
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8")
            except Exception:
                pass

            uuid = ""
            try:
                uuid = pynvml.nvmlDeviceGetUUID(handle)
                if isinstance(uuid, bytes):
                    uuid = uuid.decode("utf-8")
            except Exception:
                pass

            gpu_util = 0.0
            mem_util = 0.0
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = float(util.gpu)
                mem_util = float(util.memory)
            except Exception:
                pass

            mem_used = 0.0
            mem_total = 0.0
            mem_free = 0.0
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_used = mem.used / (1024 * 1024)
                mem_total = mem.total / (1024 * 1024)
                mem_free = mem.free / (1024 * 1024)
            except Exception:
                pass

            temp = 0.0
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                pass

            power_draw = 0.0
            power_limit = 0.0
            try:
                power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
            except Exception:
                pass

            fan_speed = 0.0
            try:
                fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
            except Exception:
                pass

            gpus.append(GpuInfo(
                index=i,
                name=name,
                uuid=uuid,
                gpu_util=gpu_util,
                mem_util=mem_util,
                mem_used_mb=mem_used,
                mem_total_mb=mem_total,
                mem_free_mb=mem_free,
                temperature_c=temp,
                power_draw_w=power_draw,
                power_limit_w=power_limit,
                fan_speed=float(fan_speed),
            ))

        return gpus
    except Exception:
        return []
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _get_nvidia_processes() -> List[GpuProcess]:
    """Get processes running on NVIDIA GPUs via pynvml."""
    try:
        import pynvml
        pynvml.nvmlInit()
    except Exception:
        return []

    try:
        device_count = pynvml.nvmlDeviceGetCount()
        processes = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            try:
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            except Exception:
                procs = []

            for proc in procs:
                name = ""
                try:
                    name = proc.name or ""
                    if isinstance(name, bytes):
                        name = name.decode("utf-8")
                except Exception:
                    pass

                mem_mb = 0.0
                try:
                    mem_mb = proc.usedGpuMemory / (1024 * 1024)
                except Exception:
                    pass

                processes.append(GpuProcess(
                    gpu_index=i,
                    pid=proc.pid,
                    process_name=name,
                    used_memory_mb=mem_mb,
                ))
        return processes
    except Exception:
        return []
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _get_rocm_details() -> List[GpuInfo]:
    """Get per-GPU details via rocm-smi."""
    tool = shutil.which("rocm-smi")
    if not tool:
        return []

    try:
        result = subprocess.run(
            ["rocm-smi", "--showuse", "--showmeminfo", "vram", "--showtemp", "--showpower", "--csv"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            result = subprocess.run(
                ["rocm-smi", "--showuse", "--csv"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return []

        gpus = []
        lines = result.stdout.strip().split("\n")
        if len(lines) < 2:
            return []

        header = [h.strip() for h in lines[0].split(",")]
        for row_idx, line in enumerate(lines[1:]):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue

            gpu_util = 0.0
            mem_used = 0.0
            mem_total = 0.0
            temp = 0.0
            power = 0.0

            for col_idx, col_name in enumerate(header):
                if col_idx >= len(parts):
                    break
                val_str = parts[col_idx].strip().rstrip("%")
                col_lower = col_name.lower()
                try:
                    val = float(val_str)
                    if "gpu use" in col_lower or "gpu_use" in col_lower:
                        gpu_util = val
                    elif "vram" in col_lower and "used" in col_lower:
                        mem_used = val / (1024 * 1024) if val > 1e6 else val
                    elif "vram" in col_lower and "total" in col_lower:
                        mem_total = val / (1024 * 1024) if val > 1e6 else val
                    elif "temp" in col_lower:
                        temp = val
                    elif "power" in col_lower:
                        power = val
                except ValueError:
                    continue

            gpus.append(GpuInfo(
                index=row_idx,
                name=f"AMD GPU {row_idx}",
                gpu_util=gpu_util,
                mem_used_mb=mem_used,
                mem_total_mb=mem_total,
                mem_free_mb=mem_total - mem_used,
                temperature_c=temp,
                power_draw_w=power,
            ))

        return gpus
    except Exception:
        return []


def _get_ascend_details() -> List[GpuInfo]:
    """Get per-NPU details via npu-smi."""
    tool = shutil.which("npu-smi")
    if not tool:
        return []

    try:
        result = subprocess.run(
            ["npu-smi", "info"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return []

        gpus = []
        idx = 0
        for line in result.stdout.split("\n"):
            line = line.strip()
            if not line:
                continue
            if "NPU" in line and "%" in line:
                parts = line.split()
                util = 0.0
                for part in parts:
                    part = part.rstrip("%")
                    try:
                        val = float(part)
                        if 0 <= val <= 100:
                            util = val
                            break
                    except ValueError:
                        continue
                gpus.append(GpuInfo(
                    index=idx,
                    name=f"Ascend NPU {idx}",
                    gpu_util=util,
                ))
                idx += 1

        return gpus
    except Exception:
        return []
