"""Device profile definitions for multi-vendor GPU support."""

from dataclasses import dataclass
from typing import Dict, Optional
import subprocess
import shutil


@dataclass
class DeviceProfile:
    """Profile for a specific GPU vendor."""
    name: str
    display_name: str
    visible_devices_env: str
    supports_gpu_mem_util: bool
    supports_tensor_parallel: bool
    monitor_tool: str


# Built-in device profiles
PROFILES: Dict[str, DeviceProfile] = {
    "nvidia": DeviceProfile(
        name="nvidia",
        display_name="NVIDIA GPU",
        visible_devices_env="CUDA_VISIBLE_DEVICES",
        supports_gpu_mem_util=True,
        supports_tensor_parallel=True,
        monitor_tool="pynvml",
    ),
    "rocm": DeviceProfile(
        name="rocm",
        display_name="AMD ROCm",
        visible_devices_env="HIP_VISIBLE_DEVICES",  # ROCm also supports CUDA_VISIBLE_DEVICES
        supports_gpu_mem_util=True,
        supports_tensor_parallel=True,
        monitor_tool="rocm_smi",
    ),
    "ascend": DeviceProfile(
        name="ascend",
        display_name="Huawei Ascend NPU",
        visible_devices_env="ASCEND_RT_VISIBLE_DEVICES",
        supports_gpu_mem_util=False,
        supports_tensor_parallel=True,
        monitor_tool="npu_smi",
    ),
    "cambricon": DeviceProfile(
        name="cambricon",
        display_name="Cambricon MLU",
        visible_devices_env="MLU_VISIBLE_DEVICES",
        supports_gpu_mem_util=False,
        supports_tensor_parallel=True,
        monitor_tool="cnmon",
    ),
    "biren": DeviceProfile(
        name="biren",
        display_name="Biren GPU",
        visible_devices_env="BR_VISIBLE_DEVICES",
        supports_gpu_mem_util=False,
        supports_tensor_parallel=True,
        monitor_tool="biren_smi",
    ),
    "metax": DeviceProfile(
        name="metax",
        display_name="Metax GPU",
        visible_devices_env="METAX_VISIBLE_DEVICES",
        supports_gpu_mem_util=False,
        supports_tensor_parallel=True,
        monitor_tool="metax_smi",
    ),
    "moorethreads": DeviceProfile(
        name="moorethreads",
        display_name="Moore Threads GPU",
        visible_devices_env="MT_VISIBLE_DEVICES",
        supports_gpu_mem_util=False,
        supports_tensor_parallel=True,
        monitor_tool="mthreads_smi",
    ),
}

# Alias for common alternative names
PROFILE_ALIASES = {
    "cuda": "nvidia",
    "hip": "rocm",
    "amd": "rocm",
    "huawei": "ascend",
    "npu": "ascend",
    "寒武纪": "cambricon",
    "壁仞": "biren",
    "摩尔线程": "moorethreads",
}


def detect_device() -> str:
    """Auto-detect current GPU type.

    Returns the profile name of the detected device, or "nvidia" as fallback.
    Detection priority: pynvml > rocm-smi > npu-smi > cnmon > biren-smi > metax-smi > mthreads-smi
    """
    # Try NVIDIA (pynvml)
    try:
        import pynvml
        pynvml.nvmlInit()
        # If we get here, NVIDIA GPU is available
        pynvml.nvmlShutdown()
        return "nvidia"
    except Exception:
        pass

    # Try ROCm (rocm-smi)
    if shutil.which("rocm-smi"):
        try:
            result = subprocess.run(
                ["rocm-smi", "--showid"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and "GPU" in result.stdout:
                return "rocm"
        except Exception:
            pass

    # Try Huawei Ascend (npu-smi)
    if shutil.which("npu-smi"):
        try:
            result = subprocess.run(
                ["npu-smi", "info"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and "NPU" in result.stdout:
                return "ascend"
        except Exception:
            pass

    # Try Cambricon (cnmon)
    if shutil.which("cnmon"):
        try:
            result = subprocess.run(
                ["cnmon", "info"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return "cambricon"
        except Exception:
            pass

    # Try Biren (biren-smi or brsmi)
    if shutil.which("biren-smi") or shutil.which("brsmi"):
        smi_tool = "biren-smi" if shutil.which("biren-smi") else "brsmi"
        try:
            result = subprocess.run(
                [smi_tool],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return "biren"
        except Exception:
            pass

    # Try Metax (metax-smi)
    if shutil.which("metax-smi"):
        try:
            result = subprocess.run(
                ["metax-smi"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return "metax"
        except Exception:
            pass

    # Try Moore Threads (mthreads-gmi or mthreads-smi)
    if shutil.which("mthreads-gmi") or shutil.which("mthreads-smi"):
        smi_tool = "mthreads-gmi" if shutil.which("mthreads-gmi") else "mthreads-smi"
        try:
            result = subprocess.run(
                [smi_tool],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return "moorethreads"
        except Exception:
            pass

    # Fallback to nvidia (most common)
    return "nvidia"


def get_device_profile(device: Optional[str] = None) -> DeviceProfile:
    """Get device profile by name or auto-detect.

    Args:
        device: Device name (e.g., "nvidia", "ascend") or None for auto-detect.
                Also supports common aliases like "cuda", "huawei", etc.

    Returns:
        DeviceProfile for the specified or detected device.
    """
    if device is None or device == "auto":
        device = detect_device()

    # Check aliases
    if device in PROFILE_ALIASES:
        device = PROFILE_ALIASES[device]

    # Return profile or fallback to nvidia
    if device in PROFILES:
        return PROFILES[device]

    # Unknown device, return nvidia as fallback
    return PROFILES["nvidia"]


def list_devices() -> Dict[str, str]:
    """List all supported device types with display names."""
    return {name: profile.display_name for name, profile in PROFILES.items()}
