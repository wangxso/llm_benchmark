"""Multi-vendor GPU support module.

This module provides device abstraction for supporting different GPU vendors:
- NVIDIA (CUDA)
- AMD ROCm (HIP)
- Huawei Ascend NPU
- Cambricon MLU
- Biren GPU
- Metax GPU
- Moore Threads GPU
"""

from .profile import (
    DeviceProfile,
    PROFILES,
    PROFILE_ALIASES,
    detect_device,
    get_device_profile,
    list_devices,
)

from .monitor import (
    GpuInfo,
    GpuProcess,
    get_gpu_utilization,
    get_gpu_details,
    get_gpu_processes,
)

__all__ = [
    "DeviceProfile",
    "PROFILES",
    "PROFILE_ALIASES",
    "detect_device",
    "get_device_profile",
    "list_devices",
    "GpuInfo",
    "GpuProcess",
    "get_gpu_utilization",
    "get_gpu_details",
    "get_gpu_processes",
]
