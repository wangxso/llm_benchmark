"""Tests for multi-vendor device support module."""

import pytest
from unittest.mock import patch, MagicMock
import subprocess

from src.device.profile import (
    DeviceProfile,
    PROFILES,
    PROFILE_ALIASES,
    detect_device,
    get_device_profile,
    list_devices,
)
from src.device.monitor import (
    GpuInfo,
    GpuProcess,
    get_gpu_utilization,
    get_gpu_details,
    get_gpu_processes,
    _get_nvidia_utilization,
    _get_rocm_utilization,
    _get_ascend_utilization,
    _get_cambricon_utilization,
    _get_biren_utilization,
    _get_metax_utilization,
    _get_moorethreads_utilization,
)


class TestDeviceProfile:
    def test_profile_fields(self):
        p = DeviceProfile(
            name="test",
            display_name="Test GPU",
            visible_devices_env="TEST_VISIBLE_DEVICES",
            supports_gpu_mem_util=True,
            supports_tensor_parallel=True,
            monitor_tool="test_smi",
        )
        assert p.name == "test"
        assert p.visible_devices_env == "TEST_VISIBLE_DEVICES"
        assert p.supports_gpu_mem_util is True

    def test_nvidia_profile(self):
        p = PROFILES["nvidia"]
        assert p.visible_devices_env == "CUDA_VISIBLE_DEVICES"
        assert p.supports_gpu_mem_util is True
        assert p.monitor_tool == "pynvml"

    def test_rocm_profile(self):
        p = PROFILES["rocm"]
        assert p.visible_devices_env == "HIP_VISIBLE_DEVICES"
        assert p.supports_gpu_mem_util is True
        assert p.monitor_tool == "rocm_smi"

    def test_ascend_profile(self):
        p = PROFILES["ascend"]
        assert p.visible_devices_env == "ASCEND_RT_VISIBLE_DEVICES"
        assert p.supports_gpu_mem_util is False
        assert p.monitor_tool == "npu_smi"

    def test_cambricon_profile(self):
        p = PROFILES["cambricon"]
        assert p.visible_devices_env == "MLU_VISIBLE_DEVICES"
        assert p.supports_gpu_mem_util is False
        assert p.monitor_tool == "cnmon"

    def test_biren_profile(self):
        p = PROFILES["biren"]
        assert p.visible_devices_env == "BR_VISIBLE_DEVICES"
        assert p.supports_gpu_mem_util is False
        assert p.monitor_tool == "biren_smi"

    def test_metax_profile(self):
        p = PROFILES["metax"]
        assert p.visible_devices_env == "METAX_VISIBLE_DEVICES"
        assert p.supports_gpu_mem_util is False

    def test_moorethreads_profile(self):
        p = PROFILES["moorethreads"]
        assert p.visible_devices_env == "MT_VISIBLE_DEVICES"
        assert p.supports_gpu_mem_util is False


class TestProfileAliases:
    def test_cuda_alias(self):
        assert PROFILE_ALIASES["cuda"] == "nvidia"

    def test_hip_alias(self):
        assert PROFILE_ALIASES["hip"] == "rocm"

    def test_amd_alias(self):
        assert PROFILE_ALIASES["amd"] == "rocm"

    def test_huawei_alias(self):
        assert PROFILE_ALIASES["huawei"] == "ascend"

    def test_npu_alias(self):
        assert PROFILE_ALIASES["npu"] == "ascend"


class TestGetDeviceProfile:
    def test_explicit_nvidia(self):
        p = get_device_profile("nvidia")
        assert p.name == "nvidia"

    def test_explicit_ascend(self):
        p = get_device_profile("ascend")
        assert p.name == "ascend"

    def test_alias_cuda(self):
        p = get_device_profile("cuda")
        assert p.name == "nvidia"

    def test_alias_huawei(self):
        p = get_device_profile("huawei")
        assert p.name == "ascend"

    def test_unknown_fallback(self):
        p = get_device_profile("unknown_vendor")
        assert p.name == "nvidia"

    def test_none_auto_detect(self):
        p = get_device_profile(None)
        assert isinstance(p, DeviceProfile)

    def test_auto_detect(self):
        p = get_device_profile("auto")
        assert isinstance(p, DeviceProfile)


class TestDetectDevice:
    @patch("src.device.profile.shutil.which", return_value=None)
    @patch("src.device.profile.PROFILES", PROFILES)
    def test_detect_fallback_no_gpu(self, mock_which):
        # No pynvml, no smi tools → fallback to nvidia
        with patch.dict("sys.modules", {"pynvml": None}):
            result = detect_device()
            assert result == "nvidia"

    @patch("shutil.which")
    def test_detect_rocm(self, mock_which):
        def which_side_effect(cmd):
            if cmd == "rocm-smi":
                return "/usr/bin/rocm-smi"
            return None

        mock_which.side_effect = which_side_effect

        with patch.dict("sys.modules", {"pynvml": None}):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0, stdout="GPU [0] : gfx900"
                )
                result = detect_device()
                assert result == "rocm"

    @patch("shutil.which")
    def test_detect_ascend(self, mock_which):
        def which_side_effect(cmd):
            if cmd == "npu-smi":
                return "/usr/local/bin/npu-smi"
            return None

        mock_which.side_effect = which_side_effect

        with patch.dict("sys.modules", {"pynvml": None}):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0, stdout="NPU 0 : Ascend 910B"
                )
                result = detect_device()
                assert result == "ascend"


class TestListDevices:
    def test_list_devices(self):
        devices = list_devices()
        assert "nvidia" in devices
        assert "rocm" in devices
        assert "ascend" in devices
        assert "cambricon" in devices
        assert "biren" in devices
        assert isinstance(devices["nvidia"], str)
        assert devices["ascend"] == "Huawei Ascend NPU"


class TestGetGpuUtilization:
    @patch("src.device.monitor.get_device_profile")
    def test_fallback_on_unknown_monitor(self, mock_get_profile):
        mock_get_profile.return_value = DeviceProfile(
            name="unknown",
            display_name="Unknown",
            visible_devices_env="UNKNOWN_VISIBLE_DEVICES",
            supports_gpu_mem_util=False,
            supports_tensor_parallel=True,
            monitor_tool="nonexistent_smi",
        )
        assert get_gpu_utilization() == 0.0

    @patch("src.device.monitor.get_device_profile")
    def test_returns_float(self, mock_get_profile):
        mock_get_profile.return_value = PROFILES["nvidia"]
        # Even without actual GPU, should not raise
        result = get_gpu_utilization()
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


class TestMonitorFallbacks:
    def test_nvidia_no_pynvml(self):
        with patch.dict("sys.modules", {"pynvml": None}):
            result = _get_nvidia_utilization()
            assert result == 0.0

    @patch("shutil.which", return_value=None)
    def test_rocm_no_tool(self, mock_which):
        result = _get_rocm_utilization()
        assert result == 0.0

    @patch("shutil.which", return_value=None)
    def test_ascend_no_tool(self, mock_which):
        result = _get_ascend_utilization()
        assert result == 0.0

    @patch("shutil.which", return_value=None)
    def test_cambricon_no_tool(self, mock_which):
        result = _get_cambricon_utilization()
        assert result == 0.0

    @patch("shutil.which", return_value=None)
    def test_biren_no_tool(self, mock_which):
        result = _get_biren_utilization()
        assert result == 0.0

    @patch("shutil.which", return_value=None)
    def test_metax_no_tool(self, mock_which):
        result = _get_metax_utilization()
        assert result == 0.0

    @patch("shutil.which", return_value=None)
    def test_moorethreads_no_tool(self, mock_which):
        result = _get_moorethreads_utilization()
        assert result == 0.0


class TestAutotuneDeviceIntegration:
    def test_tuning_config_default_device(self):
        from src.autotune.config import TuningConfig

        config = TuningConfig()
        assert config.device == "nvidia"

    def test_tuning_config_custom_device(self):
        from src.autotune.config import TuningConfig

        config = TuningConfig(device="ascend")
        assert config.device == "ascend"

    def test_tuning_config_to_vllm_args_nvidia(self):
        from src.autotune.config import TuningConfig

        config = TuningConfig(device="nvidia")
        args = config.to_vllm_args()
        assert "gpu_memory_utilization" in args

    def test_tuning_config_to_vllm_args_ascend(self):
        from src.autotune.config import TuningConfig

        config = TuningConfig(device="ascend")
        args = config.to_vllm_args()
        assert "gpu_memory_utilization" not in args

    def test_tuning_config_to_vllm_args_cambricon(self):
        from src.autotune.config import TuningConfig

        config = TuningConfig(device="cambricon")
        args = config.to_vllm_args()
        assert "gpu_memory_utilization" not in args

    def test_tuning_config_to_dict_includes_device(self):
        from src.autotune.config import TuningConfig

        config = TuningConfig(device="biren")
        d = config.to_dict()
        assert d["device"] == "biren"


class TestGpuInfo:
    def test_gpu_info_fields(self):
        info = GpuInfo(
            index=0,
            name="Test GPU",
            uuid="GPU-12345",
            gpu_util=75.0,
            mem_util=60.0,
            mem_used_mb=8000,
            mem_total_mb=16000,
            mem_free_mb=8000,
            temperature_c=65.0,
            power_draw_w=200.0,
            power_limit_w=300.0,
            fan_speed=50.0,
        )
        assert info.index == 0
        assert info.name == "Test GPU"
        assert info.uuid == "GPU-12345"
        assert info.gpu_util == 75.0
        assert info.mem_util == 60.0
        assert info.mem_used_mb == 8000
        assert info.mem_total_mb == 16000
        assert info.mem_free_mb == 8000
        assert info.temperature_c == 65.0
        assert info.power_draw_w == 200.0
        assert info.power_limit_w == 300.0
        assert info.fan_speed == 50.0

    def test_gpu_info_defaults(self):
        info = GpuInfo(index=0, name="GPU")
        assert info.uuid == ""
        assert info.gpu_util == 0.0
        assert info.mem_used_mb == 0.0
        assert info.temperature_c == 0.0


class TestGpuProcess:
    def test_gpu_process_fields(self):
        proc = GpuProcess(
            gpu_index=0,
            pid=1234,
            process_name="python",
            used_memory_mb=512.0,
        )
        assert proc.gpu_index == 0
        assert proc.pid == 1234
        assert proc.process_name == "python"
        assert proc.used_memory_mb == 512.0


class TestGetGpuDetails:
    @patch("src.device.monitor.get_device_profile")
    def test_returns_list(self, mock_get_profile):
        mock_get_profile.return_value = PROFILES["nvidia"]
        result = get_gpu_details()
        assert isinstance(result, list)

    @patch("src.device.monitor.get_device_profile")
    def test_fallback_on_unknown_monitor(self, mock_get_profile):
        mock_get_profile.return_value = DeviceProfile(
            name="unknown",
            display_name="Unknown",
            visible_devices_env="UNKNOWN_VISIBLE_DEVICES",
            supports_gpu_mem_util=False,
            supports_tensor_parallel=True,
            monitor_tool="nonexistent_smi",
        )
        assert get_gpu_details() == []

    @patch("src.device.monitor.get_device_profile")
    def test_nvidia_no_pynvml_returns_empty(self, mock_get_profile):
        mock_get_profile.return_value = PROFILES["nvidia"]
        with patch.dict("sys.modules", {"pynvml": None}):
            result = get_gpu_details()
            assert result == []


class TestGetGpuProcesses:
    @patch("src.device.monitor.get_device_profile")
    def test_returns_list(self, mock_get_profile):
        mock_get_profile.return_value = PROFILES["nvidia"]
        result = get_gpu_processes()
        assert isinstance(result, list)

    @patch("src.device.monitor.get_device_profile")
    def test_non_nvidia_returns_empty(self, mock_get_profile):
        mock_get_profile.return_value = PROFILES["ascend"]
        assert get_gpu_processes() == []
