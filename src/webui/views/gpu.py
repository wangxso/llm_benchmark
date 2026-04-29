"""GPU monitoring page with per-card details."""

import streamlit as st
import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.device import (
    detect_device,
    get_device_profile,
    get_gpu_details,
    get_gpu_processes,
    get_gpu_utilization,
    list_devices,
)


def _util_color(pct: float) -> str:
    """Return color string based on utilization percentage."""
    if pct >= 85:
        return "#ef4444"  # red
    elif pct >= 60:
        return "#f59e0b"  # yellow
    else:
        return "#22c55e"  # green


def _mem_bar(used: float, total: float) -> str:
    """Generate an HTML progress bar for memory usage."""
    if total <= 0:
        return '<div style="background:#334155;border-radius:4px;height:16px;width:100%"></div>'
    pct = min(used / total * 100, 100)
    color = _util_color(pct)
    return (
        f'<div style="background:#334155;border-radius:4px;height:16px;width:100%;position:relative">'
        f'<div style="background:{color};border-radius:4px;height:100%;width:{pct:.1f}%"></div>'
        f'<span style="position:absolute;top:0;left:8px;line-height:16px;font-size:11px;color:#e2e8f0">'
        f'{used:.0f}/{total:.0f} MB ({pct:.1f}%)</span></div>'
    )


def _util_bar(pct: float) -> str:
    """Generate an HTML progress bar for GPU utilization."""
    color = _util_color(pct)
    return (
        f'<div style="background:#334155;border-radius:4px;height:16px;width:100%;position:relative">'
        f'<div style="background:{color};border-radius:4px;height:100%;width:{pct:.1f}%"></div>'
        f'<span style="position:absolute;top:0;left:8px;line-height:16px;font-size:11px;color:#e2e8f0">'
        f'{pct:.1f}%</span></div>'
    )


def render_gpu_page():
    st.header("GPU Monitor")
    st.markdown("Real-time per-GPU card monitoring. NVIDIA CUDA is prioritized with full detail support.")

    # --- Settings ---
    col_settings1, col_settings2, col_settings3 = st.columns([2, 1, 1])
    with col_settings1:
        device_options = {"auto": "Auto (detect)", **list_devices()}
        device_names = list(device_options.keys())
        device = st.selectbox(
            "GPU Vendor",
            options=device_names,
            format_func=lambda x: device_options.get(x, x),
            index=0,
            help="Select GPU vendor. 'Auto' detects automatically (NVIDIA first).",
            key="gpu_monitor_device",
        )

    with col_settings2:
        auto_refresh = st.checkbox("Auto Refresh", value=True, key="gpu_auto_refresh")

    with col_settings3:
        refresh_interval = st.slider(
            "Interval (s)", 1, 10, 3, key="gpu_refresh_interval"
        )

    st.divider()

    # --- Detect and fetch ---
    actual_device = device if device != "auto" else detect_device()
    profile = get_device_profile(actual_device)

    # Badge
    vendor_badge = {
        "nvidia": "NVIDIA CUDA",
        "rocm": "AMD ROCm",
        "ascend": "Huawei Ascend",
        "cambricon": "Cambricon MLU",
        "biren": "Biren GPU",
        "metax": "Metax GPU",
        "moorethreads": "Moore Threads",
    }.get(actual_device, actual_device)

    st.markdown(
        f'<div style="display:inline-block;background:#1e3a5f;color:#60a5fa;'
        f'padding:4px 12px;border-radius:6px;font-size:13px;margin-bottom:8px">'
        f'{vendor_badge}</div>',
        unsafe_allow_html=True,
    )

    # Fetch data
    gpu_list = get_gpu_details(device if device != "auto" else None)
    processes = get_gpu_processes(device if device != "auto" else None)

    if not gpu_list:
        st.warning(
            "No GPU detected. Ensure GPU drivers are installed and accessible. "
            "For NVIDIA: install `pynvml` (`pip install pynvml`) and CUDA drivers."
        )
        if auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()
        return

    # --- Summary row ---
    avg_util = sum(g.gpu_util for g in gpu_list) / len(gpu_list) if gpu_list else 0
    total_mem_used = sum(g.mem_used_mb for g in gpu_list)
    total_mem = sum(g.mem_total_mb for g in gpu_list)
    avg_temp = sum(g.temperature_c for g in gpu_list) / len(gpu_list) if gpu_list else 0
    total_power = sum(g.power_draw_w for g in gpu_list)

    col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
    with col_s1:
        st.metric("GPU Count", len(gpu_list))
    with col_s2:
        st.metric("Avg Utilization", f"{avg_util:.1f}%")
    with col_s3:
        mem_pct = (total_mem_used / total_mem * 100) if total_mem > 0 else 0
        st.metric("Total VRAM", f"{total_mem_used:.0f}/{total_mem:.0f} MB", f"{mem_pct:.1f}%")
    with col_s4:
        st.metric("Avg Temperature", f"{avg_temp:.0f} °C")
    with col_s5:
        st.metric("Total Power", f"{total_power:.0f} W")

    st.divider()

    # --- Per-GPU cards ---
    num_gpus = len(gpu_list)
    cols_per_row = min(num_gpus, 4)

    for row_start in range(0, num_gpus, cols_per_row):
        row_gpus = gpu_list[row_start : row_start + cols_per_row]
        cols = st.columns(len(row_gpus))

        for col_idx, gpu in enumerate(row_gpus):
            with cols[col_idx]:
                st.markdown(f"**GPU {gpu.index}: {gpu.name}**")
                if gpu.uuid:
                    st.caption(f"`{gpu.uuid}`")

                # Utilization bar
                st.markdown("**GPU Utilization**", help="Compute utilization percentage")
                st.markdown(_util_bar(gpu.gpu_util), unsafe_allow_html=True)

                # Memory bar
                st.markdown("**VRAM**", help=f"Used: {gpu.mem_used_mb:.0f} MB / Total: {gpu.mem_total_mb:.0f} MB")
                st.markdown(_mem_bar(gpu.mem_used_mb, gpu.mem_total_mb), unsafe_allow_html=True)

                # Metrics row
                m1, m2, m3 = st.columns(3)
                with m1:
                    temp_color = "#ef4444" if gpu.temperature_c >= 80 else "#f59e0b" if gpu.temperature_c >= 60 else "#22c55e"
                    st.markdown(
                        f'<div style="text-align:center">'
                        f'<div style="font-size:11px;color:#94a3b8">Temp</div>'
                        f'<div style="font-size:16px;font-weight:600;color:{temp_color}">{gpu.temperature_c:.0f}°C</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with m2:
                    st.markdown(
                        f'<div style="text-align:center">'
                        f'<div style="font-size:11px;color:#94a3b8">Power</div>'
                        f'<div style="font-size:16px;font-weight:600">{gpu.power_draw_w:.0f}W</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with m3:
                    st.markdown(
                        f'<div style="text-align:center">'
                        f'<div style="font-size:11px;color:#94a3b8">Fan</div>'
                        f'<div style="font-size:16px;font-weight:600">{gpu.fan_speed:.0f}%</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                if gpu.power_limit_w > 0:
                    st.caption(f"Power limit: {gpu.power_limit_w:.0f} W")

                st.markdown("")  # spacer

    # --- Processes ---
    if processes:
        st.divider()
        st.subheader("GPU Processes")

        proc_data = []
        for p in processes:
            proc_data.append({
                "GPU": p.gpu_index,
                "PID": p.pid,
                "Process": p.process_name or "N/A",
                "VRAM (MB)": f"{p.used_memory_mb:.0f}",
            })

        st.dataframe(
            proc_data,
            use_container_width=True,
            hide_index=True,
        )

    # --- Auto refresh ---
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()
