"""Load testing page"""

import streamlit as st
import time
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.webui.views.providers import load_providers, Provider
from src.webui.views.balancer import get_lb_as_provider
from src.webui.task_manager import start_task, get_active_tasks, get_all_tasks


def _run_lb_background(task_id: str, stop_event, **kwargs):
    """Background target for load test."""
    from src.webui.task_manager import update_progress, complete_task, _fail

    try:
        update_progress(task_id, 0.05, "Initializing load test...")

        config = {
            "vllm": {
                "base_url": kwargs["api_base_url"].rstrip("/").rstrip("/v1"),
                "model": kwargs["model"],
            },
            "load": {
                "name": f"{kwargs['scenario']}_test",
                "type": kwargs["scenario"],
                "base_concurrency": kwargs["base_concurrency"],
                "duration": kwargs["duration"],
            },
            "request": {
                "max_tokens": kwargs["max_tokens"],
                "stream": kwargs["stream"],
            },
            "dataset": {"mode": kwargs["dataset_mode"]},
        }
        if kwargs.get("api_key"):
            config.setdefault("api", {})["key"] = kwargs["api_key"]

        from src.scenario.manager import ScenarioManager
        from src.load.generator import LoadGenerator
        from src.load.controller import TrafficController
        from src.client.openai_client import OpenAIClient
        from src.metrics.collector import MetricsCollector

        manager = ScenarioManager(config)
        generator = LoadGenerator(config)
        controller = TrafficController(config)
        client = OpenAIClient(config)
        collector = MetricsCollector(config)

        update_progress(task_id, 0.1, "Starting metrics collection...")
        collector.start()

        update_progress(task_id, 0.15, f"Running {kwargs['scenario']} load test...")
        load_scenario = generator.create_scenario()
        results = controller.run(load_scenario, generator, client, stop_event=stop_event)

        update_progress(task_id, 0.85, "Generating report...")
        collector.stop()

        from src.report.generator import ReportGenerator
        report_gen = ReportGenerator(config)
        report = report_gen.generate(results, collector.get_metrics())

        output_dir = config.get("output", {}).get("path", "./results")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"{output_dir}/benchmark_{timestamp}.json"
        report_gen.save(report, report_file)
        report["report_file"] = report_file

        complete_task(task_id, report)
    except Exception as e:
        _fail(task_id, str(e))


def render_lb_page():
    st.header("⚡ Load Testing")
    st.markdown("Run load tests against your LLM API endpoints.")

    # Show running task status
    active = {k: v for k, v in get_active_tasks().items() if v.task_type == "loadtest"}
    if active:
        for tid, task in active.items():
            st.info(f"⏳ Load test running: **{task.label}** — {task.progress_text} ({task.elapsed_str()})")
            st.progress(task.progress)
            from src.webui.task_manager import stop_task
            if st.button("⏹ Stop", key=f"stop_lb_{tid}"):
                stop_task(tid)
                st.rerun()
        st.caption("Navigate to **Results** page to see progress. You can continue using other pages.")

    # Load providers
    providers = load_providers()

    # Add Load Balancer as a provider option
    lb_provider = get_lb_as_provider()
    if lb_provider:
        providers.insert(0, lb_provider)

    if not providers:
        st.warning("No providers configured. Please add providers in Settings first.")
        return

    # Provider Selection
    with st.expander("🔑 Provider Selection", expanded=True):
        col1, col2 = st.columns([1, 3])

        with col1:
            selected_provider_name = st.selectbox(
                "Select Provider",
                [p.name for p in providers],
                key="lb_provider_select",
                help="Select a configured provider"
            )

        # Get selected provider
        selected_provider = next((p for p in providers if p.name == selected_provider_name), None)

        if selected_provider:
            # Show provider info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"**API Type:** `{selected_provider.api_type}`")
            with col2:
                st.caption(f"**Base URL:** `{selected_provider.base_url}`")
            with col3:
                st.caption(f"**Default Model:** `{selected_provider.default_model or 'Auto'}`")

            # Allow model override
            model = st.text_input(
                "Model (override)",
                value=selected_provider.default_model,
                key="lb_model",
                help="Leave as default or enter a different model name"
            )

            stream = st.checkbox("Streaming", value=False, key="lb_stream")

    # Load Test Configuration
    with st.expander("📈 Load Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            scenario = st.selectbox(
                "Scenario",
                ["fixed", "step", "burst", "streaming", "long_context"],
                key="lb_scenario",
                help="Type of load test to run"
            )
            base_concurrency = st.number_input(
                "Base Concurrency",
                min_value=1, max_value=1000, value=100,
                key="lb_base_concurrency"
            )
            duration = st.number_input(
                "Duration (seconds)",
                min_value=10, max_value=3600, value=300,
                key="lb_duration"
            )

        with col2:
            if scenario == "step":
                increment = st.number_input("Step Increment", min_value=1, value=50, key="lb_increment")
                steps = st.number_input("Number of Steps", min_value=1, value=5, key="lb_steps")
                step_duration = st.number_input("Step Duration (s)", min_value=10, value=30, key="lb_step_duration")
            elif scenario == "burst":
                peak = st.number_input("Peak Concurrency", min_value=1, value=500, key="lb_peak")
                warmup = st.number_input("Warmup Duration (s)", min_value=0, value=30, key="lb_warmup")
            else:
                st.info("Configure additional parameters based on scenario type")

            max_tokens = st.number_input(
                "Max Output Tokens",
                min_value=1, max_value=8192, value=1024,
                key="lb_max_tokens"
            )

        with col3:
            dataset_mode = st.selectbox(
                "Prompt Source",
                ["generate", "import"],
                key="lb_dataset_mode"
            )

            if dataset_mode == "import":
                dataset_path = st.text_input("Dataset Path", key="lb_dataset_path")
            else:
                max_input_len = st.number_input(
                    "Max Input Length",
                    min_value=64, max_value=32768, value=4096,
                    key="lb_max_input"
                )
                short_ratio = st.slider(
                    "Short Prompt Ratio",
                    0.0, 1.0, 0.3,
                    key="lb_short_ratio"
                )

    # Run button
    st.markdown("---")

    if st.button("🚀 Run Load Test", type="primary", width="stretch"):
        if not selected_provider:
            st.error("No provider selected")
            return

        if not selected_provider.api_key and "localhost" not in selected_provider.base_url and "127.0.0.1" not in selected_provider.base_url:
            st.warning(f"Provider '{selected_provider.name}' has no API key configured. This may cause authentication errors.")

        label = f"{scenario} ({base_concurrency}c, {duration}s)"
        start_task(
            task_type="loadtest",
            label=label,
            target_fn=_run_lb_background,
            api_base_url=selected_provider.base_url,
            model=model,
            api_key=selected_provider.api_key,
            api_type=selected_provider.api_type,
            stream=stream,
            scenario=scenario,
            base_concurrency=base_concurrency,
            duration=duration,
            max_tokens=max_tokens,
            dataset_mode=dataset_mode,
        )
        st.success(f"✅ Load test started in background: **{label}**")
        st.caption("Go to **Results** page to monitor progress and stop if needed.")
        st.rerun()

    # Display persisted results from previous run
    if "lb_report" in st.session_state and st.session_state["lb_report"]:
        st.markdown("---")
        display_load_test_results(st.session_state["lb_report"])

    # Show completed task results
    completed = {k: v for k, v in get_all_tasks().items()
                 if v.task_type == "loadtest" and v.status == "completed" and v.result}
    if completed:
        for tid, task in completed.items():
            if task.result:
                st.session_state["lb_report"] = task.result


def display_load_test_results(report: Dict):
    """Display load test results"""

    st.success("✅ Load test completed!")

    metrics = report.get("metrics", {})

    # Throughput
    st.markdown("#### Throughput")
    col1, col2 = st.columns(2)

    with col1:
        qps = metrics.get("qps", 0)
        st.metric("QPS", f"{qps:.2f} req/s")

    with col2:
        tps = metrics.get("tps", 0)
        st.metric("TPS", f"{tps:.2f} tokens/s")

    # Latency
    st.markdown("#### Latency")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("TTFT", f"{metrics.get('ttft_ms', 0):.1f} ms")

    with col2:
        st.metric("P50", f"{metrics.get('latency_p50_ms', 0):.1f} ms")

    with col3:
        st.metric("P90", f"{metrics.get('latency_p90_ms', 0):.1f} ms")

    with col4:
        st.metric("P99", f"{metrics.get('latency_p99_ms', 0):.1f} ms")

    # Reliability
    st.markdown("#### Reliability")
    col1, col2 = st.columns(2)

    with col1:
        success_rate = metrics.get("success_rate", 0)
        st.metric("Success Rate", f"{success_rate * 100:.1f}%")

    with col2:
        error_rate = metrics.get("error_rate", 0)
        st.metric("Error Rate", f"{error_rate * 100:.1f}%")

    # vLLM Metrics
    vllm_metrics = report.get("vllm_metrics", {})
    if vllm_metrics:
        st.markdown("#### vLLM Metrics")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Avg Batch Size", f"{vllm_metrics.get('batch_size', 0):.1f}")

        with col2:
            st.metric("KV Cache Usage", f"{vllm_metrics.get('kv_cache_usage', 0) * 100:.1f}%")

        with col3:
            st.metric("GPU Util", f"{vllm_metrics.get('gpu_utilization', 0) * 100:.1f}%")

    # Summary table
    st.markdown("#### Full Summary")
    table_md = "| Metric | Value |\n|--------|-------|\n"
    table_md += f"| QPS | {metrics.get('qps', 0):.2f} |\n"
    table_md += f"| TPS | {metrics.get('tps', 0):.2f} |\n"
    table_md += f"| TTFT (ms) | {metrics.get('ttft_ms', 0):.1f} |\n"
    table_md += f"| P50 (ms) | {metrics.get('latency_p50_ms', 0):.1f} |\n"
    table_md += f"| P90 (ms) | {metrics.get('latency_p90_ms', 0):.1f} |\n"
    table_md += f"| P99 (ms) | {metrics.get('latency_p99_ms', 0):.1f} |\n"
    table_md += f"| Success Rate | {metrics.get('success_rate', 0) * 100:.1f}% |\n"
    table_md += f"| Error Rate | {metrics.get('error_rate', 0) * 100:.1f}% |\n"
    st.markdown(table_md)

    # Report file
    if report.get("report_file"):
        st.markdown("#### 📁 Report Saved")
        st.code(report["report_file"])
