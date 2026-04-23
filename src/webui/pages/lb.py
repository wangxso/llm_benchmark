"""Load testing page"""

import streamlit as st
import json
import time
from pathlib import Path
from typing import Dict, Optional
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.webui.pages.providers import load_providers, Provider


def render_lb_page():
    st.header("⚡ Load Testing")
    st.markdown("Run load tests against your LLM API endpoints.")

    # Load providers
    providers = load_providers()

    if not providers:
        st.warning("No providers configured. Please add providers in Settings first.")
        if st.button("Go to Settings"):
            st.session_state['nav_to_settings'] = True
            st.rerun()
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

    if st.button("🚀 Run Load Test", type="primary", use_container_width=True):
        if not selected_provider:
            st.error("No provider selected")
            return

        # API key is optional for local vLLM servers
        if not selected_provider.api_key and "localhost" not in selected_provider.base_url and "127.0.0.1" not in selected_provider.base_url:
            st.warning(f"Provider '{selected_provider.name}' has no API key configured. This may cause authentication errors.")

        run_load_test(
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


def run_load_test(
    api_base_url: str,
    model: str,
    api_key: Optional[str],
    api_type: str,
    stream: bool,
    scenario: str,
    base_concurrency: int,
    duration: int,
    max_tokens: int,
    dataset_mode: str,
):
    """Execute the load test"""

    from src.config import load_config
    from src.scenario.manager import ScenarioManager
    from src.load.generator import LoadGenerator
    from src.load.controller import TrafficController
    from src.client.openai_client import OpenAIClient
    from src.metrics.collector import MetricsCollector

    # Build config from UI inputs
    config = {
        "vllm": {
            "host": api_base_url.rstrip("/").split("//")[1].split("/")[0].split(":")[0] if "://" in api_base_url else "localhost",
            "port": int(api_base_url.rstrip("/").split("//")[1].split("/")[0].split(":")[-1]) if ":" in api_base_url.split("//")[-1].split("/")[0] else 8000,
            "model": model,
        },
        "load": {
            "name": f"{scenario}_test",
            "type": scenario,
            "base_concurrency": base_concurrency,
            "duration": duration,
        },
        "request": {
            "max_tokens": max_tokens,
            "stream": stream,
        },
        "dataset": {
            "mode": dataset_mode,
        },
    }

    # Add API key to headers if provided
    if api_key:
        config.setdefault("api", {})["key"] = api_key

    progress_container = st.container()

    with progress_container:
        status = st.empty()
        progress_bar = st.progress(0)

        try:
            status.info("🔧 Initializing load test...")

            manager = ScenarioManager(config)
            generator = LoadGenerator(config)
            controller = TrafficController(config)
            client = OpenAIClient(config)
            collector = MetricsCollector(config)

            status.info("📊 Starting metrics collection...")
            collector.start()

            status.info(f"⚡ Running {scenario} load test at {base_concurrency} concurrency for {duration}s...")
            progress_bar.progress(0.3)

            results = controller.run(scenario, generator, client)

            progress_bar.progress(0.8)
            status.info("📈 Generating report...")

            collector.stop()

            from src.report.generator import ReportGenerator
            report_gen = ReportGenerator(config)
            report = report_gen.generate(results, collector.get_metrics())

            progress_bar.progress(1.0)
            status.empty()
            progress_bar.empty()

            # Display results
            display_load_test_results(report)

        except Exception as e:
            status.empty()
            progress_bar.empty()
            st.error(f"Load test failed: {str(e)}")
            import traceback
            with st.expander("Show error details"):
                st.code(traceback.format_exc())


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
