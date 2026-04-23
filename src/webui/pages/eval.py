"""Benchmark evaluation page"""

import streamlit as st
import asyncio
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.eval import get_benchmark, list_benchmarks, EvalRunner
from src.webui.pages.providers import load_providers, Provider


def render_eval_page():
    st.header("📊 Benchmark Evaluation")
    st.markdown("Run benchmark evaluations on your models.")

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
                key="eval_provider_select",
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
                key="eval_model",
                help="Leave as default or enter a different model name"
            )

    # HF Token (optional)
    hf_token = st.text_input("HuggingFace Token (for gated datasets)", type="password", key="eval_hf_token")

    # Request Settings
    with st.expander("⚙️ Request Settings", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            concurrency = st.slider("Concurrency", 1, 32, 8, key="eval_concurrency")
            timeout = st.slider("Timeout (s)", 10, 300, 60, key="eval_timeout")

        with col2:
            rate_limit = st.number_input("Rate Limit (RPM)", min_value=0, value=0,
                                          help="0 = unlimited", key="eval_rate_limit")
            max_samples = st.number_input("Max Samples", min_value=0, value=0,
                                           help="0 = all samples", key="eval_samples")

        with col3:
            offline = st.checkbox("Offline Mode", value=False,
                                   help="Use cached datasets only", key="eval_offline")
            output_dir = st.text_input("Output Directory", value="./results", key="eval_output")

    # Benchmark Selection
    st.markdown("#### Benchmark Selection")
    benchmarks = list_benchmarks()

    # Get benchmark info
    benchmark_info = {}
    for name in benchmarks:
        try:
            benchmark_cls = get_benchmark(name)
            benchmark = benchmark_cls()
            benchmark_info[name] = {
                "name": benchmark.name,
                "description": benchmark.description,
                "requires_auth": getattr(benchmark, "requires_auth", False),
            }
        except:
            benchmark_info[name] = {
                "name": name,
                "description": "Unknown",
                "requires_auth": False,
            }

    col1, col2 = st.columns([1, 2])

    with col1:
        selected_benchmark = st.selectbox(
            "Select Benchmark",
            benchmarks,
            key="eval_benchmark",
            format_func=lambda x: f"{x} - {benchmark_info[x]['description'][:30]}..."
        )

    with col2:
        prompt_style = st.selectbox(
            "Prompt Style",
            ["cot", "mmlu_pro", "zero_shot", "few_shot", "zero_shot_cn", "few_shot_cn", "cot_cn"],
            index=0,  # Default to COT
            key="eval_prompt_style",
            help="COT/MMLU_Pro: Think step by step, output 'The answer is (X)'"
        )

    # Show benchmark info
    if selected_benchmark:
        info = benchmark_info[selected_benchmark]
        st.caption(f"**{info['name']}**: {info['description']}")
        if info["requires_auth"]:
            st.warning("⚠️ This dataset requires HuggingFace authentication")

    # Run button
    st.markdown("---")

    if st.button("🚀 Run Evaluation", type="primary", use_container_width=True):
        if not selected_provider:
            st.error("No provider selected")
            return

        if not selected_provider.api_key:
            st.error(f"Provider '{selected_provider.name}' has no API key configured")
            return

        if not model:
            st.error("Please enter a model name")
            return

        # Run evaluation
        run_evaluation(
            benchmark_name=selected_benchmark,
            api_type=selected_provider.api_type,
            api_base_url=selected_provider.base_url,
            api_key=selected_provider.api_key,
            model=model,
            hf_token=hf_token if hf_token else None,
            concurrency=concurrency,
            timeout=timeout,
            rate_limit=float(rate_limit) if rate_limit > 0 else 0,
            max_samples=max_samples if max_samples > 0 else None,
            offline=offline,
            prompt_style=prompt_style,
            output_dir=output_dir,
        )


def run_evaluation(
    benchmark_name: str,
    api_type: str,
    api_base_url: str,
    api_key: str,
    model: str,
    hf_token: Optional[str],
    concurrency: int,
    timeout: int,
    rate_limit: float,
    max_samples: Optional[int],
    offline: bool,
    prompt_style: str,
    output_dir: str,
):
    """Run the evaluation with progress tracking"""

    # Create progress placeholders
    progress_container = st.container()

    with progress_container:
        st.info(f"Starting evaluation: {benchmark_name}")

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Load benchmark
            status_text.text("Loading benchmark...")
            benchmark_cls = get_benchmark(benchmark_name)
            benchmark = benchmark_cls()

            # Create runner
            runner = EvalRunner(
                benchmark=benchmark,
                model=model,
                concurrency=concurrency,
                timeout=timeout,
                rate_limit=rate_limit,
                offline=offline,
                hf_token=hf_token,
                api_base_url=api_base_url,
                api_key=api_key,
                api_type=api_type,
            )

            # Run evaluation
            status_text.text("Running evaluation...")

            # Run async evaluation with progress
            start_time = time.time()

            # We need to run the actual async evaluation
            report = run_eval_with_progress(
                runner=runner,
                prompt_style=prompt_style,
                max_samples=max_samples,
                output_dir=output_dir,
                progress_bar=progress_bar,
                status_text=status_text,
            )

            elapsed = time.time() - start_time

            # Display results
            progress_bar.progress(1.0)
            status_text.empty()

            display_eval_results(report, elapsed)

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Evaluation failed: {str(e)}")
            import traceback
            with st.expander("Show error details"):
                st.code(traceback.format_exc())


def run_eval_with_progress(
    runner: EvalRunner,
    prompt_style: str,
    max_samples: Optional[int],
    output_dir: str,
    progress_bar,
    status_text,
) -> Dict:
    """Run evaluation with UI progress updates"""

    # Run the evaluation
    report = runner.run(
        prompt_style=prompt_style,
        max_samples=max_samples,
        output_dir=output_dir,
    )

    return report


def display_eval_results(report: Dict, elapsed: float):
    """Display evaluation results"""

    st.success(f"✅ Evaluation completed in {elapsed:.1f}s")

    # Key metrics - now shows attempted vs successful
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        accuracy = report.get('overall_accuracy', 0) * 100
        st.metric("Accuracy", f"{accuracy:.2f}%")

    with col2:
        correct = report.get('correct', 0)
        total = report.get('total_questions', 0)
        successful = report.get('successful_count', total)
        st.metric("Correct", f"{correct}/{successful}")

    with col3:
        attempted = report.get('attempted_count', total)
        failed = report.get('failed_count', 0)
        st.metric("Success/Failed", f"{successful}/{failed}")

    with col4:
        model = report.get('model', 'unknown')
        st.metric("Model", model)

    # Errors
    if failed > 0:
        st.markdown("#### Errors")
        error_types = report.get("error_types", {})
        error_details = report.get("error_details", {})

        for err_type, count in sorted(error_types.items(), key=lambda x: -x[1]):
            st.write(f"- **{err_type}**: {count}")
            if err_type in error_details:
                for detail, detail_count in sorted(error_details[err_type].items(), key=lambda x: -x[1])[:3]:
                    st.write(f"  - [{detail_count}x] {detail[:80]}...")

    # By Subject
    subjects = report.get("subjects", {})
    if subjects:
        st.markdown("#### Results by Subject")

        # Use markdown table instead of dataframe
        table_md = "| Subject | Accuracy | Correct |\n|---------|----------|--------|\n"
        for subject, stats in subjects.items():
            table_md += f"| {subject} | {stats['accuracy'] * 100:.1f}% | {stats['correct']}/{stats['total']} |\n"
        st.markdown(table_md)

    # By Category
    categories = report.get("categories", {})
    if categories:
        st.markdown("#### Results by Category")

        table_md = "| Category | Accuracy | Correct |\n|----------|----------|--------|\n"
        for cat, stats in sorted(categories.items()):
            if cat != "Average":
                table_md += f"| {cat} | {stats['accuracy'] * 100:.1f}% | {stats['correct']}/{stats['total']} |\n"
        st.markdown(table_md)

    # Answer Comparison Table
    details = report.get("details", [])
    if details:
        st.markdown("#### Answer Comparison")
        st.caption("Shows actual vs predicted answers for debugging answer extraction")

        # Filter successful results
        successful_details = [d for d in details if d.get("success")]

        # Show first N results
        show_count = min(30, len(successful_details))

        # Create comparison table - show more response text
        comp_md = "| # | Subject | Actual | Pred | Response |\n|---|---------|--------|------|----------|\n"

        for i, d in enumerate(successful_details[:show_count]):
            actual = d.get("actual", "?")
            predicted = d.get("predicted", "N/A") or "N/A"
            subject = d.get("subject", "unknown")[:12]
            response = d.get("response", "")[:80].replace("\n", " ").replace("|", "\\|")

            marker = "✓" if actual == predicted else "✗"
            comp_md += f"| {i+1} | {subject} | **{actual}** | {predicted} | {response}... {marker} |\n"

        st.markdown(comp_md)

        if len(successful_details) > show_count:
            st.caption(f"... and {len(successful_details) - show_count} more results")

    # Report file
    if report.get("report_file"):
        st.markdown("#### Report File")
        st.code(report["report_file"])
