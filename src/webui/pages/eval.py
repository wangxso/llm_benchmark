"""Benchmark evaluation page"""

import streamlit as st
import asyncio
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

# Import eval module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.eval import get_benchmark, list_benchmarks, EvalRunner


def render_eval_page():
    st.header("📊 Benchmark Evaluation")
    st.markdown("Run benchmark evaluations on your models.")

    # Configuration
    with st.expander("⚙️ Configuration", expanded=True):
        # API Settings
        st.markdown("#### API Settings")
        col1, col2 = st.columns(2)

        with col1:
            api_type = st.selectbox("API Type", ["openai", "anthropic"], key="eval_api_type")
            api_base_url = st.text_input(
                "API Base URL",
                value="https://api.openai.com/v1",
                key="eval_api_url",
                help="e.g., https://api.openai.com/v1 or https://api.minimaxi.com/anthropic"
            )
            model = st.text_input("Model Name", value="gpt-4", key="eval_model")

        with col2:
            api_key = st.text_input("API Key", type="password", key="eval_api_key")
            hf_token = st.text_input("HuggingFace Token", type="password", key="eval_hf_token",
                                       help="Required for gated datasets like GPQA")

        # Request Settings
        st.markdown("#### Request Settings")
        col1, col2, col3 = st.columns(3)

        with col1:
            concurrency = st.slider("Concurrency", 1, 32, 8, key="eval_concurrency")
            timeout = st.slider("Timeout (s)", 10, 300, 60, key="eval_timeout")

        with col2:
            rate_limit = st.number_input("Rate Limit (RPS)", min_value=0, value=0,
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
            ["zero_shot", "few_shot", "cot", "zero_shot_cn", "few_shot_cn", "cot_cn"],
            key="eval_prompt_style"
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
        # Validate
        if not api_key:
            st.error("Please enter API Key")
            return
        if not api_base_url:
            st.error("Please enter API Base URL")
            return
        if not model:
            st.error("Please enter Model Name")
            return

        # Run evaluation
        run_evaluation(
            benchmark_name=selected_benchmark,
            api_type=api_type,
            api_base_url=api_base_url,
            api_key=api_key,
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
        metrics_text = st.empty()

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

    # Import needed modules
    import asyncio
    from tqdm.asyncio import tqdm_asyncio

    # Monkey patch tqdm to update streamlit progress
    original_tqdm = tqdm_asyncio

    class StreamlitTqdm:
        def __init__(self, *args, **kwargs):
            self.total = kwargs.get('total', 100)
            self.count = 0

        def update(self, n=1):
            self.count += n
            if self.total > 0:
                progress = min(self.count / self.total, 1.0)
                progress_bar.progress(progress)

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

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

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Accuracy", f"{report.get('overall_accuracy', 0) * 100:.2f}%")

    with col2:
        correct = report.get('correct', 0)
        total = report.get('total_questions', 0)
        st.metric("Correct", f"{correct}/{total}")

    with col3:
        failed = report.get('failed_count', 0)
        st.metric("Failed", failed)

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

        # Convert to DataFrame for better display
        data = []
        for subject, stats in subjects.items():
            data.append({
                "Subject": subject,
                "Accuracy": f"{stats['accuracy'] * 100:.1f}%",
                "Correct": f"{stats['correct']}/{stats['total']}",
            })

        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # By Category
    categories = report.get("categories", {})
    if categories:
        st.markdown("#### Results by Category")

        data = []
        for cat, stats in sorted(categories.items()):
            if cat != "Average":
                data.append({
                    "Category": cat,
                    "Accuracy": f"{stats['accuracy'] * 100:.1f}%",
                    "Correct": f"{stats['correct']}/{stats['total']}",
                })

        if data:
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True, hide_index=True)

    # Report file
    if report.get("report_file"):
        st.markdown("#### Report File")
        st.code(report["report_file"])
