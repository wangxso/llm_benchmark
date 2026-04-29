"""Auto-Tuning page for vLLM parameter optimization."""

import streamlit as st
import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.autotune import (
    AutoTuner,
    SearchSpace,
    ParameterRange,
    TuningConfig,
    TuningResult,
    Objective,
    generate_deploy_template,
    save_tuning_report,
)
from src.autotune.config import get_default_vllm_space, get_high_throughput_space, get_low_latency_space
from src.autotune.search import create_search_strategy
from src.device import list_devices, detect_device
from src.webui.task_manager import start_task, get_active_tasks, get_all_tasks, stop_task


def _make_session_dir(base_dir: str, model_path: str) -> str:
    """Create a unique session directory: {base}/{model_name}_{timestamp}/"""
    from datetime import datetime
    model_name = Path(model_path).name or "model"
    # Sanitize model name
    model_name = model_name.replace("/", "_").replace("\\", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(base_dir) / f"{model_name}_{timestamp}"
    session_dir.mkdir(parents=True, exist_ok=True)
    return str(session_dir)


def _run_autotune_background(task_id: str, stop_event, **kwargs):
    """Background target for auto-tuning."""
    from src.webui.task_manager import update_progress, complete_task, _fail
    from src.autotune.optimizer import TuningProgress

    try:
        update_progress(task_id, 0.0, "Starting auto-tuning...")

        # Create unique session directory
        session_dir = _make_session_dir(kwargs["output_dir"], kwargs["model_path"])

        def progress_callback(progress: TuningProgress):
            pct = progress.completed_trials / progress.total_trials if progress.total_trials > 0 else 0
            text = f"Trial {progress.current_trial}/{progress.total_trials} (best: {progress.best_score:.2f})"
            update_progress(task_id, pct, text)

        tuner = AutoTuner(
            model_path=kwargs["model_path"],
            gpu_ids=kwargs["gpu_ids"],
            device=kwargs["device"],
            search_space=kwargs["search_space"],
            strategy=kwargs["strategy"],
            objective=kwargs["objective"],
            max_trials=kwargs["max_trials"],
            startup_timeout=kwargs["startup_timeout"],
            seed=kwargs["seed"],
            verbose=True,
            log_dir=f"{session_dir}/logs",
            progress_callback=progress_callback,
        )

        result = tuner.run(stop_event=stop_event)

        # Save results into session directory
        save_tuning_report(tuner.results, Path(session_dir) / "tuning_report.json")
        if result and result.error is None:
            generate_deploy_template(
                result.config,
                Path(session_dir) / "best_config.yaml",
                model_path=kwargs["model_path"],
                gpu_ids=kwargs["gpu_ids"],
            )

        complete_task(task_id, {
            "best": result,
            "results": tuner.results,
            "output_dir": session_dir,
        })
    except Exception as e:
        _fail(task_id, str(e))


def render_autotune_page():
    """Render the Auto-Tuning page."""
    st.header("🔧 Auto-Tuning")
    st.markdown("Automatically find optimal vLLM parameters using Bayesian optimization.")

    # Initialize session state
    if "autotune_results" not in st.session_state:
        st.session_state["autotune_results"] = []
    if "autotune_best" not in st.session_state:
        st.session_state["autotune_best"] = None
    if "autotune_progress" not in st.session_state:
        st.session_state["autotune_progress"] = {}

    # Show running task status
    active = {k: v for k, v in get_active_tasks().items() if v.task_type == "autotune"}
    if active:
        for tid, task in active.items():
            st.info(f"⏳ Tuning running: **{task.label}** — {task.progress_text} ({task.elapsed_str()})")
            st.progress(task.progress)
            if st.button("⏹ Stop", key=f"stop_autotune_{tid}"):
                stop_task(tid)
                st.rerun()
        st.caption("Navigate to **Results** page to see progress. You can continue using other pages.")

    # Check for completed tasks and update session state
    completed = {k: v for k, v in get_all_tasks().items()
                 if v.task_type == "autotune" and v.status == "completed" and v.result}
    for tid, task in completed.items():
        if task.result and isinstance(task.result, dict):
            if task.result.get("results"):
                st.session_state["autotune_results"] = task.result["results"]
            if task.result.get("best"):
                st.session_state["autotune_best"] = task.result["best"]

    # Tabs for different sections
    tab1, tab2, tab3 = st.tabs(["⚙️ Configuration", "🎯 Run Tuning", "📊 Results"])

    with tab1:
        render_config_section()

    with tab2:
        render_run_section()

    with tab3:
        render_results_section()


def render_config_section():
    """Render configuration section."""
    st.subheader("Basic Configuration")

    # Model and GPU settings
    col1, col2 = st.columns(2)

    with col1:
        model_path = st.text_input(
            "Model Path",
            value="",
            placeholder="/path/to/model or model_name",
            help="Local path to model or model name for API",
            key="autotune_model_path"
        )

        gpu_ids = st.text_input(
            "GPU IDs",
            value="0",
            placeholder="0 or 0,1,2,3",
            help="Comma-separated GPU IDs to use",
            key="autotune_gpu_ids"
        )

    with col2:
        # Device type selector
        device_options = {"auto": "Auto (detect)", **list_devices()}
        device_names = list(device_options.keys())

        device = st.selectbox(
            "GPU Vendor",
            options=device_names,
            format_func=lambda x: device_options.get(x, x),
            index=0,  # "auto" is first
            help="Select GPU vendor type. 'Auto' will detect automatically.",
            key="autotune_device"
        )

        # Optimization settings
        objective = st.selectbox(
            "Optimization Objective",
            ["throughput", "latency", "balanced"],
            index=0,
            help="throughput: maximize tokens/s | latency: minimize P99 | balanced: trade-off",
            key="autotune_objective"
        )

        strategy = st.selectbox(
            "Search Strategy",
            ["bayesian", "random", "grid"],
            index=0,
            help="Bayesian: smart search | Random: baseline | Grid: exhaustive",
            key="autotune_strategy"
        )

    # Search space preset
    st.subheader("Search Space")

    preset = st.selectbox(
        "Search Space Preset",
        ["Default", "High Throughput", "Low Latency", "Custom"],
        index=0,
        help="Pre-defined search space configurations",
        key="autotune_preset"
    )

    if preset == "Custom":
        render_custom_search_space()
    else:
        # Show preset description
        preset_info = {
            "Default": "Balanced search space for general use cases",
            "High Throughput": "Optimized for maximum tokens/s, allows higher latency",
            "Low Latency": "Optimized for minimum latency, may reduce throughput"
        }
        st.info(preset_info.get(preset, ""))

        # Show parameter ranges
        gpu_count = len(gpu_ids.split(",")) if gpu_ids else 1
        if preset == "Default":
            space = get_default_vllm_space(gpu_count)
        elif preset == "High Throughput":
            space = get_high_throughput_space(gpu_count)
        else:
            space = get_low_latency_space(gpu_count)

        display_search_space(space)

    # Load test settings
    st.subheader("Load Test Settings")

    col1, col2, col3 = st.columns(3)

    with col1:
        max_trials = st.number_input(
            "Max Trials",
            min_value=1,
            max_value=100,
            value=10,
            help="Number of configurations to test",
            key="autotune_max_trials"
        )

        concurrency = st.number_input(
            "Concurrency",
            min_value=1,
            max_value=1000,
            value=100,
            help="Number of concurrent requests during load test",
            key="autotune_concurrency"
        )

    with col2:
        duration = st.number_input(
            "Duration (seconds)",
            min_value=10,
            max_value=600,
            value=60,
            help="Duration of each load test",
            key="autotune_duration"
        )

        startup_timeout = st.number_input(
            "Startup Timeout (seconds)",
            min_value=60,
            max_value=600,
            value=300,
            help="Timeout for vLLM instance startup",
            key="autotune_startup_timeout"
        )

    with col3:
        seed = st.number_input(
            "Random Seed",
            min_value=0,
            max_value=99999,
            value=42,
            help="Seed for reproducibility",
            key="autotune_seed"
        )

        output_dir = st.text_input(
            "Output Directory",
            value="./results/autotune",
            key="autotune_output_dir"
        )


def render_custom_search_space():
    """Render custom search space configuration."""
    st.markdown("**Custom Parameter Ranges**")

    # GPU memory utilization
    col1, col2, col3 = st.columns(3)
    with col1:
        gpu_mem_min = st.number_input(
            "GPU Memory Min",
            min_value=0.5,
            max_value=1.0,
            value=0.7,
            step=0.05,
            key="autotune_gpu_mem_min"
        )
    with col2:
        gpu_mem_max = st.number_input(
            "GPU Memory Max",
            min_value=0.5,
            max_value=1.0,
            value=0.95,
            step=0.05,
            key="autotune_gpu_mem_max"
        )
    with col3:
        gpu_mem_step = st.number_input(
            "GPU Memory Step",
            min_value=0.01,
            max_value=0.2,
            value=0.05,
            step=0.01,
            key="autotune_gpu_mem_step"
        )

    # Max model len
    st.markdown("**Max Model Length Options**")
    max_model_len_options = st.multiselect(
        "Select values",
        [2048, 4096, 8192, 16384, 32768, 65536],
        default=[4096, 8192, 16384],
        key="autotune_max_model_len_opts"
    )

    # Max num seqs
    col1, col2, col3 = st.columns(3)
    with col1:
        max_seqs_min = st.number_input(
            "Max Num Seqs Min",
            min_value=8,
            max_value=512,
            value=32,
            step=8,
            key="autotune_max_seqs_min"
        )
    with col2:
        max_seqs_max = st.number_input(
            "Max Num Seqs Max",
            min_value=8,
            max_value=512,
            value=256,
            step=8,
            key="autotune_max_seqs_max"
        )
    with col3:
        max_seqs_step = st.number_input(
            "Max Num Seqs Step",
            min_value=8,
            max_value=128,
            value=32,
            step=8,
            key="autotune_max_seqs_step"
        )


def display_search_space(space: SearchSpace):
    """Display search space parameters."""
    param_data = []
    for param in space.parameters:
        if param.values:
            range_str = f"[{', '.join(map(str, param.values))}]"
        else:
            range_str = f"[{param.min_val} - {param.max_val}]"
            if param.step:
                range_str += f" (step: {param.step})"

        param_data.append({
            "Parameter": param.name,
            "Type": param.param_type,
            "Range": range_str
        })

    st.dataframe(param_data, width="stretch", hide_index=True)

    # Show constraints
    if space.constraints:
        st.markdown("**Constraints**")
        constraint_md = "| Metric | Min | Max |\n|--------|-----|-----|\n"
        for name, (min_val, max_val) in space.constraints.items():
            constraint_md += f"| {name} | {min_val} | {max_val} |\n"
        st.markdown(constraint_md)


def render_run_section():
    """Render run section with progress display."""
    st.subheader("Run Auto-Tuning")

    # Validate configuration
    model_path = st.session_state.get("autotune_model_path", "")
    gpu_ids = st.session_state.get("autotune_gpu_ids", "0")

    col1, col2 = st.columns([3, 1])

    with col1:
        if not model_path:
            st.warning("Please configure model path in the Configuration tab.")

    # Check if tuning is running
    active = {k: v for k, v in get_active_tasks().items() if v.task_type == "autotune"}
    is_running = len(active) > 0

    with col2:
        run_button = st.button(
            "🚀 Start Tuning",
            type="primary",
            disabled=not model_path or is_running,
            width="stretch"
        )

    # Run tuning
    if run_button and model_path:
        _start_autotuning()

    # Display progress from completed results
    if st.session_state.get("autotune_results"):
        display_progress()

    # Display best result so far
    if st.session_state.get("autotune_best"):
        display_best_result(st.session_state["autotune_best"])


def _start_autotuning():
    """Start auto-tuning as a background task."""
    model_path = st.session_state.get("autotune_model_path", "")
    gpu_ids = st.session_state.get("autotune_gpu_ids", "0")
    device = st.session_state.get("autotune_device", "nvidia")
    if device == "auto":
        device = detect_device()
    objective = st.session_state.get("autotune_objective", "throughput")
    strategy = st.session_state.get("autotune_strategy", "bayesian")
    max_trials = st.session_state.get("autotune_max_trials", 10)
    startup_timeout = st.session_state.get("autotune_startup_timeout", 300)
    seed = st.session_state.get("autotune_seed", 42)
    output_dir = st.session_state.get("autotune_output_dir", "./results/autotune")

    # Get search space
    preset = st.session_state.get("autotune_preset", "Default")
    gpu_count = len(gpu_ids.split(",")) if gpu_ids else 1

    if preset == "Default":
        search_space = get_default_vllm_space(gpu_count)
    elif preset == "High Throughput":
        search_space = get_high_throughput_space(gpu_count)
    elif preset == "Low Latency":
        search_space = get_low_latency_space(gpu_count)
    else:
        search_space = build_custom_search_space(gpu_count)

    label = f"{strategy} ({max_trials} trials)"
    start_task(
        task_type="autotune",
        label=label,
        target_fn=_run_autotune_background,
        model_path=model_path,
        gpu_ids=gpu_ids,
        device=device,
        search_space=search_space,
        strategy=strategy,
        objective=objective,
        max_trials=max_trials,
        startup_timeout=startup_timeout,
        seed=seed,
        output_dir=output_dir,
    )
    st.success(f"✅ Auto-tuning started in background: **{label}**")
    st.caption("Go to **Results** page to monitor progress and stop if needed.")
    st.rerun()


def build_custom_search_space(gpu_count: int) -> SearchSpace:
    """Build custom search space from UI inputs."""
    gpu_mem_min = st.session_state.get("autotune_gpu_mem_min", 0.7)
    gpu_mem_max = st.session_state.get("autotune_gpu_mem_max", 0.95)
    gpu_mem_step = st.session_state.get("autotune_gpu_mem_step", 0.05)

    max_model_len_opts = st.session_state.get("autotune_max_model_len_opts", [4096, 8192, 16384])

    max_seqs_min = st.session_state.get("autotune_max_seqs_min", 32)
    max_seqs_max = st.session_state.get("autotune_max_seqs_max", 256)
    max_seqs_step = st.session_state.get("autotune_max_seqs_step", 32)

    # Determine tensor parallel values
    tp_values = [1]
    if gpu_count >= 2:
        tp_values = [1, 2]
    if gpu_count >= 4:
        tp_values = [1, 2, 4]
    if gpu_count >= 8:
        tp_values = [1, 2, 4, 8]

    return SearchSpace(
        parameters=[
            ParameterRange(
                name="gpu_memory_utilization",
                min_val=gpu_mem_min,
                max_val=gpu_mem_max,
                step=gpu_mem_step,
            ),
            ParameterRange(
                name="tensor_parallel",
                values=tp_values,
            ),
            ParameterRange(
                name="max_model_len",
                values=max_model_len_opts if max_model_len_opts else [4096, 8192, 16384],
            ),
            ParameterRange(
                name="max_num_seqs",
                min_val=max_seqs_min,
                max_val=max_seqs_max,
                step=max_seqs_step,
            ),
        ]
    )


def display_progress():
    """Display tuning progress from results."""
    results = st.session_state.get("autotune_results", [])
    if not results:
        return

    completed = len(results)
    best_score = max((r.score for r in results), default=0)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Completed Trials", completed)
    with col2:
        st.metric("Best Score", f"{best_score:.2f}")
    with col3:
        errors = sum(1 for r in results if r.error)
        st.metric("Errors", errors)


def display_best_result(result: TuningResult):
    """Display best result."""
    st.subheader("🏆 Best Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**vLLM Parameters**")
        config = result.config

        param_md = f"""
| Parameter | Value |
|-----------|-------|
| Device | {config.device} |
| GPU Memory Utilization | {config.gpu_memory_utilization} |
| Tensor Parallel | {config.tensor_parallel} |
| Max Model Length | {config.max_model_len} |
| Max Num Seqs | {config.max_num_seqs} |
"""
        st.markdown(param_md)

    with col2:
        st.markdown("**Performance Metrics**")

        metrics = result.metrics
        metric_md = f"""
| Metric | Value |
|--------|-------|
| TPS | {result.tps:.2f} tokens/s |
| QPS | {metrics.get('qps', 0):.2f} req/s |
| Latency P50 | {metrics.get('latency_p50', 0):.1f} ms |
| Latency P99 | {result.latency_p99:.1f} ms |
| Success Rate | {result.success_rate * 100:.1f}% |
"""
        st.markdown(metric_md)


def render_results_section():
    """Render results section - show history and export options."""
    st.subheader("Tuning History")

    results = st.session_state.get("autotune_results", [])

    if not results:
        st.info("No tuning results yet. Run a tuning session first.")
        return

    # Display results table
    display_results_table(results)

    # Export options
    st.subheader("Export Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📥 Download JSON Report", width="stretch"):
            download_json_report(results)

    with col2:
        if st.button("📥 Download Best Config", width="stretch"):
            download_best_config()

    with col3:
        if st.button("📥 Download CSV History", width="stretch"):
            download_csv_history(results)


def display_results_table(results: List[TuningResult]):
    """Display results in a table."""
    table_data = []

    for r in results:
        table_data.append({
            "Trial": r.trial_id,
            "Score": f"{r.score:.2f}",
            "TPS": f"{r.tps:.1f}",
            "P99 (ms)": f"{r.latency_p99:.1f}",
            "Success": f"{r.success_rate * 100:.0f}%",
            "GPU Mem": r.config.gpu_memory_utilization,
            "TP": r.config.tensor_parallel,
            "Max Len": r.config.max_model_len,
            "Max Seqs": r.config.max_num_seqs,
            "Status": "✓" if r.error is None else "✗",
        })

    st.dataframe(table_data, width="stretch", hide_index=True)


def download_json_report(results: List[TuningResult]):
    """Download JSON report."""
    import base64

    report_data = save_tuning_report(results, Path("./results/autotune/temp_report.json"))

    json_str = json.dumps(report_data, indent=2)
    b64 = base64.b64encode(json_str.encode()).decode()

    href = f'<a href="data:application/json;base64,{b64}" download="tuning_report.json">Download JSON Report</a>'
    st.markdown(href, unsafe_allow_html=True)


def download_best_config():
    """Download best configuration as YAML."""
    import base64

    best = st.session_state.get("autotune_best")
    if not best:
        st.warning("No best result available")
        return

    template = generate_deploy_template(best.config)

    import yaml
    yaml_str = yaml.dump(template, default_flow_style=False, sort_keys=False)
    b64 = base64.b64encode(yaml_str.encode()).decode()

    href = f'<a href="data:application/x-yaml;base64,{b64}" download="best_config.yaml">Download Best Config</a>'
    st.markdown(href, unsafe_allow_html=True)


def download_csv_history(results: List[TuningResult]):
    """Download CSV history."""
    import base64
    import csv
    from io import StringIO

    output = StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        "trial_id", "score", "tps", "qps", "latency_p50", "latency_p99",
        "success_rate", "gpu_memory_utilization", "tensor_parallel",
        "max_model_len", "max_num_seqs", "error"
    ])

    # Data
    for r in results:
        writer.writerow([
            r.trial_id,
            r.score,
            r.tps,
            r.metrics.get("qps", 0),
            r.metrics.get("latency_p50", 0),
            r.latency_p99,
            r.success_rate,
            r.config.gpu_memory_utilization,
            r.config.tensor_parallel,
            r.config.max_model_len,
            r.config.max_num_seqs,
            r.error or "",
        ])

    csv_str = output.getvalue()
    b64 = base64.b64encode(csv_str.encode()).decode()

    href = f'<a href="data:text/csv;base64,{b64}" download="tuning_history.csv">Download CSV History</a>'
    st.markdown(href, unsafe_allow_html=True)
