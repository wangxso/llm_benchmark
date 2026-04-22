"""Results browsing page"""

import streamlit as st
import json
import glob
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np


def render_results_page():
    st.header("📁 Results")
    st.markdown("Browse and analyze evaluation and load test results.")

    # Results directory
    results_dir = st.text_input("Results Directory", value="./results", key="results_dir")

    if not Path(results_dir).exists():
        st.warning(f"Directory '{results_dir}' does not exist")
        return

    # Scan for results
    result_files = scan_results(results_dir)

    if not result_files:
        st.info("No results found. Run some evaluations or load tests first.")
        return

    # Create tabs for List and Detail views
    tab_list, tab_detail = st.tabs(["📋 Results List", "📊 Result Detail"])

    with tab_list:
        # Filter by type
        col1, col2 = st.columns([1, 3])

        with col1:
            result_type = st.selectbox(
                "Result Type",
                ["All", "Evaluation", "Load Test"],
                key="results_type_filter"
            )

        with col2:
            search = st.text_input("Search", placeholder="Filter by model, benchmark...", key="results_search")

        # Filter results
        filtered_files = filter_results(result_files, result_type, search)

        st.markdown(f"**Found {len(filtered_files)} results**")

        # Display results in a more compact table format
        for file_info in filtered_files[:20]:
            col1, col2, col3, col4 = st.columns([2, 1, 2, 1])

            with col1:
                st.markdown(f"**{file_info['name']}**")

            with col2:
                st.caption(file_info['type'])

            with col3:
                if file_info.get('model'):
                    st.caption(f"🤖 {file_info['model']}")
                st.caption(f"📅 {file_info['timestamp'][:16] if file_info.get('timestamp') else 'N/A'}")

            with col4:
                if st.button("📊 View", key=f"view_{file_info['path']}"):
                    st.session_state['selected_result'] = file_info['path']
                    st.rerun()

    with tab_detail:
        if 'selected_result' in st.session_state and st.session_state['selected_result']:
            # Clear button
            if st.button("⬅️ Back to List"):
                st.session_state['selected_result'] = None
                st.rerun()

            show_result_detail(st.session_state['selected_result'])
        else:
            st.info("👈 Select a result from the 'Results List' tab to view details")


def scan_results(results_dir: str) -> List[Dict]:
    """Scan results directory for result files"""
    results = []

    # Scan for eval results
    eval_files = glob.glob(f"{results_dir}/eval_*.json")
    for f in eval_files:
        if "_details.json" in f:
            continue
        try:
            with open(f, "r") as file:
                data = json.load(file)
                results.append({
                    "path": f,
                    "name": f"📊 {data.get('benchmark', 'Eval')}",
                    "type": "Evaluation",
                    "timestamp": data.get("timestamp", ""),
                    "model": data.get("model", ""),
                    "data": data,
                })
        except:
            pass

    # Scan for load test results
    benchmark_files = glob.glob(f"{results_dir}/benchmark_*.json")
    for f in benchmark_files:
        try:
            with open(f, "r") as file:
                data = json.load(file)
                results.append({
                    "path": f,
                    "name": f"⚡ Load Test",
                    "type": "Load Test",
                    "timestamp": data.get("test_info", {}).get("timestamp", ""),
                    "model": data.get("test_info", {}).get("model", ""),
                    "data": data,
                })
        except:
            pass

    # Sort by timestamp descending
    results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    return results


def filter_results(results: List[Dict], result_type: str, search: str) -> List[Dict]:
    """Filter results by type and search query"""
    filtered = []

    for r in results:
        # Filter by type
        if result_type != "All":
            if result_type == "Evaluation" and r["type"] != "Evaluation":
                continue
            if result_type == "Load Test" and r["type"] != "Load Test":
                continue

        # Filter by search
        if search:
            search_lower = search.lower()
            if (search_lower not in r.get("name", "").lower() and
                search_lower not in r.get("model", "").lower() and
                search_lower not in r.get("type", "").lower()):
                continue

        filtered.append(r)

    return filtered


def show_result_detail(file_path: str):
    """Show detailed result in a modal/expander"""
    with st.expander(f"📄 {Path(file_path).name}", expanded=True):
        with open(file_path, "r") as f:
            data = json.load(f)

        # Determine result type
        if "benchmark" in data:
            # Evaluation result
            show_eval_detail(data)
        else:
            # Load test result
            show_load_test_detail(data)


def show_eval_detail(data: Dict):
    """Show evaluation result details with charts"""
    st.markdown("## 📊 Evaluation Details")

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        accuracy = data.get('overall_accuracy', 0) * 100
        st.metric("Accuracy", f"{accuracy:.2f}%")

    with col2:
        correct = data.get('correct', 0)
        total = data.get('total_questions', 0)
        st.metric("Correct", f"{correct}/{total}")

    with col3:
        failed = data.get('failed_count', 0)
        st.metric("Failed", failed)

    with col4:
        st.metric("Model", data.get("model", "unknown"))

    st.markdown("")  # Spacing

    # Accuracy pie chart and error breakdown
    subjects = data.get("subjects", {})
    error_types = data.get("error_types", {})

    if subjects or error_types:
        col1, col2 = st.columns(2)

        with col1:
            # Subject accuracy bar chart
            if subjects:
                st.markdown("### 📈 Accuracy by Subject")

                fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
                names = list(subjects.keys())
                accuracies = [s['accuracy'] * 100 for s in subjects.values()]

                colors = ['#4ecdc4' if acc >= 50 else '#ff6b6b' for acc in accuracies]
                bars = ax.bar(range(len(names)), accuracies, color=colors)
                ax.set_xticks(range(len(names)))
                ax.set_xticklabels(names, rotation=45, ha='right')
                ax.set_ylabel('Accuracy (%)')
                ax.set_ylim(0, 100)
                ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)

                # Add value labels on bars
                for bar, acc in zip(bars, accuracies):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)

                plt.tight_layout()
                st.pyplot(fig, clear_figure=True)

        with col2:
            # Error pie chart or correct/incorrect pie
            if error_types and data.get("failed_count", 0) > 0:
                st.markdown("### 🥧 Error Distribution")

                fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
                err_names = list(error_types.keys())
                err_counts = list(error_types.values())

                colors = plt.cm.Set3(np.linspace(0, 1, len(err_names)))
                wedges, texts, autotexts = ax.pie(err_counts, labels=err_names,
                                                   autopct='%1.1f%%', colors=colors)

                plt.tight_layout()
                st.pyplot(fig, clear_figure=True)

            elif subjects:
                st.markdown("### 🎯 Correct vs Incorrect")

                correct_count = data.get('correct', 0)
                incorrect_count = data.get('total_questions', 0) - correct_count

                if correct_count + incorrect_count > 0:
                    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
                    ax.pie([correct_count, incorrect_count],
                          labels=['Correct', 'Incorrect'],
                          autopct='%1.1f%%',
                          colors=['#4ecdc4', '#ff6b6b'])

                    plt.tight_layout()
                    st.pyplot(fig, clear_figure=True)

    st.markdown("")  # Spacing

    # Detailed error list
    if data.get("failed_count", 0) > 0:
        st.markdown("### ❌ Error Details")
        error_details = data.get("error_details", {})

        for err_type, count in sorted(error_types.items(), key=lambda x: -x[1]):
            with st.expander(f"**{err_type}** ({count} errors)", expanded=False):
                if err_type in error_details:
                    for detail, detail_count in sorted(error_details[err_type].items(),
                                                        key=lambda x: -x[1])[:5]:
                        st.write(f"- [{detail_count}x] {detail[:100]}...")

    # Subject details table
    if subjects:
        st.markdown("### 📋 Detailed Results by Subject")

        table_md = "| Subject | Accuracy | Correct | Total |\n|---------|----------|---------|-------|\n"
        for subject, stats in sorted(subjects.items(), key=lambda x: -x[1]['accuracy']):
            table_md += f"| {subject} | {stats['accuracy'] * 100:.1f}% | {stats['correct']} | {stats['total']} |\n"
        st.markdown(table_md)

    # Category breakdown
    categories = data.get("categories", {})
    if categories:
        st.markdown("### 📁 Results by Category")

        cat_data = []
        for cat, stats in sorted(categories.items()):
            if cat != "Average":
                cat_data.append({
                    "Category": cat,
                    "Accuracy": stats['accuracy'] * 100,
                    "Correct": stats['correct'],
                    "Total": stats['total']
                })

        if cat_data:
            col1, col2 = st.columns([2, 1])

            with col1:
                fig, ax = plt.subplots(figsize=(8, 3), dpi=100)
                cat_names = [c['Category'] for c in cat_data]
                cat_accs = [c['Accuracy'] for c in cat_data]

                colors = ['#4ecdc4' if acc >= 50 else '#ff6b6b' for acc in cat_accs]
                bars = ax.bar(range(len(cat_names)), cat_accs, color=colors)
                ax.set_xticks(range(len(cat_names)))
                ax.set_xticklabels(cat_names, rotation=45, ha='right')
                ax.set_ylabel('Accuracy (%)')
                ax.set_ylim(0, 100)
                ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)

                for bar, acc in zip(bars, cat_accs):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)

                plt.tight_layout()
                st.pyplot(fig, clear_figure=True)

            with col2:
                table_md = "| Category | Acc |\n|----------|-----|\n"
                for cat in cat_data:
                    table_md += f"| {cat['Category']} | {cat['Accuracy']:.1f}% |\n"
                st.markdown(table_md)


def show_load_test_detail(data: Dict):
    """Show load test result details with charts"""
    st.markdown("## ⚡ Load Test Details")

    metrics = data.get("metrics", {})

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("QPS", f"{metrics.get('qps', 0):.2f} req/s")

    with col2:
        st.metric("TPS", f"{metrics.get('tps', 0):.2f} tok/s")

    with col3:
        st.metric("P99 Latency", f"{metrics.get('latency_p99_ms', 0):.1f} ms")

    with col4:
        st.metric("Success Rate", f"{metrics.get('success_rate', 0) * 100:.1f}%")

    st.markdown("")  # Spacing

    # Latency distribution
    st.markdown("### 📊 Latency Distribution")

    latency_cols = ['TTFT', 'P50', 'P90', 'P99']
    latency_values = [
        metrics.get('ttft_ms', 0),
        metrics.get('latency_p50_ms', 0),
        metrics.get('latency_p90_ms', 0),
        metrics.get('latency_p99_ms', 0)
    ]

    col1, col2 = st.columns([2, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
        bars = ax.bar(latency_cols, latency_values, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'])
        ax.set_ylabel('Latency (ms)')
        ax.set_title('Latency Percentiles')

        for bar, val in zip(bars, latency_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

    with col2:
        table_md = "| Metric | Value |\n|--------|-------|\n"
        for i, metric in enumerate(latency_cols):
            table_md += f"| {metric} | {latency_values[i]:.1f} ms |\n"
        st.markdown(table_md)

    st.markdown("")  # Spacing

    # Success/Error breakdown
    st.markdown("### 🎯 Success vs Error Rate")

    success_rate = metrics.get('success_rate', 0)
    error_rate = metrics.get('error_rate', 0)

    col1, col2 = st.columns([1, 2])

    with col1:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
        ax.pie([success_rate * 100, error_rate * 100],
              labels=['Success', 'Error'],
              autopct='%1.1f%%',
              colors=['#4ecdc4', '#ff6b6b'],
              explode=(0.05, 0))
        ax.set_title('Request Success Rate')

        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

    with col2:
        # vLLM metrics if available
        vllm_metrics = data.get("vllm_metrics", {})
        if vllm_metrics:
            st.markdown("**vLLM Metrics**")

            # Create a horizontal bar chart for vLLM metrics
            fig, ax = plt.subplots(figsize=(6, 3), dpi=100)
            vllm_names = ['KV Cache\nUsage', 'GPU\nUtilization']
            vllm_values = [
                vllm_metrics.get('kv_cache_usage', 0) * 100,
                vllm_metrics.get('gpu_utilization', 0) * 100
            ]

            bars = ax.barh(vllm_names, vllm_values, color=['#9b59b6', '#1abc9c'])
            ax.set_xlim(0, 100)
            ax.set_xlabel('Percentage (%)')

            for bar, val in zip(bars, vllm_values):
                ax.text(val + 2, bar.get_y() + bar.get_height()/2,
                       f'{val:.1f}%', va='center', fontsize=10)

            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)

            st.metric("Avg Batch Size", f"{vllm_metrics.get('batch_size', 0):.1f}")
        else:
            st.info("No vLLM-specific metrics available")

    # Full summary
    st.markdown("### 📋 Complete Metrics Summary")

    summary_cols = st.columns(4)
    metrics_list = [
        ("QPS", f"{metrics.get('qps', 0):.2f}"),
        ("TPS", f"{metrics.get('tps', 0):.2f}"),
        ("TTFT", f"{metrics.get('ttft_ms', 0):.1f} ms"),
        ("P50", f"{metrics.get('latency_p50_ms', 0):.1f} ms"),
        ("P90", f"{metrics.get('latency_p90_ms', 0):.1f} ms"),
        ("P99", f"{metrics.get('latency_p99_ms', 0):.1f} ms"),
        ("Success", f"{success_rate * 100:.1f}%"),
        ("Error", f"{error_rate * 100:.1f}%"),
    ]

    for i, (name, value) in enumerate(metrics_list):
        col_idx = i % 4
        with summary_cols[col_idx]:
            st.metric(name, value)
