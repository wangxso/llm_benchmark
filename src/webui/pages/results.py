"""Results browsing page"""

import streamlit as st
import json
import glob
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd


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

    # Filter by type
    col1, col2 = st.columns([1, 3])

    with col1:
        result_type = st.selectbox(
            "Result Type",
            ["All", "Evaluation", "Load Test"],
            key="results_type_filter"
        )

    with col2:
        # Search/filter
        search = st.text_input("Search", placeholder="Filter by model, benchmark...", key="results_search")

    # Filter results
    filtered_files = filter_results(result_files, result_type, search)

    st.markdown(f"**Found {len(filtered_files)} results**")

    # Display results
    for file_info in filtered_files[:20]:  # Limit to 20 results
        with st.container():
            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                st.markdown(f"**{file_info['name']}**")
                st.caption(f"📅 {file_info['timestamp']}")

            with col2:
                st.text(file_info['type'])
                if file_info.get('model'):
                    st.caption(f"Model: {file_info['model']}")

            with col3:
                if st.button("View", key=f"view_{file_info['path']}"):
                    show_result_detail(file_info['path'])

            st.divider()


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
    """Show evaluation result details"""
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Accuracy", f"{data.get('overall_accuracy', 0) * 100:.2f}%")
        st.metric("Model", data.get("model", "unknown"))

    with col2:
        correct = data.get("correct", 0)
        total = data.get("total_questions", 0)
        st.metric("Correct", f"{correct}/{total}")
        st.metric("Failed", data.get("failed_count", 0))

    # Errors
    if data.get("failed_count", 0) > 0:
        st.markdown("#### Errors")
        error_types = data.get("error_types", {})
        for err_type, count in error_types.items():
            st.write(f"- **{err_type}**: {count}")

    # Subjects
    subjects = data.get("subjects", {})
    if subjects:
        st.markdown("#### By Subject")
        # Use markdown table instead of dataframe
        table_md = "| Subject | Accuracy | Correct |\n|---------|----------|--------|\n"
        for subject, stats in subjects.items():
            table_md += f"| {subject} | {stats['accuracy'] * 100:.1f}% | {stats['correct']}/{stats['total']} |\n"
        st.markdown(table_md)


def show_load_test_detail(data: Dict):
    """Show load test result details"""
    metrics = data.get("metrics", {})

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("QPS", f"{metrics.get('qps', 0):.2f}")
    with col2:
        st.metric("TPS", f"{metrics.get('tps', 0):.2f}")
    with col3:
        st.metric("P99", f"{metrics.get('latency_p99_ms', 0):.1f}ms")
    with col4:
        st.metric("Success", f"{metrics.get('success_rate', 0) * 100:.1f}%")
