"""Settings page"""

import streamlit as st
from pathlib import Path
import os


def render_settings_page():
    st.header("⚙️ Settings")
    st.markdown("Configure global settings and preferences.")

    # API Defaults
    with st.expander("🔑 API Defaults", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.text_input(
                "Default OpenAI Base URL",
                value="https://api.openai.com/v1",
                key="settings_openai_url"
            )
            st.text_input(
                "Default Anthropic Base URL",
                value="https://api.anthropic.com",
                key="settings_anthropic_url"
            )

        with col2:
            default_timeout = st.slider(
                "Default Timeout (seconds)",
                10, 300, 60,
                key="settings_default_timeout"
            )
            default_concurrency = st.slider(
                "Default Concurrency",
                1, 32, 8,
                key="settings_default_concurrency"
            )

    # HuggingFace
    with st.expander("🤗 HuggingFace"):
        hf_token = os.environ.get("HF_TOKEN", "")
        st.text_input(
            "HF Token",
            value=hf_token,
            type="password",
            key="settings_hf_token",
            help="Set HF_TOKEN environment variable for persistent configuration"
        )
        st.caption("Used for accessing gated datasets like GPQA")

    # Cache Settings
    with st.expander("💾 Cache"):
        cache_dir = Path.home() / ".cache" / "huggingface" / "datasets"

        if cache_dir.exists():
            # Calculate cache size
            total_size = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
            size_mb = total_size / (1024 * 1024)

            st.metric("Dataset Cache Size", f"{size_mb:.1f} MB")
            st.caption(f"Location: {cache_dir}")

            if st.button("Clear Cache"):
                import shutil
                try:
                    shutil.rmtree(cache_dir)
                    st.success("Cache cleared!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to clear cache: {e}")
        else:
            st.info("No cached datasets found")

    # Results
    with st.expander("📊 Results"):
        default_output = st.text_input(
            "Default Output Directory",
            value="./results",
            key="settings_output_dir"
        )

        auto_save = st.checkbox(
            "Auto-save results",
            value=True,
            key="settings_auto_save"
        )

    # About
    with st.expander("ℹ️ About"):
        st.markdown("""
        **LLM Benchmark** v0.1.0

        A comprehensive benchmarking platform for LLMs including:
        - 📊 **Evaluation**: GPQA, MMLU-Pro, C-Eval, etc.
        - ⚡ **Load Testing**: Fixed, Step, Burst scenarios
        - 🔍 **Model Check**: Quick capability verification

        [GitHub Repository](https://github.com/wangxso/llm_benchmark)
        """)
