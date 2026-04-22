"""Streamlit WebUI for LLM Benchmark"""

import streamlit as st

st.set_page_config(
    page_title="LLM Benchmark",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("🚀 LLM Benchmark")
st.sidebar.caption("High-Concurrency Simulation Testing Platform")

page = st.sidebar.radio(
    "Navigation",
    ["Model Check", "Evaluation", "Load Testing", "Results", "Settings"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")

# Navigation
if page == "Model Check":
    from .pages.check import render_check_page
    render_check_page()
elif page == "Evaluation":
    from .pages.eval import render_eval_page
    render_eval_page()
elif page == "Load Testing":
    from .pages.lb import render_lb_page
    render_lb_page()
elif page == "Results":
    from .pages.results import render_results_page
    render_results_page()
elif page == "Settings":
    from .pages.settings import render_settings_page
    render_settings_page()
