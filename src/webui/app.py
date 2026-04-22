"""Streamlit WebUI for LLM Benchmark"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

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

# Navigation with absolute imports
if page == "Model Check":
    from src.webui.pages.check import render_check_page
    render_check_page()
elif page == "Evaluation":
    from src.webui.pages.eval import render_eval_page
    render_eval_page()
elif page == "Load Testing":
    from src.webui.pages.lb import render_lb_page
    render_lb_page()
elif page == "Results":
    from src.webui.pages.results import render_results_page
    render_results_page()
elif page == "Settings":
    from src.webui.pages.settings import render_settings_page
    render_settings_page()
