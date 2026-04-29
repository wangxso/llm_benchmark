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
    ["Model Check", "Evaluation", "Load Testing", "Auto-Tuning", "Load Balancer", "GPU Monitor", "Results", "Settings"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")

# Navigation
if page == "Model Check":
    from src.webui.views.check import render_check_page
    render_check_page()
elif page == "Evaluation":
    from src.webui.views.eval import render_eval_page
    render_eval_page()
elif page == "Load Testing":
    from src.webui.views.lb import render_lb_page
    render_lb_page()
elif page == "Auto-Tuning":
    from src.webui.views.autotune import render_autotune_page
    render_autotune_page()
elif page == "Load Balancer":
    from src.webui.views.balancer import render_balancer_page
    render_balancer_page()
elif page == "GPU Monitor":
    from src.webui.views.gpu import render_gpu_page
    render_gpu_page()
elif page == "Results":
    from src.webui.views.results import render_results_page
    render_results_page()
elif page == "Settings":
    from src.webui.views.settings import render_settings_page
    render_settings_page()
