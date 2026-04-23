"""vLLM Load Balancer management page"""

import streamlit as st
import requests
import yaml
import time
from typing import Dict, Any, Optional
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def get_lb_config():
    """Get LB configuration"""
    return st.session_state.get("lb_config", {
        "api_url": "http://localhost:9000",
        "admin_token": "",
        "auto_refresh": True,
        "refresh_interval": 3,
    })


def set_lb_config(config: dict):
    """Set LB configuration"""
    st.session_state["lb_config"] = config


def api_request(endpoint: str, method: str = "GET", json_data: Any = None, config: dict = None) -> Optional[Dict]:
    """Make API request to load balancer"""
    if config is None:
        config = get_lb_config()

    headers = {"Content-Type": "application/json"}
    if config.get("admin_token"):
        headers["x-admin-token"] = config["admin_token"]

    url = f"{config['api_url'].rstrip('/')}{endpoint}"

    try:
        if method == "GET":
            resp = requests.get(url, headers=headers, timeout=10)
        elif method == "POST":
            resp = requests.post(url, headers=headers, json=json_data, timeout=10)
        elif method == "PUT":
            resp = requests.put(url, headers=headers, json=json_data, timeout=10)
        else:
            return None

        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 401:
            return {"error": "Invalid admin token"}
        else:
            return {"error": f"HTTP {resp.status_code}: {resp.text[:200]}"}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to load balancer"}
    except requests.exceptions.Timeout:
        return {"error": "Request timeout"}
    except Exception as e:
        return {"error": str(e)}


def render_balancer_page():
    st.header("⚖️ Load Balancer")
    st.markdown("Manage vLLM multi-instance load balancer.")

    # LB Connection Settings
    with st.expander("🔗 Connection Settings", expanded=False):
        config = get_lb_config()

        col1, col2 = st.columns([2, 1])
        with col1:
            api_url = st.text_input(
                "Balancer API URL",
                value=config.get("api_url", "http://localhost:9000"),
                key="lb_api_url_input"
            )
        with col2:
            admin_token = st.text_input(
                "Admin Token",
                value=config.get("admin_token", ""),
                type="password",
                key="lb_admin_token_input"
            )

        auto_refresh = st.checkbox("Auto Refresh", value=config.get("auto_refresh", True), key="lb_auto_refresh")
        refresh_interval = st.slider("Refresh Interval (s)", 1, 10, config.get("refresh_interval", 3), key="lb_refresh_interval")

        if st.button("Save Settings"):
            set_lb_config({
                "api_url": api_url,
                "admin_token": admin_token,
                "auto_refresh": auto_refresh,
                "refresh_interval": refresh_interval,
            })
            st.success("Settings saved!")

    config = get_lb_config()

    # Test connection and get state
    state = api_request("/admin/state", config=config)

    if "error" in state:
        st.error(f"❌ {state['error']}")
        st.caption("Make sure the load balancer is running: `python -m lb.cli serve --config config.yaml`")
        return

    # Summary metrics
    st.markdown("### 📊 Summary")

    instances = state.get("instances", [])
    healthy_count = sum(1 for i in instances if i.get("healthy"))
    running_count = sum(1 for i in instances if i.get("running"))
    total_inflight = sum(i.get("inflight_requests", 0) for i in instances)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Instances", f"{healthy_count}/{len(instances)}", "healthy")
    with col2:
        st.metric("Running", f"{running_count}/{len(instances)}")
    with col3:
        st.metric("Inflight", total_inflight)
    with col4:
        monitor = state.get("monitor", {})
        qps = monitor.get("qps", 0)
        st.metric("QPS", f"{qps:.1f}")

    # Instance table
    st.markdown("### 🖥️ Instances")

    if not instances:
        st.info("No instances configured")
    else:
        for inst in instances:
            cfg = inst.get("config", {})
            key_metrics = inst.get("key_metrics", {})

            with st.container():
                col1, col2, col3, col4, col5 = st.columns([2, 1, 2, 2, 1])

                with col1:
                    status_emoji = "🟢" if inst.get("healthy") else ("🟡" if inst.get("running") else "🔴")
                    st.markdown(f"**{status_emoji} {cfg.get('id', 'unknown')}**")
                    st.caption(f"Model: `{cfg.get('model', 'N/A')}`")

                with col2:
                    st.metric("PID", inst.get("pid") or "N/A")
                    st.metric("Inflight", inst.get("inflight_requests", 0))

                with col3:
                    kv_cache = key_metrics.get("kv_cache_usage", 0) * 100
                    gpu_util = key_metrics.get("gpu_utilization", 0) * 100
                    st.metric("KV Cache", f"{kv_cache:.1f}%")
                    st.metric("GPU", f"{gpu_util:.1f}%")

                with col4:
                    running = int(key_metrics.get("running_requests", 0))
                    waiting = int(key_metrics.get("waiting_requests", 0))
                    st.metric("Running", running)
                    st.metric("Waiting", waiting)

                with col5:
                    if inst.get("running"):
                        color = "secondary"
                        label = "Stop"
                        action = "stop"
                    else:
                        color = "primary"
                        label = "Start"
                        action = "start"

                    if st.button(label, key=f"btn_{cfg.get('id')}_{action}", type=color):
                        endpoint = f"/admin/instances/{cfg.get('id')}/{action}"
                        result = api_request(endpoint, method="POST", config=config)
                        if "error" in result:
                            st.error(result["error"])
                        else:
                            st.success(f"Instance {action}ed")
                            time.sleep(1)
                            st.rerun()

                st.divider()

    # Config editor
    st.markdown("### ⚙️ Configuration")

    config_resp = api_request("/admin/config", config=config)
    if "error" not in config_resp:
        config_text = config_resp.get("text", "")
        config_path = config_resp.get("path", "")

        if config_path:
            st.caption(f"Config file: `{config_path}`")

        new_config = st.text_area("YAML Config", value=config_text, height=300, key="lb_config_editor")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save & Apply", type="primary"):
                result = api_request("/admin/config", method="PUT", json_data={"text": new_config}, config=config)
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success("Config saved and applied!")
                    time.sleep(1)
                    st.rerun()

        with col2:
            if st.button("🔄 Reload from Disk"):
                result = api_request("/admin/reload", method="POST", config=config)
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success("Config reloaded!")
                    time.sleep(1)
                    st.rerun()

    # Auto refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()
