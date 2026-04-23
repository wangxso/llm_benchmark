"""Model capability check page"""

import streamlit as st
import asyncio
import aiohttp
import json
from typing import List, Dict
import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.webui.views.providers import load_providers, Provider
from src.webui.views.balancer import get_lb_as_provider


def render_check_page():
    st.header("🔍 Model Capability Check")
    st.markdown("Quick test to verify model is working and not degraded.")

    # Load providers
    providers = load_providers()

    # Add Load Balancer as a provider option
    lb_provider = get_lb_as_provider()
    if lb_provider:
        providers.insert(0, lb_provider)

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
            provider_names = [p.name for p in providers]
            selected_provider_name = st.selectbox(
                "Select Provider",
                provider_names,
                [p.name for p in providers],
                key="check_provider_select",
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
                key="check_model",
                help="Leave as default or enter a different model name"
            )

            timeout = st.slider("Timeout (seconds)", 10, 300, 60, key="check_timeout")

    # Test cases
    st.markdown("### Test Cases")
    st.caption("Tests basic math, knowledge, logic, coding, and Chinese capabilities.")

    if st.button("🚀 Run Check", type="primary", use_container_width=True):
        if not selected_provider:
            st.error("No provider selected")
            return

        # API key is optional for local vLLM servers
        if not selected_provider.api_key and "localhost" not in selected_provider.base_url and "127.0.0.1" not in selected_provider.base_url:
            st.warning(f"Provider '{selected_provider.name}' has no API key configured. This may cause authentication errors.")
            return

        # Run tests
        results = run_capability_check(
            api_type=selected_provider.api_type,
            model=model,
            api_base_url=selected_provider.base_url,
            api_key=selected_provider.api_key,
            timeout=timeout
        )

        # Display results
        st.markdown("---")
        display_check_results(results)


def run_capability_check(
    api_type: str,
    model: str,
    api_base_url: str,
    api_key: str,
    timeout: int
) -> List[Dict]:
    """Run capability check tests"""

    tests = [
        {
            "name": "Simple Math",
            "emoji": "🔢",
            "prompt": "What is 15 + 27? Answer with just the number.",
            "check": lambda r: "42" in r,
        },
        {
            "name": "Basic Knowledge",
            "emoji": "🌍",
            "prompt": "What is the capital of France? Answer with just the city name.",
            "check": lambda r: "paris" in r.lower(),
        },
        {
            "name": "Logic Reasoning",
            "emoji": "🧠",
            "prompt": "If all cats are mammals, and Tom is a cat, is Tom a mammal? Answer yes or no.",
            "check": lambda r: "yes" in r.lower(),
        },
        {
            "name": "Code Ability",
            "emoji": "💻",
            "prompt": "Write a Python function to calculate factorial. Just output the function code, no explanation.",
            "check": lambda r: "def " in r and "factorial" in r.lower(),
        },
        {
            "name": "Chinese",
            "emoji": "🀄",
            "prompt": "用中文回答：1+1等于几？只回答数字。",
            "check": lambda r: "2" in r or "二" in r,
        },
    ]

    # Prepare headers and URL
    base_url = api_base_url.rstrip("/")

    if api_type == "anthropic":
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
        if base_url.endswith("/anthropic"):
            url = f"{base_url}/v1/messages"
        else:
            url = f"{base_url}/messages" if not base_url.endswith("/messages") else base_url
    else:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        url = f"{base_url}/chat/completions"

    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, test in enumerate(tests):
        status_text.text(f"Testing: {test['name']}...")

        try:
            response_text = asyncio.run(send_request(
                url=url,
                headers=headers,
                prompt=test["prompt"],
                model=model,
                api_type=api_type,
                timeout=timeout
            ))

            passed = test["check"](response_text)
            results.append({
                "name": test["name"],
                "emoji": test["emoji"],
                "passed": passed,
                "response": response_text[:200] + "..." if len(response_text) > 200 else response_text,
            })
        except Exception as e:
            results.append({
                "name": test["name"],
                "emoji": test["emoji"],
                "passed": False,
                "error": str(e)[:100],
            })

        progress_bar.progress((i + 1) / len(tests))

    progress_bar.empty()
    status_text.empty()

    return results


async def send_request(
    url: str,
    headers: Dict,
    prompt: str,
    model: str,
    api_type: str,
    timeout: int
) -> str:
    """Send a single request to the API"""

    async with aiohttp.ClientSession() as session:
        if api_type == "anthropic":
            payload = {
                "model": model,
                "max_tokens": 500,
                "messages": [{"role": "user", "content": prompt}],
            }
        else:
            payload = {
                "model": model,
                "max_tokens": 500,
                "temperature": 0.0,
                "messages": [{"role": "user", "content": prompt}],
            }

        async with session.post(
            url,
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                try:
                    err_data = json.loads(text)
                    detail = err_data.get("error", {}).get("message", text[:100])
                except:
                    detail = text[:100]
                raise Exception(f"HTTP {resp.status}: {detail}")

            data = await resp.json()

            if api_type == "anthropic":
                content = data.get("content", [{}])[0].get("text", "") if isinstance(data.get("content"), list) else data.get("content", "")
            else:
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            return content


def display_check_results(results: List[Dict]):
    """Display check results"""
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    pass_rate = passed / total * 100 if total > 0 else 0

    # Summary metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Passed", f"{passed}/{total}")
    with col2:
        st.metric("Pass Rate", f"{pass_rate:.0f}%")
    with col3:
        if pass_rate >= 80:
            st.metric("Verdict", "✅ Normal", delta_color="normal")
        elif pass_rate >= 40:
            st.metric("Verdict", "⚠️ Partial", delta_color="off")
        else:
            st.metric("Verdict", "❌ Degraded", delta_color="inverse")

    # Detailed results
    st.markdown("### Test Results")

    for r in results:
        with st.container():
            col1, col2 = st.columns([1, 4])

            with col1:
                if r["passed"]:
                    st.success(f"✅ {r['emoji']} {r['name']}")
                else:
                    st.error(f"❌ {r['emoji']} {r['name']}")

            with col2:
                if r["passed"]:
                    st.text(r.get("response", "N/A"))
                else:
                    st.text(r.get("error", r.get("response", "Unknown")))

            st.divider()
