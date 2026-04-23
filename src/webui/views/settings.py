"""Settings page with provider configuration"""

import streamlit as st
from pathlib import Path
import os
from typing import Optional

from src.webui.views.providers import (
    Provider, load_providers, save_providers, add_provider,
    update_provider, delete_provider, get_default_providers
)


def render_settings_page():
    st.header("⚙️ Settings")
    st.markdown("Configure providers and preferences.")

    # Provider Management
    render_providers_section()

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

            if st.button("Clear Cache", key="clear_cache_btn"):
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


def render_providers_section():
    """Render provider management section"""
    with st.expander("🔑 API Providers", expanded=True):
        st.markdown("Configure your API providers for quick selection in evaluations and load tests.")

        providers = load_providers()

        # Show existing providers
        if providers:
            st.markdown("#### Configured Providers")

            for provider in providers:
                col1, col2, col3, col4 = st.columns([2, 1, 2, 1])

                with col1:
                    st.markdown(f"**{provider.name}**")
                    if provider.description:
                        st.caption(provider.description)

                with col2:
                    st.caption(f"Type: `{provider.api_type}`")

                with col3:
                    key_masked = provider.api_key[:8] + "..." if provider.api_key else "Not set"
                    st.caption(f"Key: {key_masked}")
                    st.caption(f"Model: {provider.default_model or 'Auto'}")

                with col4:
                    # Edit and Delete buttons
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("✏️", key=f"edit_{provider.name}", help="Edit provider"):
                            st.session_state['editing_provider'] = provider.name
                    with c2:
                        if st.button("🗑️", key=f"delete_{provider.name}", help="Delete provider"):
                            delete_provider(provider.name)
                            st.rerun()

            st.markdown("---")

        # Edit or Add provider form
        editing_name = st.session_state.get('editing_provider')

        if editing_name:
            st.markdown(f"#### Edit Provider: {editing_name}")
            provider = next((p for p in providers if p.name == editing_name), None)
            if provider:
                render_provider_form(provider, is_edit=True)
        else:
            st.markdown("#### Add New Provider")
            render_provider_form(None, is_edit=False)

        # Quick add default providers
        if not providers:
            st.markdown("---")
            st.markdown("#### Quick Start")
            if st.button("Add Default Providers", type="secondary"):
                default_providers = get_default_providers()
                for p in default_providers:
                    add_provider(p)
                st.success(f"Added {len(default_providers)} default providers!")
                st.rerun()


def render_provider_form(existing: Optional[Provider], is_edit: bool):
    """Render the provider form"""
    col1, col2 = st.columns(2)

    with col1:
        if is_edit and existing:
            name = existing.name
            st.text_input("Provider Name", value=name, disabled=True, key="provider_name_display")
            name_hidden = st.text_input("Name", value=name, key="provider_name_hidden")
        else:
            name = st.text_input("Provider Name", placeholder="e.g., OpenAI, MiniMax", key="provider_name")
            name_hidden = name

        api_type = st.selectbox(
            "API Type",
            ["openai", "anthropic"],
            index=0 if not existing or existing.api_type == "openai" else 1,
            key="provider_api_type",
            help="OpenAI: for OpenAI-compatible APIs. Anthropic: for Claude-style APIs."
        )

        base_url = st.text_input(
            "Base URL",
            value=existing.base_url if existing else "",
            placeholder="e.g., https://api.openai.com/v1",
            key="provider_base_url"
        )

    with col2:
        api_key = st.text_input(
            "API Key",
            value=existing.api_key if existing else "",
            type="password",
            key="provider_api_key"
        )

        default_model = st.text_input(
            "Default Model",
            value=existing.default_model if existing else "",
            placeholder="e.g., gpt-4, claude-3-haiku-20240307",
            key="provider_default_model"
        )

        description = st.text_input(
            "Description",
            value=existing.description if existing else "",
            placeholder="Brief description",
            key="provider_description"
        )

    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        save_label = "Update Provider" if is_edit else "Add Provider"
        if st.button(save_label, type="primary", key="save_provider_btn"):
            if not name_hidden:
                st.error("Provider name is required")
            elif not base_url:
                st.error("Base URL is required")
            else:
                new_provider = Provider(
                    name=name_hidden,
                    api_type=api_type,
                    base_url=base_url,
                    api_key=api_key,
                    default_model=default_model,
                    description=description
                )

                if is_edit:
                    if update_provider(new_provider):
                        st.success(f"Provider '{name_hidden}' updated!")
                        st.session_state['editing_provider'] = None
                        st.rerun()
                    else:
                        st.error("Failed to update provider")
                else:
                    if add_provider(new_provider):
                        st.success(f"Provider '{name_hidden}' added!")
                        st.rerun()
                    else:
                        st.error("Provider with this name already exists")

    with col2:
        if is_edit:
            if st.button("Cancel", key="cancel_edit_btn"):
                st.session_state['editing_provider'] = None
                st.rerun()
