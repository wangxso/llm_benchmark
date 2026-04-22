"""Provider management for LLM APIs"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import os

# Provider config file path
PROVIDERS_FILE = Path(__file__).parent.parent.parent.parent / "providers.json"


@dataclass
class Provider:
    """Provider configuration"""
    name: str
    api_type: str  # openai, anthropic
    base_url: str
    api_key: str
    default_model: str = ""
    description: str = ""


def load_providers() -> List[Provider]:
    """Load providers from config file"""
    if not PROVIDERS_FILE.exists():
        return []

    try:
        with open(PROVIDERS_FILE, "r") as f:
            data = json.load(f)
            return [Provider(**p) for p in data.get("providers", [])]
    except:
        return []


def save_providers(providers: List[Provider]):
    """Save providers to config file"""
    data = {
        "providers": [asdict(p) for p in providers]
    }
    with open(PROVIDERS_FILE, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_provider(name: str) -> Optional[Provider]:
    """Get a provider by name"""
    providers = load_providers()
    for p in providers:
        if p.name == name:
            return p
    return None


def add_provider(provider: Provider) -> bool:
    """Add a new provider"""
    providers = load_providers()
    for p in providers:
        if p.name == provider.name:
            return False  # Already exists
    providers.append(provider)
    save_providers(providers)
    return True


def update_provider(provider: Provider) -> bool:
    """Update an existing provider"""
    providers = load_providers()
    for i, p in enumerate(providers):
        if p.name == provider.name:
            providers[i] = provider
            save_providers(providers)
            return True
    return False


def delete_provider(name: str) -> bool:
    """Delete a provider"""
    providers = load_providers()
    for i, p in enumerate(providers):
        if p.name == name:
            providers.pop(i)
            save_providers(providers)
            return True
    return False


def get_default_providers() -> List[Provider]:
    """Get list of default providers"""
    return [
        Provider(
            name="OpenAI",
            api_type="openai",
            base_url="https://api.openai.com/v1",
            api_key="",
            default_model="gpt-4",
            description="OpenAI GPT models"
        ),
        Provider(
            name="Anthropic",
            api_type="anthropic",
            base_url="https://api.anthropic.com",
            api_key="",
            default_model="claude-3-haiku-20240307",
            description="Anthropic Claude models"
        ),
        Provider(
            name="MiniMax",
            api_type="anthropic",
            base_url="https://api.minimaxi.com/anthropic",
            api_key="",
            default_model="MiniMax-Text-01",
            description="MiniMax models (Anthropic compatible)"
        ),
        Provider(
            name="Alibaba Qwen",
            api_type="openai",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key="",
            default_model="qwen-turbo",
            description="Alibaba Qwen models"
        ),
        Provider(
            name="DeepSeek",
            api_type="openai",
            base_url="https://api.deepseek.com/v1",
            api_key="",
            default_model="deepseek-chat",
            description="DeepSeek models"
        ),
        Provider(
            name="Local vLLM",
            api_type="openai",
            base_url="http://localhost:8000/v1",
            api_key="",
            default_model="",
            description="Local vLLM server"
        ),
    ]
