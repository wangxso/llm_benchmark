import asyncio
import random
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from .dataset import DatasetManager, Prompt


class LoadType(Enum):
    FIXED = "fixed"
    STEP = "step"
    BURST = "burst"
    STREAMING = "streaming"
    LONG_CONTEXT = "long_context"


@dataclass
class LoadScenario:
    """Load test scenario"""

    scenario_type: LoadType = LoadType.FIXED
    model: str = ""
    concurrency: int = 100
    duration: int = 300
    warmup_duration: int = 10

    step_increment: int = 100
    step_duration: int = 60
    max_concurrency: int = 1000

    peak_concurrency: int = 2000

    stream: bool = False
    max_input_len: int = 4096
    max_output_len: int = 2048

    dataset: Optional[DatasetManager] = None


class LoadGenerator:
    """Load generator for creating test requests"""

    def __init__(self, config: Dict):
        self.config = config
        self.dataset = DatasetManager(config)

        self.stream = config.get("request", {}).get("stream", False)
        self.max_tokens = config.get("request", {}).get("max_tokens", 1024)
        self.temperature = config.get("request", {}).get("temperature", 0.7)

    def create_scenario(self) -> LoadScenario:
        """Create load scenario from config"""
        load_config = self.config.get("load", {})
        dataset_config = self.config.get("dataset", {})

        scenario = LoadScenario(
            scenario_type=LoadType(load_config.get("type", "fixed")),
            concurrency=load_config.get("base_concurrency", 100),
            duration=load_config.get("duration", 300),
            warmup_duration=load_config.get("warmup_duration", 10),
            step_increment=load_config.get("step_increment", 100),
            step_duration=load_config.get("step_duration", 60),
            max_concurrency=load_config.get("max_concurrency", 1000),
            peak_concurrency=load_config.get("peak_concurrency", 2000),
            stream=self.stream,
            max_input_len=dataset_config.get("generate", {}).get("max_input_len", 4096),
            max_output_len=dataset_config.get("generate", {}).get(
                "max_output_len", 2048
            ),
            dataset=self.dataset,
        )

        return scenario

    def generate_request(self, prompt: Prompt = None) -> Dict[str, Any]:
        """Generate a single request payload"""
        if prompt is None:
            prompt = self.dataset.get_prompt()

        request = {
            "model": self.config.get("vllm", {}).get(
                "model", "Qwen/Qwen2.5-7B-Instruct"
            ),
            "messages": [{"role": "user", "content": prompt.text}],
            "temperature": self.temperature,
            "max_tokens": prompt.max_tokens or self.max_tokens,
        }

        if self.stream or prompt.type == "streaming":
            request["stream"] = True

        return request

    def generate_batch(self, size: int) -> List[Dict[str, Any]]:
        """Generate a batch of requests"""
        prompts = self.dataset.get_batch(size)
        return [self.generate_request(p) for p in prompts]

    def create_multi_model_scenario(
        self, models: List[str], mix_ratios: List[int] = None
    ):
        """Create multi-model load scenario"""
        if mix_ratios is None:
            mix_ratios = [1] * len(models)

        total = sum(mix_ratios)
        ratios = [r / total for r in mix_ratios]

        scenarios = []
        for model, ratio in zip(models, ratios):
            scenario = self.create_scenario()
            scenario.model = model
            scenario.concurrency = int(scenario.concurrency * ratio)
            scenarios.append(scenario)

        return scenarios
