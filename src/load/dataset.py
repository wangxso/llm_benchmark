import json
import random
import csv
from typing import List, Dict, Any
from pathlib import Path


TOPICS = [
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "quantum computing",
    "blockchain",
    "cybersecurity",
    "cloud computing",
    "internet of things",
    "5G networks",
    "renewable energy",
    "biotechnology",
    "space exploration",
    "climate change",
    "oceanography",
    "neuroscience",
    "philosophy",
    "psychology",
    "economics",
    "history",
    "literature",
]


SHORT_TEMPLATES = [
    "What is {}?",
    "Explain {} in detail.",
    "How to {}?",
    "What are the benefits of {}?",
    "Tell me about {}.",
    "What is the history of {}?",
    "How does {} work?",
    "What are the advantages of {}?",
    "Why is {} important?",
    "Can you summarize {}?",
]


LONG_TEMPLATES = [
    "Write a comprehensive report about {} including history, current state, and future prospects.",
    "Explain the technical details of {} and its impact on society.",
    "Provide a detailed analysis of {} with examples and case studies.",
    "Describe the development timeline of {} and its significance.",
]


MULTI_TURN_TEMPLATE = [
    {"role": "user", "content": "What is {}?"},
    {
        "role": "assistant",
        "content": "{} is an important concept in modern technology...",
    },
    {"role": "user", "content": "Can you provide more details?"},
]


class Prompt:
    """Prompt data structure"""

    def __init__(self, text: str, prompt_type: str = "short", max_tokens: int = 1024):
        self.text = text
        self.type = prompt_type
        self.max_tokens = max_tokens

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.text,
            "type": self.type,
            "max_tokens": self.max_tokens,
        }


class DatasetManager:
    """Dataset manager - supports import and generate modes"""

    def __init__(self, config: Dict):
        self.mode = config.get("dataset", {}).get("mode", "generate")
        self.import_config = config.get("dataset", {}).get("import", {})
        self.generate_config = config.get("dataset", {}).get("generate", {})

        self.short_ratio = self.generate_config.get("short_ratio", 0.7)
        self.long_ratio = self.generate_config.get("long_ratio", 0.3)
        self.max_input_len = self.generate_config.get("max_input_len", 4096)
        self.max_output_len = self.generate_config.get("max_output_len", 2048)
        self.text_field = self.import_config.get("text_field", "prompt")
        self.type_field = self.import_config.get("type_field", "type")
        self.max_tokens_field = self.import_config.get("max_tokens_field", "max_tokens")

        self._prompts = []
        self._load_or_generate()

    def _load_or_generate(self):
        """Load prompts based on mode"""
        if self.mode in ("import", "mixed"):
            import_path = self.import_config.get("path")
            if import_path and Path(import_path).exists():
                self._prompts = self.load_imported(import_path)

                if self.mode == "mixed":
                    needed = 1000 - len(self._prompts)
                    if needed > 0:
                        self._prompts.extend(self.generate_synthetic(needed))
            else:
                self._prompts = self.generate_synthetic(1000)
        else:
            self._prompts = self.generate_synthetic(1000)

    def load_imported(self, path: str) -> List[Prompt]:
        """Load prompts from JSON/JSONL/CSV file"""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        prompts = []

        if path.suffix == ".json":
            with open(path, "r") as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError("JSON dataset must be a list of objects")

            for item in data:
                prompt = self._build_prompt(item)
                if prompt:
                    prompts.append(prompt)

        elif path.suffix == ".jsonl":
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    prompt = self._build_prompt(data)
                    if prompt:
                        prompts.append(prompt)

        elif path.suffix == ".csv":
            with open(path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    prompt = self._build_prompt(row)
                    if prompt:
                        prompts.append(prompt)

        else:
            raise ValueError(f"Unsupported format: {path.suffix}")

        return prompts

    def _build_prompt(self, data: Dict[str, Any]) -> Prompt | None:
        text = data.get(self.text_field)
        if text is None:
            text = data.get("prompt", data.get("text", data.get("instruction", "")))

        if not text:
            return None

        prompt_type = data.get(self.type_field, data.get("type", "short"))
        max_tokens = data.get(
            self.max_tokens_field, data.get("max_tokens", self.max_output_len)
        )

        return Prompt(str(text), str(prompt_type), int(max_tokens))

    def generate_synthetic(self, count: int) -> List[Prompt]:
        """Generate synthetic prompts"""
        prompts = []

        for _ in range(count):
            if random.random() < self.short_ratio:
                prompts.append(self._generate_short())
            else:
                prompts.append(self._generate_long())

        return prompts

    def _generate_short(self) -> Prompt:
        """Generate short Q&A prompt"""
        template = random.choice(SHORT_TEMPLATES)
        topic = random.choice(TOPICS)
        text = template.format(topic)
        return Prompt(text, "short", 256)

    def _generate_long(self) -> Prompt:
        """Generate long context prompt"""
        template = random.choice(LONG_TEMPLATES)
        topic = random.choice(TOPICS)
        text = template.format(topic)

        if random.random() < 0.3:
            text = self._add_repetition(text)

        return Prompt(text, "long", self.max_output_len)

    def _add_repetition(self, text: str) -> str:
        """Add context repetition to simulate longer input"""
        parts = text.split(" about ")
        if len(parts) > 1:
            topic = parts[1].split(" ")[0]
            text += f" Focus on {topic}, {topic}, and {topic}."
        return text

    def get_prompt(self) -> Prompt:
        """Get a random prompt"""
        return random.choice(self._prompts)

    def get_batch(self, size: int) -> List[Prompt]:
        """Get a batch of random prompts"""
        return random.choices(self._prompts, k=size)

    def get_all(self) -> List[Prompt]:
        """Get all prompts"""
        return self._prompts

    def __len__(self):
        return len(self._prompts)
