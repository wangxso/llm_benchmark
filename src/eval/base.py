"""Base class for benchmark datasets"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datasets import load_dataset
import os


# ModelScope to HuggingFace dataset mapping
MODELSCOPE_MAPPING = {
    "mmlu-pro": "iic/MMLU-Pro",
    "gpqa": "iic/GPQA",
    "ceval": "iic/CEval",
    "mmlu": "iic/MMLU",
}


class BaseBenchmark(ABC):
    """Abstract base class for benchmark datasets"""

    name: str = ""
    hf_path: str = ""
    hf_name: Optional[str] = None
    description: str = ""
    requires_auth: bool = False

    # ModelScope support
    modelscope_path: Optional[str] = None  # Override for ModelScope path

    def load(
        self,
        split: str = "test",
        subject: Optional[str] = None,
        max_samples: Optional[int] = None,
        token: Optional[str] = None,
        offline: bool = False,
        source: str = "huggingface",  # "huggingface" or "modelscope"
    ) -> List[Dict[str, Any]]:
        """Load dataset from HuggingFace or ModelScope

        Args:
            split: Dataset split to use
            subject: Optional subject filter
            max_samples: Maximum samples to load
            token: HuggingFace token for gated datasets
            offline: Use cached dataset only (no network)
            source: Dataset source - "huggingface" or "modelscope"

        Returns:
            List of items with unified format:
            - question: str
            - choices: List[str] (A, B, C, D options)
            - answer: str (correct option letter: A/B/C/D)
            - subject: Optional[str]
        """
        # Get token from parameter, environment, or huggingface_hub
        if token is None:
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            if token is None:
                try:
                    from huggingface_hub import HfFolder
                    token = HfFolder.get_token()
                except Exception:
                    pass

        # Determine dataset path based on source
        if source == "modelscope":
            dataset_path = self.modelscope_path or MODELSCOPE_MAPPING.get(self.name.lower().replace("-", "").replace("_", "").lower())
            if not dataset_path:
                # Try auto-converting: TIGER-Lab/MMLU-Pro -> iic/MMLU-Pro
                parts = self.hf_path.split("/")
                if len(parts) == 2:
                    dataset_path = f"iic/{parts[1]}"
                else:
                    dataset_path = self.hf_path
        else:
            dataset_path = self.hf_path

        try:
            load_kwargs = {
                "path": dataset_path,
                "split": split,
                "trust_remote_code": True,
            }
            if self.hf_name:
                load_kwargs["name"] = self.hf_name
            if token and source == "huggingface":
                load_kwargs["token"] = token
            if offline:
                load_kwargs["download_mode"] = "force_redownload" if os.environ.get("FORCE_REDOWNLOAD") else "reuse_cache_if_exists"

            ds = load_dataset(**load_kwargs)
        except Exception as e:
            error_msg = str(e)
            if source == "modelscope":
                # Fallback to HuggingFace if ModelScope fails
                print(f"ModelScope load failed, trying HuggingFace: {e}")
                return self.load(split=split, subject=subject, max_samples=max_samples,
                                token=token, offline=offline, source="huggingface")

            if "gated dataset" in error_msg.lower() or "authenticated" in error_msg.lower():
                raise RuntimeError(
                    f"Dataset '{self.hf_path}' requires authentication.\n"
                    f"Please set HF_TOKEN environment variable or run: huggingface-cli login"
                )
            raise RuntimeError(f"Failed to load {self.name}: {e}")

        items = []
        for row in ds:
            item = self._parse_row(row)
            if item:
                if subject is None or item.get("subject") == subject:
                    items.append(item)

        if max_samples and len(items) > max_samples:
            items = items[:max_samples]

        return items

    @abstractmethod
    def _parse_row(self, row: Dict) -> Optional[Dict[str, Any]]:
        """Parse a dataset row into unified format"""
        pass

    def get_subjects(self) -> List[str]:
        """Get list of available subjects"""
        return []
