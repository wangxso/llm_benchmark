"""Base class for benchmark datasets"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datasets import load_dataset, Dataset
import os
import tempfile


# ModelScope to HuggingFace dataset mapping
MODELSCOPE_MAPPING = {
    # Keys must match: self.name.lower().replace("-", "").replace("_", "")
    "mmlupro": "modelscope/MMLU-Pro",
    "mmluredux": "AI-ModelScope/mmlu-redux",
    "gpqa": "modelscope/gpqa",
    "supergpqa": "m-a-p/SuperGPQA",
    "ceval": "evalscope/ceval",
    "mmlu": "modelscope/MMLU",
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
            if source == "modelscope":
                ds = self._load_from_modelscope(dataset_path, split)
            else:
                load_kwargs = {
                    "path": dataset_path,
                    "split": split,
                }
                if self.hf_name:
                    load_kwargs["name"] = self.hf_name
                if token:
                    load_kwargs["token"] = token
                if offline:
                    load_kwargs["download_mode"] = "force_redownload" if os.environ.get("FORCE_REDOWNLOAD") else "reuse_cache_if_exists"

                ds = load_dataset(**load_kwargs)
        except Exception as e:
            error_msg = str(e)

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

    def _load_from_modelscope(self, dataset_path: str, split: str):
        """Load dataset using ModelScope SDK (bypasses HuggingFace connectivity)"""
        try:
            from modelscope.hub.snapshot_download import snapshot_download
        except ImportError:
            raise RuntimeError(
                "ModelScope SDK not installed. Install it with: pip install modelscope"
            )

        cache_dir = os.path.join(tempfile.gettempdir(), "modelscope_datasets", dataset_path.replace("/", "_"))
        print(f"[ModelScope] Downloading {dataset_path} to {cache_dir}...")

        local_dir = snapshot_download(
            repo_id=dataset_path,
            repo_type="dataset",
            cache_dir=cache_dir,
        )

        # Bypass dataset_infos.json which may have incompatible feature definitions.
        # Use pandas to read data files directly, then convert to Dataset.
        import pandas as pd

        # File extensions that contain actual data (not metadata like dataset_info.json)
        _DATA_EXTS = (".csv", ".parquet", ".jsonl")

        def _find_data_files(base_dir, pattern=None):
            """Find data files in base_dir, optionally filtered by filename pattern."""
            files = []
            for f in sorted(os.listdir(base_dir)):
                fp = os.path.join(base_dir, f)
                if os.path.isfile(fp) and f.endswith(_DATA_EXTS):
                    if pattern is None or f.startswith(pattern):
                        files.append(fp)
            return files

        def _load_files(files):
            """Load data files with pandas and return a Dataset."""
            if not files:
                return None
            ext = os.path.splitext(files[0])[1]
            if ext == ".parquet":
                dfs = [pd.read_parquet(f) for f in files]
            elif ext == ".jsonl":
                dfs = [pd.read_json(f, lines=True) for f in files]
            elif ext == ".csv":
                dfs = [pd.read_csv(f, dtype=str) for f in files]
            else:
                return None
            df = pd.concat(dfs, ignore_index=True)
            return Dataset.from_pandas(df)

        # 1) Try split-specific files in data/ or root
        data_dir = os.path.join(local_dir, "data")
        if not os.path.isdir(data_dir):
            data_dir = local_dir
        result = _load_files(_find_data_files(data_dir, pattern=split))
        if result is not None:
            return result

        # 2) Try any data files in data/ or root
        result = _load_files(_find_data_files(data_dir))
        if result is not None:
            return result

        # 3) Recursive search: look in subdirectories (e.g. MMLU-Redux subject dirs)
        # If split matches a subdirectory name, load from that directory only.
        # Otherwise, load all matching data files from all subdirectories.
        all_files = []
        for entry in sorted(os.listdir(local_dir)):
            sub = os.path.join(local_dir, entry)
            if os.path.isdir(sub) and not entry.startswith("."):
                if entry == split:
                    all_files.extend(_find_data_files(sub))
                    break
                else:
                    all_files.extend(_find_data_files(sub, pattern=split))
        if all_files:
            result = _load_files(all_files)
            if result is not None:
                return result

        # 4) Last resort: let datasets library figure it out
        load_kwargs = {"path": local_dir, "split": split}
        if self.hf_name:
            load_kwargs["name"] = self.hf_name
        return load_dataset(**load_kwargs)

    @abstractmethod
    def _parse_row(self, row: Dict) -> Optional[Dict[str, Any]]:
        """Parse a dataset row into unified format"""
        pass

    def get_subjects(self) -> List[str]:
        """Get list of available subjects"""
        return []
