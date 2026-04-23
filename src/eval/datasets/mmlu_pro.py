"""MMLU-Pro benchmark loader

MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark
https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
"""

from typing import Dict, Any, Optional, List
from ..base import BaseBenchmark
from ..registry import register_benchmark


MMLU_PRO_CATEGORIES = {
    # STEM
    "math": "STEM",
    "physics": "STEM",
    "chemistry": "STEM",
    "biology": "STEM",
    "computer science": "STEM",
    "engineering": "STEM",
    # Social Science
    "economics": "Social Science",
    "psychology": "Social Science",
    "business": "Social Science",
    # Humanities
    "philosophy": "Humanities",
    "history": "Humanities",
    "law": "Humanities",
    # Other
    "health": "Other",
    "other": "Other",
}


@register_benchmark("mmlu-pro")
class MMLUProBenchmark(BaseBenchmark):
    name = "MMLU-Pro"
    hf_path = "TIGER-Lab/MMLU-Pro"
    description = "Enhanced MMLU with more challenging questions and 10 options"

    def get_category_map(self) -> Dict[str, str]:
        """Return subject to category mapping"""
        return MMLU_PRO_CATEGORIES

    def load(
        self,
        split: str = "test",
        subject: Optional[str] = None,
        max_samples: Optional[int] = None,
        token: Optional[str] = None,
        offline: bool = False,
    ) -> List[Dict[str, Any]]:
        """Load MMLU-Pro dataset"""
        return super().load(split=split, subject=subject, max_samples=max_samples, token=token, offline=offline)

    def _parse_row(self, row: Dict) -> Optional[Dict[str, Any]]:
        """Parse MMLU-Pro row format

        MMLU-Pro fields:
        - question_id: unique id
        - question: the question text
        - options: list of 10 options (A-J), some may be "N/A"
        - answer: correct option letter (A-J)
        - answer_index: correct option index (0-9)
        - cot_content: chain-of-thought content
        - category: subject area
        - src: source
        """
        question = row.get("question", "")
        raw_options = row.get("options", [])
        answer = row.get("answer", "")

        if not question or not raw_options:
            return None

        # Filter out "N/A" options (official evaluator behavior)
        options = [opt for opt in raw_options if opt != "N/A"]

        if not options:
            return None

        # Recalculate answer index after filtering N/A
        # answer_index points to position in original options list
        answer_index = row.get("answer_index", 0)
        answer_letter = answer.upper() if isinstance(answer, str) else chr(ord("A") + answer_index)

        # Find the correct answer in filtered options
        # The answer letter should still be valid (A-J) based on filtered options
        return {
            "question": question,
            "choices": options,
            "answer": answer_letter,
            "subject": row.get("category", "unknown"),
        }

    def get_subjects(self) -> List[str]:
        return [
            "math",
            "physics",
            "chemistry",
            "biology",
            "computer science",
            "engineering",
            "health",
            "economics",
            "psychology",
            "business",
            "philosophy",
            "history",
            "law",
            "other",
        ]
