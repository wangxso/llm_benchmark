"""SuperGPQA benchmark loader

SuperGPQA: A comprehensive benchmark for large language models
https://huggingface.co/datasets/m-a-p/SuperGPQA
"""

from typing import Dict, Any, Optional, List
from ..base import BaseBenchmark
from ..registry import register_benchmark


SUPER_GPQA_CATEGORIES = {
    # STEM
    "Biology": "STEM",
    "Chemistry": "STEM",
    "Computer Science": "STEM",
    "Engineering": "STEM",
    "Mathematics": "STEM",
    "Medicine": "STEM",
    "Physics": "STEM",
    # Social Science
    "Business": "Social Science",
    "Economics": "Social Science",
    "Education": "Social Science",
    "Political Science": "Social Science",
    "Psychology": "Social Science",
    "Sociology": "Social Science",
    # Humanities
    "Geography": "Humanities",
    "History": "Humanities",
    "Law": "Humanities",
    "Literature": "Humanities",
    "Philosophy": "Humanities",
    # Other
    "Health": "Other",
    "Physical Education": "Other",
}


@register_benchmark("super-gpqa")
class SuperGPQABenchmark(BaseBenchmark):
    name = "SuperGPQA"
    hf_path = "m-a-p/SuperGPQA"
    description = "Comprehensive graduate-level Q&A benchmark across 26 disciplines"

    def get_category_map(self) -> Dict[str, str]:
        """Return subject to category mapping"""
        return SUPER_GPQA_CATEGORIES

    def load(
        self,
        split: str = "train",
        subject: Optional[str] = None,
        max_samples: Optional[int] = None,
        token: Optional[str] = None,
        offline: bool = False,
    ) -> List[Dict[str, Any]]:
        """Load SuperGPQA dataset"""
        return super().load(split="train", subject=subject, max_samples=max_samples, token=token, offline=offline)

    def _parse_row(self, row: Dict) -> Optional[Dict[str, Any]]:
        """Parse SuperGPQA row format

        SuperGPQA fields:
        - question: the question text
        - options: list of options (A, B, C, D)
        - answer_letter: correct answer (A/B/C/D)
        - discipline: subject area
        - field: broader field
        """
        question = row.get("question", "")
        options = row.get("options", [])
        answer = row.get("answer_letter", "")

        if not question or not options:
            return None

        # Normalize answer to uppercase
        if isinstance(answer, str):
            answer_letter = answer.upper()
        else:
            return None

        # Ensure we have 4 options
        choices = options[:4] if len(options) >= 4 else options

        return {
            "question": question,
            "choices": choices,
            "answer": answer_letter,
            "subject": row.get("discipline", row.get("field", "unknown")),
        }

    def get_subjects(self) -> List[str]:
        return [
            "Biology",
            "Business",
            "Chemistry",
            "Computer Science",
            "Economics",
            "Education",
            "Engineering",
            "Geography",
            "Health",
            "History",
            "Law",
            "Literature",
            "Mathematics",
            "Medicine",
            "Philosophy",
            "Physical Education",
            "Physics",
            "Political Science",
            "Psychology",
            "Sociology",
        ]
