"""GPQA benchmark loader

GPQA: Graduate-Level Google-Proof Q&A Benchmark
https://huggingface.co/datasets/idavidrein/gpqa
"""

import random
from typing import Dict, Any, Optional, List
from ..base import BaseBenchmark
from ..registry import register_benchmark


GPQA_CATEGORIES = {
    "Biology": "STEM",
    "Physics": "STEM",
    "Chemistry": "STEM",
}


@register_benchmark("gpqa")
class GPQABenchmark(BaseBenchmark):
    name = "GPQA"
    hf_path = "idavidrein/gpqa"
    hf_name = "gpqa_main"
    description = "Graduate-Level Google-Proof Q&A Benchmark"
    requires_auth = True

    def get_category_map(self) -> Dict[str, str]:
        """Return subject to category mapping"""
        return GPQA_CATEGORIES

    def load(
        self,
        split: str = "train",
        subject: Optional[str] = None,
        max_samples: Optional[int] = None,
        token: Optional[str] = None,
        offline: bool = False,
    ) -> List[Dict[str, Any]]:
        """Load GPQA dataset

        GPQA uses 'train' split as it's a single dataset.
        """
        return super().load(split="train", subject=subject, max_samples=max_samples, token=token, offline=offline)

    def _parse_row(self, row: Dict) -> Optional[Dict[str, Any]]:
        """Parse GPQA row format

        GPQA fields:
        - Question: the question text
        - Correct Answer: the correct answer text
        - Incorrect Answer 1/2/3: distractor answers
        - High-level domain: subject area (Biology, Physics, Chemistry)
        - Subdomain: more specific subfield
        """
        question = row.get("Question", "")
        correct = row.get("Correct Answer", "")
        incorrect = [
            row.get("Incorrect Answer 1", ""),
            row.get("Incorrect Answer 2", ""),
            row.get("Incorrect Answer 3", ""),
        ]

        if not question or not correct:
            return None

        # Shuffle choices and track correct position
        choices = [correct] + incorrect
        random.shuffle(choices)

        # Find correct answer position
        correct_idx = choices.index(correct)
        answer_letter = chr(ord("A") + correct_idx)

        # Use High-level domain as subject
        subject = row.get("High-level domain", row.get("Subdomain", "unknown"))

        return {
            "question": question,
            "choices": choices,
            "answer": answer_letter,
            "subject": subject,
        }

    def get_subjects(self) -> List[str]:
        return [
            "Biology",
            "Physics",
            "Chemistry",
        ]
