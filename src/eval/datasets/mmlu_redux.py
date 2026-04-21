"""MMLU-Redux benchmark loader

MMLU-Redux: A version of MMLU with errors corrected and ambiguous questions removed
https://huggingface.co/datasets/edinburgh-dawg/mmlu-redux
"""

from typing import Dict, Any, Optional, List
from ..base import BaseBenchmark
from ..registry import register_benchmark


@register_benchmark("mmlu-redux")
class MMLUReduxBenchmark(BaseBenchmark):
    name = "MMLU-Redux"
    hf_path = "edinburgh-dawg/mmlu-redux"
    description = "MMLU with corrected errors and removed ambiguous questions"

    def load(
        self,
        split: str = "test",
        subject: Optional[str] = None,
        max_samples: Optional[int] = None,
        token: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Load MMLU-Redux dataset"""
        # MMLU-Redux uses subject-specific splits
        if subject:
            try:
                return super().load(split=subject, max_samples=max_samples, token=token)
            except Exception:
                pass
        return super().load(split=split, subject=subject, max_samples=max_samples, token=token)

    def _parse_row(self, row: Dict) -> Optional[Dict[str, Any]]:
        """Parse MMLU-Redux row format

        MMLU-Redux fields similar to MMLU:
        - question: the question text
        - choices: list of 4 options
        - answer: correct option (0-3 or A-D)
        - subject: subject area
        """
        question = row.get("question", "")
        choices = row.get("choices", [])
        answer = row.get("answer")

        if not question or not choices:
            return None

        # Handle different answer formats
        if isinstance(answer, str):
            if answer in ["A", "B", "C", "D"]:
                answer_letter = answer
            else:
                try:
                    answer_idx = int(answer)
                    answer_letter = chr(ord("A") + answer_idx)
                except ValueError:
                    return None
        elif isinstance(answer, int):
            answer_letter = chr(ord("A") + answer)
        else:
            return None

        return {
            "question": question,
            "choices": choices[:4] if len(choices) >= 4 else choices,
            "answer": answer_letter,
            "subject": row.get("subject", "unknown"),
        }

    def get_subjects(self) -> List[str]:
        return [
            "abstract_algebra",
            "anatomy",
            "astronomy",
            "business_ethics",
            "clinical_knowledge",
            "college_biology",
            "college_chemistry",
            "college_computer_science",
            "college_mathematics",
            "college_medicine",
            "college_physics",
            "computer_security",
            "conceptual_physics",
            "econometrics",
            "electrical_engineering",
            "elementary_mathematics",
            "formal_logic",
            "global_facts",
            "high_school_biology",
            "high_school_chemistry",
            "high_school_computer_science",
            "high_school_european_history",
            "high_school_geography",
            "high_school_government_and_politics",
            "high_school_macroeconomics",
            "high_school_mathematics",
            "high_school_microeconomics",
            "high_school_physics",
            "high_school_psychology",
            "high_school_statistics",
            "high_school_us_history",
            "high_school_world_history",
            "human_aging",
            "human_sexuality",
            "international_law",
            "jurisprudence",
            "logical_fallacies",
            "machine_learning",
            "management",
            "marketing",
            "medical_genetics",
            "miscellaneous",
            "moral_disputes",
            "moral_scenarios",
            "nutrition",
            "philosophy",
            "prehistory",
            "professional_accounting",
            "professional_law",
            "professional_medicine",
            "professional_psychology",
            "public_relations",
            "security_studies",
            "sociology",
            "us_foreign_policy",
            "virology",
            "world_religions",
        ]
