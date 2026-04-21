"""C-Eval benchmark loader

C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models
https://huggingface.co/datasets/ceval/ceval-exam
"""

from typing import Dict, Any, Optional, List
from ..base import BaseBenchmark
from ..registry import register_benchmark


# C-Eval 各学科配置
CEVAL_SUBJECTS = [
    "accountant", "advanced_mathematics", "art_studies", "basic_medicine",
    "business_administration", "chinese_language_and_literature", "civil_servant",
    "clinical_medicine", "college_chemistry", "college_economics", "college_physics",
    "college_programming", "computer_architecture", "computer_network",
    "discrete_mathematics", "education_science", "electrical_engineer",
    "environmental_impact_assessment_engineer", "fire_engineer", "high_school_biology",
    "high_school_chemistry", "high_school_chinese", "high_school_geography",
    "high_school_history", "high_school_mathematics", "high_school_physics",
    "high_school_politics", "ideological_and_moral_cultivation", "law",
    "legal_professional", "logic", "mao_zedong_thought", "marxism",
    "metrology_engineer", "middle_school_biology", "middle_school_chemistry",
    "middle_school_geography", "middle_school_history", "middle_school_mathematics",
    "middle_school_physics", "middle_school_politics", "modern_chinese_history",
    "operating_system", "physician", "plant_protection", "probability_and_statistics",
    "professional_tour_guide", "sports_science", "tax_accountant",
    "teacher_qualification", "urban_and_rural_planner", "veterinary_medicine",
]

# C-Eval 学科分类
CEVAL_CATEGORIES = {
    "STEM": [
        "advanced_mathematics", "college_chemistry", "college_economics",
        "college_physics", "college_programming", "computer_architecture",
        "computer_network", "discrete_mathematics", "electrical_engineer",
        "high_school_biology", "high_school_chemistry", "high_school_mathematics",
        "high_school_physics", "middle_school_biology", "middle_school_chemistry",
        "middle_school_mathematics", "middle_school_physics", "operating_system",
        "probability_and_statistics", "veterinary_medicine",
    ],
    "Social Science": [
        "business_administration", "education_science", "high_school_geography",
        "high_school_politics", "mao_zedong_thought", "marxism",
        "middle_school_geography", "middle_school_politics", "teacher_qualification",
    ],
    "Humanities": [
        "art_studies", "chinese_language_and_literature", "high_school_chinese",
        "high_school_history", "ideological_and_moral_cultivation", "law",
        "legal_professional", "logic", "middle_school_history",
        "modern_chinese_history", "professional_tour_guide",
    ],
    "Other": [
        "accountant", "basic_medicine", "civil_servant", "clinical_medicine",
        "environmental_impact_assessment_engineer", "fire_engineer",
        "metrology_engineer", "physician", "plant_protection", "sports_science",
        "tax_accountant", "urban_and_rural_planner",
    ],
}


def get_category(subject: str) -> str:
    """Get category for a subject"""
    for category, subjects in CEVAL_CATEGORIES.items():
        if subject in subjects:
            return category
    return "Other"


@register_benchmark("ceval")
class CEvalBenchmark(BaseBenchmark):
    name = "C-Eval"
    hf_path = "ceval/ceval-exam"
    description = "中文综合能力评测，覆盖52个学科"

    def get_category_map(self) -> Dict[str, str]:
        """Return subject to category mapping"""
        mapping = {}
        for category, subjects in CEVAL_CATEGORIES.items():
            for subject in subjects:
                mapping[subject] = category
        return mapping

    def load(
        self,
        split: str = "val",
        subject: Optional[str] = None,
        max_samples: Optional[int] = None,
        token: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Load C-Eval dataset

        C-Eval has splits: val, test, dev
        Each subject is a separate config.
        """
        from datasets import load_dataset
        import os

        # Get token
        if token is None:
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            if token is None:
                try:
                    from huggingface_hub import HfFolder
                    token = HfFolder.get_token()
                except Exception:
                    pass

        items = []
        subjects_to_load = [subject] if subject else CEVAL_SUBJECTS

        for subj in subjects_to_load:
            try:
                load_kwargs = {
                    "path": self.hf_path,
                    "name": subj,
                    "split": split,
                    "trust_remote_code": True,
                }
                if token:
                    load_kwargs["token"] = token

                ds = load_dataset(**load_kwargs)

                for row in ds:
                    item = self._parse_row(row, subj)
                    if item:
                        items.append(item)
                        if max_samples and len(items) >= max_samples:
                            return items

            except Exception:
                # Skip subjects that fail to load
                continue

        return items

    def _parse_row(self, row: Dict, subject: str) -> Optional[Dict[str, Any]]:
        """Parse C-Eval row format

        C-Eval fields:
        - question: 题目
        - A, B, C, D: 选项
        - answer: 正确答案 (A/B/C/D)
        - explanation: 解释（可选）
        """
        question = row.get("question", "")
        choice_a = row.get("A", "")
        choice_b = row.get("B", "")
        choice_c = row.get("C", "")
        choice_d = row.get("D", "")
        answer = row.get("answer", "")

        if not question or not answer:
            return None

        choices = [choice_a, choice_b, choice_c, choice_d]
        # Filter out empty choices
        choices = [c for c in choices if c]

        if len(choices) < 2:
            return None

        answer_letter = answer.upper()

        return {
            "question": question,
            "choices": choices,
            "answer": answer_letter,
            "subject": subject,
        }

    def get_subjects(self) -> List[str]:
        """C-Eval subjects"""
        return CEVAL_SUBJECTS
