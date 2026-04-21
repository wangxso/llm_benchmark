"""Answer extraction and scoring utilities"""

import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def extract_answer(response: str, num_options: int = 10) -> Optional[str]:
    """Extract answer letter from model response

    Supports multiple formats:
    - Single letter: "A", "B", "C", "D", etc. (up to num_options)
    - With parentheses: "(A)", "[A]"
    - With word: "Answer: A", "The answer is A"
    - Chinese: "答案是A", "选择A"

    Args:
        response: Model response text
        num_options: Maximum number of options (default 10 for A-J)

    Returns:
        Answer letter or None if not found
    """
    response = response.strip().upper()

    # Build valid option letters
    valid_letters = "".join(chr(ord("A") + i) for i in range(num_options))

    # try direct single letter match
    if response in list(valid_letters):
        return response

    # Pattern priority: more specific patterns first
    option_pattern = f"[A-{valid_letters[-1]}]"
    patterns = [
        # "Answer: A" or "answer is A" patterns
        rf"(?:ANSWER|ANS)\s*[:：]?\s*({option_pattern})",
        rf"(?:THE\s+)?ANSWER\s+IS\s+({option_pattern})",
        # Chinese patterns: 答案是A, 答案：A, 选择A
        rf"答案\s*(?:是)?\s*[:：]?\s*({option_pattern})",
        rf"选择\s*[:：]?\s*({option_pattern})",
        # Parentheses/brackets: (A) [A] 【A】
        rf"[【\(\[]({option_pattern})[】\)\]]",
        # Standalone letter at start/end of response
        rf"^\s*({option_pattern})\s*[\.。]?$",
        rf"[,\.。\s]({option_pattern})[\s\.。,]*$",
        # Any single letter (fallback)
        rf"\b({option_pattern})\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            return match.group(1)

    return None


def score_results(
    results: List[Dict],
) -> Dict:
    """Calculate accuracy metrics from evaluation results

    Args:
        results: List of dicts with 'predicted', 'actual', 'subject' fields

    Returns:
        Dict with overall accuracy and per-subject breakdown
    """
    total = len(results)
    if total == 0:
        return {
            "overall_accuracy": 0.0,
            "total_questions": 0,
            "correct": 0,
            "subjects": {},
        }

    correct = 0
    subject_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for result in results:
        predicted = result.get("predicted")
        actual = result.get("actual")
        subject = result.get("subject", "unknown")

        subject_stats[subject]["total"] += 1

        if predicted == actual:
            correct += 1
            subject_stats[subject]["correct"] += 1

    subject_accuracy = {}
    for subject, stats in subject_stats.items():
        subject_accuracy[subject] = {
            "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0,
            "correct": stats["correct"],
            "total": stats["total"],
        }

    return {
        "overall_accuracy": correct / total,
        "total_questions": total,
        "correct": correct,
        "subjects": subject_accuracy,
    }


def compute_confusion_matrix(
    results: List[Dict],
) -> Dict[str, Dict[str, int]]:
    """Compute confusion matrix for A/B/C/D predictions"""
    confusion = {letter: {l: 0 for l in "ABCD"} for letter in "ABCD"}

    for result in results:
        predicted = result.get("predicted")
        actual = result.get("actual")
        if predicted in "ABCD" and actual in "ABCD":
            confusion[actual][predicted] += 1

    return confusion


def score_results_with_categories(
    results: List[Dict],
    category_map: Optional[Dict[str, str]] = None,
) -> Dict:
    """Calculate accuracy metrics with category aggregation

    Args:
        results: List of dicts with 'predicted', 'actual', 'subject' fields
        category_map: Optional mapping from subject to category
                     e.g., {"physics": "STEM", "history": "Humanities"}

    Returns:
        Dict with overall accuracy, per-subject and per-category breakdown
    """
    total = len(results)
    if total == 0:
        return {
            "overall_accuracy": 0.0,
            "total_questions": 0,
            "correct": 0,
            "subjects": {},
            "categories": {},
        }

    correct = 0
    subject_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    category_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for result in results:
        predicted = result.get("predicted")
        actual = result.get("actual")
        subject = result.get("subject", "unknown")

        subject_stats[subject]["total"] += 1

        if predicted == actual:
            correct += 1
            subject_stats[subject]["correct"] += 1

        # Category aggregation
        if category_map:
            category = category_map.get(subject, "Other")
            category_stats[category]["total"] += 1
            if predicted == actual:
                category_stats[category]["correct"] += 1

    subject_accuracy = {}
    for subject, stats in subject_stats.items():
        subject_accuracy[subject] = {
            "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0,
            "correct": stats["correct"],
            "total": stats["total"],
        }

    category_accuracy = {}
    if category_map:
        for category, stats in category_stats.items():
            category_accuracy[category] = {
                "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0,
                "correct": stats["correct"],
                "total": stats["total"],
            }
        # Calculate average across categories
        if category_accuracy:
            avg = sum(c["accuracy"] for c in category_accuracy.values()) / len(category_accuracy)
            category_accuracy["Average"] = {
                "accuracy": avg,
                "correct": correct,
                "total": total,
            }

    return {
        "overall_accuracy": correct / total,
        "total_questions": total,
        "correct": correct,
        "subjects": subject_accuracy,
        "categories": category_accuracy,
    }
