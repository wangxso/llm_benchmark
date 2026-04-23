"""Answer extraction and scoring utilities - aligned with official MMLU-Pro evaluator"""

import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def extract_answer(text: str, num_options: int = 10) -> Optional[str]:
    """Extract answer letter using official MMLU-Pro extraction logic

    Priority:
    1. Match "answer is (X)" or "answer is X" (case insensitive)
    2. Match "Answer: X" or "answer: X"
    3. Find last standalone letter A-J in text

    Args:
        text: Model response text
        num_options: Maximum number of options (default 10 for A-J)

    Returns:
        Answer letter or None if not found
    """
    if not text:
        return None

    # Build valid option letters
    valid_letters = set(chr(ord("A") + i) for i in range(num_options))

    # Level 1: Match "answer is (X)" or "answer is X"
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        letter = match.group(1).upper()
        if letter in valid_letters:
            return letter

    # Level 2: Match "Answer: X" or "answer: X"
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        letter = match.group(1).upper()
        if letter in valid_letters:
            return letter

    # Level 3: Find last standalone letter A-J (official fallback)
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        letter = match.group(0).upper()
        if letter in valid_letters:
            return letter

    return None


def extract_answer_with_fallback(text: str, num_options: int = 10) -> str:
    """Extract answer, use random choice if extraction fails

    Args:
        text: Model response text
        num_options: Number of options for random fallback

    Returns:
        Answer letter (never None)
    """
    import random
    result = extract_answer(text, num_options)
    if result is None:
        # Random choice as fallback
        return random.choice([chr(ord("A") + i) for i in range(num_options)])
    return result


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
