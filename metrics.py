"""
metrics.py - Retrieval and generation metrics.
"""
from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & set(relevant)) / len(set(relevant))


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Standard SQuAD-style answer normalization."""
    # Strip accents
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Collapse whitespace
    return " ".join(text.split())

def exact_match(prediction: str, gold_answers: list[str]) -> float:
    pred = _normalise(prediction)
    return float(any(_normalise(a) == pred for a in gold_answers))

def token_f1(prediction: str, gold_answers: list[str]) -> float:
    pred_tokens = _normalise(prediction).split()
    if not pred_tokens:
        return 0.0
    best = 0.0
    for ans in gold_answers:
        gold_tokens = _normalise(ans).split()
        common = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
        if common == 0:
            continue
        p = common / len(pred_tokens)
        r = common / len(gold_tokens)
        best = max(best, 2 * p * r / (p + r))
    return best

def bertscore(predictions: list[str], references: list[list[str]]) -> dict:
    from bert_score import score

    if not predictions:
        return {"bertscore_p": 0.0, "bertscore_r": 0.0, "bertscore_f1": 0.0}

    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")

    flat_predictions: list[str] = []
    flat_references: list[str] = []
    reference_counts: list[int] = []
    for prediction, answer_list in zip(predictions, references):
        cleaned_answers = [answer.strip() for answer in answer_list if isinstance(answer, str) and answer.strip()]
        if not cleaned_answers:
            cleaned_answers = [""]
        reference_counts.append(len(cleaned_answers))
        flat_predictions.extend([prediction] * len(cleaned_answers))
        flat_references.extend(cleaned_answers)

    precision, recall, f1 = score(
        flat_predictions,
        flat_references,
        lang="en",
        rescale_with_baseline=True,
        verbose=False,
    )

    best_precision: list[float] = []
    best_recall: list[float] = []
    best_f1: list[float] = []
    start = 0
    for count in reference_counts:
        precision_slice = precision[start : start + count]
        recall_slice = recall[start : start + count]
        f1_slice = f1[start : start + count]
        best_index = int(f1_slice.argmax().item())
        best_precision.append(float(precision_slice[best_index]))
        best_recall.append(float(recall_slice[best_index]))
        best_f1.append(float(f1_slice[best_index]))
        start += count

    return {
        "bertscore_p": sum(best_precision) / len(best_precision),
        "bertscore_r": sum(best_recall) / len(best_recall),
        "bertscore_f1": sum(best_f1) / len(best_f1),
    }
