"""Ranking and answer metrics."""
from __future__ import annotations

import math
import re
import string
import unicodedata
from collections import Counter
from typing import Iterable, List, Sequence, Set


def recall_at_k(ranked: Sequence[str], relevant: Set[str], k: int) -> float:
    """
    1.0 if any relevant chunk appears in the top k, else 0.0.

    SQuAD has one answer location per question, so this equals hit rate@k.
    Averaged over queries it is the usual "recall@k" reported for QA retrieval.
    """
    return 1.0 if any(chunk_id in relevant for chunk_id in ranked[:k]) else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: Set[str], k: int) -> float:
    """1 / rank of the first relevant chunk within the top k, else 0.0."""
    for rank, chunk_id in enumerate(ranked[:k], start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def percentile(values: Iterable[float], p: float) -> float:
    """Nearest-rank percentile: the smallest value with at least p% of values at or below it."""
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


def _normalize_answer(text: str) -> List[str]:
    """
    SQuAD's official normalization (lowercase, strip punctuation and articles), extended to
    Unicode: newer models write curly quotes, en/em dashes and m³, which the ASCII-only
    original leaves in place, so "Brocard’s" failed to match "Brocard's".
    """
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(
        ch for ch in text if ch not in set(string.punctuation) and not unicodedata.category(ch).startswith("P")
    )
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return text.split()


def token_f1(prediction: str, gold_answers: Sequence[str]) -> float:
    """Best SQuAD token-level F1 of a prediction against any gold answer."""
    pred_tokens = _normalize_answer(prediction)
    best = 0.0
    for gold in gold_answers:
        gold_tokens = _normalize_answer(gold)
        common = Counter(pred_tokens) & Counter(gold_tokens)
        overlap = sum(common.values())
        if overlap == 0:
            continue
        precision = overlap / len(pred_tokens)
        recall = overlap / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def answer_recall(prediction: str, gold_answers: Sequence[str]) -> float:
    """
    1.0 if any normalized gold answer appears inside the prediction.

    Token F1 punishes the long, sentence-style answers an LLM gives; this is
    the more useful correctness signal for a chat-style RAG system.
    """
    pred = f" {' '.join(_normalize_answer(prediction))} "
    for gold in gold_answers:
        gold_norm = " ".join(_normalize_answer(gold))
        if gold_norm and f" {gold_norm} " in pred:
            return 1.0
    return 0.0
