"""Metric primitives shared by the evaluators.

Kept dependency-light and explicit: retrieval metrics are a few lines each, and the
judge metrics wrap ``sklearn`` only for Cohen's kappa. Everything here is pure and
unit-tested so the headline numbers can't drift silently.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


def hit_rate_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    """1.0 if any relevant id appears in the top-k, else 0.0."""
    return 1.0 if set(ranked[:k]) & relevant else 0.0


def recall_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    """Fraction of the relevant ids that appear in the top-k."""
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank(relevant: set[str], ranked: list[str]) -> float:
    """1/rank of the first relevant id (0.0 if none retrieved)."""
    for i, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def cohen_kappa(human: list[str], judge: list[str]) -> float | None:
    """Cohen's kappa between two label sequences.

    Returns ``None`` when it is undefined - fewer than two items, or only one label
    value present across both raters (kappa needs variation to mean anything). This is
    reported honestly rather than papered over with a 0.
    """
    if len(human) < 2:
        return None
    if len(set(human) | set(judge)) < 2:
        return None
    from sklearn.metrics import cohen_kappa_score

    labels = sorted(set(human) | set(judge))
    return round(float(cohen_kappa_score(human, judge, labels=labels)), 3)


@dataclass
class PRF:
    """Precision / recall / F1 for one target class."""

    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def prf_for_class(human: list[str], judge: list[str], target: str) -> PRF:
    """Precision/recall/F1 treating ``target`` as the positive class.

    For the judge, the positive class is ``violation``: catching the bad responses is
    the job, so this is the metric that most directly says whether the judge is useful.
    """
    tp = sum(1 for h, j in zip(human, judge, strict=True) if h == target and j == target)
    fp = sum(1 for h, j in zip(human, judge, strict=True) if h != target and j == target)
    fn = sum(1 for h, j in zip(human, judge, strict=True) if h == target and j != target)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return PRF(round(precision, 3), round(recall, 3), round(f1, 3), tp, fp, fn)


def confusion(human: list[str], judge: list[str]) -> dict[tuple[str, str], int]:
    """Counts of (human_label, judge_label) pairs."""
    return dict(Counter(zip(human, judge, strict=True)))


def top_disagreement(human: list[str], judge: list[str]) -> tuple[str, str, int] | None:
    """The most frequent (human, judge) pair where the two disagree, or None."""
    off = {pair: n for pair, n in confusion(human, judge).items() if pair[0] != pair[1]}
    if not off:
        return None
    (h, j), n = max(off.items(), key=lambda kv: kv[1])
    return h, j, n
