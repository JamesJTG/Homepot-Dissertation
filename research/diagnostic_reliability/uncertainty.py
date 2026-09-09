"""Sampling-based uncertainty measures for structured diagnosis labels."""

from __future__ import annotations

from collections import Counter
import math
from typing import Iterable

from .taxonomy import DiagnosisLabel


def diagnosis_distribution(labels: Iterable[DiagnosisLabel]) -> dict[str, float]:
    values = [label.value for label in labels]
    if not values:
        raise ValueError("At least one diagnosis label is required")
    counts = Counter(values)
    n = len(values)
    return {label: count / n for label, count in sorted(counts.items())}


def diagnostic_entropy(labels: Iterable[DiagnosisLabel], *, normalise: bool = True) -> float:
    """Entropy over structured diagnosis classes.

    When normalised, K is the number of labels *observed in the repeated
    generations*. Perfect agreement is 0.0; a uniform split across observed
    alternatives is 1.0.
    """
    dist = diagnosis_distribution(labels)
    entropy = -sum(p * math.log(p) for p in dist.values())
    if not normalise or len(dist) <= 1:
        return entropy
    return entropy / math.log(len(dist))


def modal_fraction(labels: Iterable[DiagnosisLabel]) -> float:
    values = [label.value for label in labels]
    if not values:
        raise ValueError("At least one diagnosis label is required")
    return max(Counter(values).values()) / len(values)
