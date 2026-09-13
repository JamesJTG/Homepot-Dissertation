"""Sampling-based uncertainty measures for structured diagnosis labels."""

from __future__ import annotations

from collections import Counter
import math
from typing import Iterable

from .taxonomy import DiagnosisLabel


# Repeated generations may produce any member of the complete diagnostic
# output taxonomy, including UNKNOWN. Using a fixed support makes normalised
# entropy comparable across experimental cases.
DIAGNOSIS_SUPPORT_SIZE = len(DiagnosisLabel)


def diagnosis_distribution(
    labels: Iterable[DiagnosisLabel],
) -> dict[str, float]:
    """Return the empirical distribution of labels that were observed."""

    values = [
        label.value
        for label in labels
    ]

    if not values:
        raise ValueError(
            "At least one diagnosis label is required"
        )

    counts = Counter(values)
    n = len(values)

    return {
        label: count / n
        for label, count in sorted(
            counts.items()
        )
    }


def diagnostic_entropy(
    labels: Iterable[DiagnosisLabel],
    *,
    normalise: bool = True,
) -> float:
    """Return entropy over repeated structured diagnostic outputs.

    Raw entropy is calculated from the empirical distribution of labels
    observed in the repeated generations.

    When ``normalise`` is True, entropy is divided by ``log(K)``, where K is
    the fixed size of the complete DiagnosisLabel output space. This makes
    values comparable across experimental cases even when different numbers
    of distinct labels happen to be observed.

    The full output taxonomy includes UNKNOWN because it is a legitimate
    model output and therefore part of the uncertainty support.

    A value of 0.0 represents complete agreement. A theoretical value of 1.0
    represents a uniform distribution over the complete diagnostic taxonomy.
    With a finite number of repeated generations, the empirical maximum may
    be below 1.0 when not every class can be represented equally.
    """

    dist = diagnosis_distribution(
        labels
    )

    entropy = -sum(
        probability
        * math.log(probability)
        for probability in dist.values()
    )

    if not normalise:
        return entropy

    if entropy == 0.0:
        return 0.0

    if DIAGNOSIS_SUPPORT_SIZE <= 1:
        return 0.0

    return (
        entropy
        / math.log(DIAGNOSIS_SUPPORT_SIZE)
    )


def modal_fraction(
    labels: Iterable[DiagnosisLabel],
) -> float:
    """Return the fraction assigned to the most frequent diagnosis."""

    values = [
        label.value
        for label in labels
    ]

    if not values:
        raise ValueError(
            "At least one diagnosis label is required"
        )

    return (
        max(
            Counter(values).values()
        )
        / len(values)
    )