"""Frozen post-inference diagnostic reliability estimator."""

from __future__ import annotations

import math


# Frozen A8.11.2 final-development M_EU parameters.
FLIC_INTERCEPT = -19.046914056
BETA_E = 6.446372648
BETA_MEAN_SELF_CONFIDENCE = 14.868349383
BETA_DIAGNOSTIC_ENTROPY = 3.118253729


def _sigmoid(value: float) -> float:
    """Return a numerically stable logistic transformation."""
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)

    z = math.exp(value)
    return z / (1.0 + z)


def estimate_diagnostic_reliability(
    *,
    evidence_ready: bool,
    mean_self_confidence: float,
    diagnostic_entropy: float,
) -> float:
    """Estimate the probability that a substantive diagnosis is correct.

    Implements the M_EU estimator frozen at A8.11.2.

    This estimator is defined only for substantive diagnoses.
    UNKNOWN is handled separately by the orchestration layer.
    """
    if not 0.0 <= mean_self_confidence <= 1.0:
        raise ValueError(
            "mean_self_confidence must be between 0 and 1"
        )

    if not 0.0 <= diagnostic_entropy <= 1.0:
        raise ValueError(
            "diagnostic_entropy must be between 0 and 1"
        )

    evidence_value = 1.0 if evidence_ready else 0.0

    linear_predictor = (
        FLIC_INTERCEPT
        + BETA_E * evidence_value
        + BETA_MEAN_SELF_CONFIDENCE * mean_self_confidence
        + BETA_DIAGNOSTIC_ENTROPY * diagnostic_entropy
    )

    return _sigmoid(linear_predictor)