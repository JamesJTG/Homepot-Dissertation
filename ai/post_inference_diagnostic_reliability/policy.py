"""Frozen post-inference diagnostic reliability policy."""

from __future__ import annotations

from enum import Enum


# Frozen A8.10 development-selected Issue threshold.
ISSUE_THRESHOLD = 0.64


class ReliabilityPolicyState(str, Enum):
    """Decision-support states produced by the reliability gate."""

    WITHHOLD = "WITHHOLD"
    QUALIFY = "QUALIFY"
    ISSUE = "ISSUE"


def apply_reliability_policy(
    *,
    evidence_ready: bool,
    predicted_reliability: float,
) -> ReliabilityPolicyState:
    """Apply the validation-preserving A8.10 reliability policy.

    Non-actionable upstream evidence must always remain WITHHOLD.
    Downstream reliability therefore cannot override the existing
    validation envelope.
    """
    if not 0.0 <= predicted_reliability <= 1.0:
        raise ValueError(
            "predicted_reliability must be between 0 and 1"
        )

    if not evidence_ready:
        return ReliabilityPolicyState.WITHHOLD

    if predicted_reliability >= ISSUE_THRESHOLD:
        return ReliabilityPolicyState.ISSUE

    return ReliabilityPolicyState.QUALIFY