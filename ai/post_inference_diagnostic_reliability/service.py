"""Orchestration for the Post-Inference Diagnostic Reliability Layer."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from statistics import fmean

from research.diagnostic_reliability.parser import ParsedDiagnosis
from research.diagnostic_reliability.taxonomy import DiagnosisLabel
from research.diagnostic_reliability.uncertainty import (
    diagnostic_entropy,
    modal_fraction,
)

from .estimator import estimate_diagnostic_reliability
from .models import (
    DiagnosticDecisionState,
    DiagnosticReliabilityResult,
    decision_state_from_policy,
)
from .policy import apply_reliability_policy


def _modal_diagnosis(
    labels: Sequence[DiagnosisLabel],
) -> DiagnosisLabel:
    """Return the unique modal diagnosis.

    The A8.6 experiment observed no modal ties. Production integration
    therefore rejects a tied mode rather than introducing an unvalidated
    tie-breaking rule.
    """
    if not labels:
        raise ValueError(
            "At least one diagnosis label is required"
        )

    counts = Counter(labels)

    highest_count = max(counts.values())

    modes = [
        label
        for label, count in counts.items()
        if count == highest_count
    ]

    if len(modes) != 1:
        raise ValueError(
            "Repeated diagnoses do not have a unique modal diagnosis"
        )

    return modes[0]


def assess_diagnostic_reliability(
    *,
    generations: Sequence[ParsedDiagnosis],
    evidence_ready: bool,
) -> DiagnosticReliabilityResult:
    """Assess reliability of repeated structured diagnostic generations.

    This service preserves the frozen experimental construction:

        repeated diagnoses
            -> modal diagnosis
            -> mean self-confidence
            -> normalised diagnostic entropy
            -> M_EU
            -> reliability policy

    A modal UNKNOWN represents abstention and does not enter M_EU.
    """
    if not generations:
        raise ValueError(
            "At least one parsed generation is required"
        )

    labels = [
        generation.diagnosis
        for generation in generations
    ]

    confidences = [
        generation.confidence
        for generation in generations
    ]

    diagnosis = _modal_diagnosis(
        labels
    )

    mean_confidence = float(
        fmean(confidences)
    )

    entropy = float(
        diagnostic_entropy(
            labels,
            normalise=True,
        )
    )

    agreement = float(
        modal_fraction(labels)
    )

    if diagnosis is DiagnosisLabel.UNKNOWN:
        return DiagnosticReliabilityResult(
            modal_diagnosis=diagnosis,
            evidence_ready=evidence_ready,
            mean_self_confidence=mean_confidence,
            diagnostic_entropy=entropy,
            modal_fraction=agreement,
            predicted_reliability=None,
            decision_state=DiagnosticDecisionState.ABSTAIN,
            generation_count=len(generations),
        )

    predicted_reliability = estimate_diagnostic_reliability(
        evidence_ready=evidence_ready,
        mean_self_confidence=mean_confidence,
        diagnostic_entropy=entropy,
    )

    policy_state = apply_reliability_policy(
        evidence_ready=evidence_ready,
        predicted_reliability=predicted_reliability,
    )

    return DiagnosticReliabilityResult(
        modal_diagnosis=diagnosis,
        evidence_ready=evidence_ready,
        mean_self_confidence=mean_confidence,
        diagnostic_entropy=entropy,
        modal_fraction=agreement,
        predicted_reliability=predicted_reliability,
        decision_state=decision_state_from_policy(
            policy_state
        ),
        generation_count=len(generations),
    )