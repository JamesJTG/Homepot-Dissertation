"""Result models for the Post-Inference Diagnostic Reliability Layer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from research.diagnostic_reliability.taxonomy import DiagnosisLabel

from .policy import ReliabilityPolicyState


class DiagnosticDecisionState(str, Enum):
    """Final decision-support state produced by the layer."""

    ABSTAIN = "ABSTAIN"
    WITHHOLD = "WITHHOLD"
    QUALIFY = "QUALIFY"
    ISSUE = "ISSUE"


@dataclass(frozen=True)
class DiagnosticReliabilityResult:
    """Complete output of post-inference diagnostic reliability assessment."""

    modal_diagnosis: DiagnosisLabel
    evidence_ready: bool

    mean_self_confidence: float
    diagnostic_entropy: float
    modal_fraction: float

    predicted_reliability: float | None
    decision_state: DiagnosticDecisionState

    generation_count: int

    @property
    def is_abstention(self) -> bool:
        return self.decision_state is DiagnosticDecisionState.ABSTAIN

    @property
    def has_reliability_estimate(self) -> bool:
        return self.predicted_reliability is not None


def decision_state_from_policy(
    state: ReliabilityPolicyState,
) -> DiagnosticDecisionState:
    """Map the frozen reliability-gate state to the layer output state."""

    return DiagnosticDecisionState(state.value)