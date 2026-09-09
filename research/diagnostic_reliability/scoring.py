"""Objective diagnostic correctness and abstention scoring."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .taxonomy import DiagnosisLabel, EvidenceSufficiency, FaultLabel


class ErrorCategory(str, Enum):
    CORRECT = "CORRECT"
    WRONG_FAULT = "WRONG_FAULT"
    FALSE_NORMAL = "FALSE_NORMAL"
    FALSE_FAULT = "FALSE_FAULT"
    UNNECESSARY_ABSTENTION = "UNNECESSARY_ABSTENTION"
    APPROPRIATE_ABSTENTION = "APPROPRIATE_ABSTENTION"
    UNSUPPORTED_DIAGNOSIS = "UNSUPPORTED_DIAGNOSIS"


@dataclass(frozen=True)
class ScoreResult:
    diagnostic_correctness: int | None
    decision_appropriateness: int
    error_category: ErrorCategory


def score_diagnosis(
    truth: FaultLabel,
    diagnosis: DiagnosisLabel,
    sufficiency: EvidenceSufficiency,
) -> ScoreResult:
    """Score primary correctness separately from secondary decision appropriateness.

    UNKNOWN is excluded from primary correctness (None), matching the M1.4
    methodological decision, and evaluated through decision appropriateness.
    """
    if diagnosis is DiagnosisLabel.UNKNOWN:
        if sufficiency is EvidenceSufficiency.INSUFFICIENT:
            return ScoreResult(None, 1, ErrorCategory.APPROPRIATE_ABSTENTION)
        return ScoreResult(None, 0, ErrorCategory.UNNECESSARY_ABSTENTION)

    if sufficiency is EvidenceSufficiency.INSUFFICIENT:
        return ScoreResult(0, 0, ErrorCategory.UNSUPPORTED_DIAGNOSIS)

    if diagnosis.value == truth.value:
        return ScoreResult(1, 1, ErrorCategory.CORRECT)

    if truth is FaultLabel.NORMAL:
        return ScoreResult(0, 0, ErrorCategory.FALSE_FAULT)
    if diagnosis is DiagnosisLabel.NORMAL:
        return ScoreResult(0, 0, ErrorCategory.FALSE_NORMAL)
    return ScoreResult(0, 0, ErrorCategory.WRONG_FAULT)
