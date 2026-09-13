"""Tests for frozen validation semantics in the experimental prompt."""

from __future__ import annotations

from research.diagnostic_reliability.prompt import (
    SYSTEM_PROMPT,
    VALIDATION_SEMANTICS,
)


def test_validation_semantics_are_part_of_system_prompt() -> None:
    assert VALIDATION_SEMANTICS in SYSTEM_PROMPT


def test_validation_is_defined_as_evidence_readiness() -> None:
    lower = VALIDATION_SEMANTICS.lower()

    assert "readiness" in lower
    assert "integrity" in lower
    assert "operational health" in lower


def test_validation_does_not_imply_normal_operation() -> None:
    assert (
        "do not imply a NORMAL operational condition"
        in VALIDATION_SEMANTICS
    )


def test_validation_does_not_imply_diagnostic_correctness() -> None:
    assert (
        "do not indicate that a diagnosis is correct"
        in VALIDATION_SEMANTICS
    )