"""Tests for strict experimental diagnostic parsing."""

import json

import pytest

from research.diagnostic_reliability.parser import parse_diagnosis
from research.diagnostic_reliability.taxonomy import DiagnosisLabel


def test_valid_diagnosis_parses() -> None:
    raw = json.dumps(
        {
            "diagnosis": "CPU_SATURATION",
            "confidence": 0.82,
            "explanation": "CPU remains elevated across the observation window.",
        }
    )

    parsed = parse_diagnosis(raw)

    assert parsed.diagnosis is DiagnosisLabel.CPU_SATURATION
    assert parsed.confidence == 0.82
    assert parsed.explanation == (
        "CPU remains elevated across the observation window."
    )


def test_extra_key_is_rejected() -> None:
    raw = json.dumps(
        {
            "diagnosis": "NORMAL",
            "confidence": 0.9,
            "explanation": "Evidence is consistent with normal operation.",
            "extra": "not allowed",
        }
    )

    with pytest.raises(ValueError):
        parse_diagnosis(raw)


def test_non_string_diagnosis_is_rejected() -> None:
    raw = json.dumps(
        {
            "diagnosis": 123,
            "confidence": 0.5,
            "explanation": "Malformed diagnosis field.",
        }
    )

    with pytest.raises(ValueError):
        parse_diagnosis(raw)


def test_unknown_taxonomy_value_is_rejected() -> None:
    raw = json.dumps(
        {
            "diagnosis": "DISK_FAILURE",
            "confidence": 0.5,
            "explanation": "Unsupported label.",
        }
    )

    with pytest.raises(ValueError):
        parse_diagnosis(raw)


def test_string_confidence_is_rejected() -> None:
    raw = json.dumps(
        {
            "diagnosis": "NORMAL",
            "confidence": "0.75",
            "explanation": "Confidence must be numeric JSON.",
        }
    )

    with pytest.raises(ValueError):
        parse_diagnosis(raw)


def test_boolean_confidence_is_rejected() -> None:
    raw = json.dumps(
        {
            "diagnosis": "NORMAL",
            "confidence": True,
            "explanation": "Boolean confidence is invalid.",
        }
    )

    with pytest.raises(ValueError):
        parse_diagnosis(raw)


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_out_of_range_confidence_is_rejected(confidence: float) -> None:
    raw = json.dumps(
        {
            "diagnosis": "NORMAL",
            "confidence": confidence,
            "explanation": "Confidence is outside the permitted range.",
        }
    )

    with pytest.raises(ValueError):
        parse_diagnosis(raw)


def test_non_string_explanation_is_rejected() -> None:
    raw = json.dumps(
        {
            "diagnosis": "NORMAL",
            "confidence": 0.8,
            "explanation": 123,
        }
    )

    with pytest.raises(ValueError):
        parse_diagnosis(raw)


def test_blank_explanation_is_rejected() -> None:
    raw = json.dumps(
        {
            "diagnosis": "NORMAL",
            "confidence": 0.8,
            "explanation": "   ",
        }
    )

    with pytest.raises(ValueError):
        parse_diagnosis(raw)


def test_non_object_json_is_rejected() -> None:
    with pytest.raises(ValueError):
        parse_diagnosis('["NORMAL", 0.8]')