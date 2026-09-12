"""Strict parsing for structured experimental LLM outputs."""

from __future__ import annotations

from dataclasses import dataclass
import json

from .taxonomy import DiagnosisLabel


@dataclass(frozen=True)
class ParsedDiagnosis:
    diagnosis: DiagnosisLabel
    confidence: float
    explanation: str


def parse_diagnosis(text: str) -> ParsedDiagnosis:
    """Parse and validate the exact experimental JSON response."""
    payload = json.loads(text)

    if not isinstance(payload, dict):
        raise ValueError("Response must be a JSON object")

    if set(payload) != {"diagnosis", "confidence", "explanation"}:
        raise ValueError(
            "Response must contain exactly diagnosis, confidence, explanation"
        )

    if not isinstance(payload["diagnosis"], str):
        raise ValueError("diagnosis must be a string")

    diagnosis = DiagnosisLabel(payload["diagnosis"])

    confidence_raw = payload["confidence"]
    if isinstance(confidence_raw, bool) or not isinstance(
        confidence_raw, (int, float)
    ):
        raise ValueError("confidence must be a number")

    confidence = float(confidence_raw)

    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")

    explanation_raw = payload["explanation"]
    if not isinstance(explanation_raw, str):
        raise ValueError("explanation must be a string")

    explanation = explanation_raw.strip()
    if not explanation:
        raise ValueError("explanation must be non-empty")

    return ParsedDiagnosis(
        diagnosis=diagnosis,
        confidence=confidence,
        explanation=explanation,
    )