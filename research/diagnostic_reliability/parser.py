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
    if set(payload) != {"diagnosis", "confidence", "explanation"}:
        raise ValueError("Response must contain exactly diagnosis, confidence, explanation")

    diagnosis = DiagnosisLabel(payload["diagnosis"])
    confidence = float(payload["confidence"])
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    explanation = str(payload["explanation"]).strip()
    if not explanation:
        raise ValueError("explanation must be non-empty")
    return ParsedDiagnosis(diagnosis, confidence, explanation)
