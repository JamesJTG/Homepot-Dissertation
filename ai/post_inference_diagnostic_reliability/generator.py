"""Structured repeated diagnostic generation for the reliability layer."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import ollama

from research.diagnostic_reliability.parser import (
    ParsedDiagnosis,
    parse_diagnosis,
)
from research.diagnostic_reliability.prompt import (
    DIAGNOSTIC_PROMPT,
    SYSTEM_PROMPT,
)
from research.diagnostic_reliability.taxonomy import DiagnosisLabel

from ai.llm import LLMService


GENERATION_COUNT = 7
DIAGNOSTIC_TEMPERATURE = 0.4


def _diagnostic_schema() -> dict[str, Any]:
    """Return the structured diagnostic schema used by the frozen experiment."""
    return {
        "type": "object",
        "properties": {
            "diagnosis": {
                "type": "string",
                "enum": [
                    label.value
                    for label in DiagnosisLabel
                ],
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "explanation": {
                "type": "string",
                "minLength": 1,
            },
        },
        "required": [
            "diagnosis",
            "confidence",
            "explanation",
        ],
        "additionalProperties": False,
    }


class DiagnosticGenerationError(RuntimeError):
    """Raised when a valid structured diagnostic generation cannot be obtained."""


class DiagnosticGenerator:
    """Generate repeated structured diagnoses using HOMEPOT's Ollama service.

    The generation settings reproduce the frozen measurement process used
    to construct the downstream uncertainty variables in the dissertation
    experiment.
    """

    def __init__(
        self,
        *,
        llm_service: LLMService,
    ) -> None:
        self.llm_service = llm_service

    def _generate_one(
        self,
        *,
        context: str,
    ) -> ParsedDiagnosis:
        """Generate and strictly parse one structured diagnosis."""
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {DIAGNOSTIC_PROMPT}"
                ),
            },
        ]

        try:
            response = self.llm_service.client.chat(
                model=self.llm_service.model,
                messages=messages,
                format=_diagnostic_schema(),
                options={
                    "temperature": DIAGNOSTIC_TEMPERATURE,
                    "num_ctx": int(
                        self.llm_service.config[
                            "llm"
                        ].get(
                            "context_window",
                            4096,
                        )
                    ),
                },
            )
        except Exception as exc:
            raise DiagnosticGenerationError(
                "Ollama diagnostic generation failed"
            ) from exc

        try:
            content = str(
                response["message"]["content"]
            ).strip()
        except Exception as exc:
            raise DiagnosticGenerationError(
                "Ollama returned an invalid diagnostic response structure"
            ) from exc

        if not content:
            raise DiagnosticGenerationError(
                "Ollama returned an empty diagnostic response"
            )

        try:
            return parse_diagnosis(content)
        except Exception as exc:
            raise DiagnosticGenerationError(
                "Ollama response failed strict diagnostic parsing"
            ) from exc

    def generate_repeated(
        self,
        *,
        context: str,
    ) -> Sequence[ParsedDiagnosis]:
        """Return exactly seven valid structured diagnostic generations."""
        if not context.strip():
            raise ValueError(
                "context must not be empty"
            )

        generations = [
            self._generate_one(
                context=context,
            )
            for _ in range(
                GENERATION_COUNT
            )
        ]

        if len(generations) != GENERATION_COUNT:
            raise DiagnosticGenerationError(
                "Repeated generation did not produce exactly seven diagnoses"
            )

        return tuple(generations)