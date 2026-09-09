"""Small experiment-runner primitives; live Ollama execution is attached in M6.2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .parser import ParsedDiagnosis, parse_diagnosis
from .prompt import DIAGNOSTIC_PROMPT, SYSTEM_PROMPT


@dataclass(frozen=True)
class GenerationResult:
    generation_id: int
    raw_response: str
    parsed: ParsedDiagnosis


def run_repeated_generations(
    generate: Callable[[str, str, str], str],
    *,
    context: str,
    n_generations: int,
) -> list[GenerationResult]:
    """Run an injectable LLM callable repeatedly against one frozen context.

    ``generate`` receives (prompt, context, system_prompt). Injection keeps the
    core runner unit-testable without requiring a live Ollama service.
    """
    if n_generations < 1:
        raise ValueError("n_generations must be at least 1")

    results: list[GenerationResult] = []
    for generation_id in range(1, n_generations + 1):
        raw = generate(DIAGNOSTIC_PROMPT, context, SYSTEM_PROMPT)
        results.append(
            GenerationResult(generation_id, raw, parse_diagnosis(raw))
        )
    return results
