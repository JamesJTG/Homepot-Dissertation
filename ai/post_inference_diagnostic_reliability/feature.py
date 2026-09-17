"""Application-facing Post-Inference Diagnostic Reliability feature."""

from __future__ import annotations

from ai.gates.envelope import EnvelopeResult
from ai.llm import LLMService

from .generator import DiagnosticGenerator
from .models import DiagnosticReliabilityResult
from .service import assess_diagnostic_reliability


class PostInferenceDiagnosticReliabilityService:
    """Run the complete post-inference diagnostic reliability workflow.

    This service forms the application boundary between HOMEPOT's existing
    upstream Validation Envelope and the dissertation reliability layer.

    Evidence readiness is derived exclusively from
    ``EnvelopeResult.is_actionable``.
    """

    def __init__(
        self,
        *,
        llm_service: LLMService,
    ) -> None:
        self.generator = DiagnosticGenerator(
            llm_service=llm_service,
        )

    def assess(
        self,
        *,
        context: str,
        envelope: EnvelopeResult,
    ) -> DiagnosticReliabilityResult:
        """Generate repeated diagnoses and assess their reliability."""
        if not context.strip():
            raise ValueError(
                "context must not be empty"
            )

        generations = self.generator.generate_repeated(
            context=context,
        )

        return assess_diagnostic_reliability(
            generations=generations,
            evidence_ready=envelope.is_actionable,
        )