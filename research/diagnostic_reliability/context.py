"""Build leakage-controlled model-visible diagnostic context."""

from __future__ import annotations

from typing import Any, Mapping

from .scenarios import Scenario


# Only validation/readiness fields explicitly approved for model exposure.
#
# Private experimental metadata, detailed diagnostic evidence, scenario
# identifiers, injected fault labels, and answer-bearing derived fields must
# never be passed through automatically.
MODEL_VISIBLE_VALIDATION_FIELDS = (
    "trust_mode",
    "trust_score",
    "actionable",
    "failed_gate_id",
)


def _safe_validation_items(
    validation: Mapping[str, Any],
) -> list[tuple[str, Any]]:
    """Return only validation fields approved for model-visible context."""
    return [
        (key, validation[key])
        for key in MODEL_VISIBLE_VALIDATION_FIELDS
        if key in validation
    ]


def build_model_context(
    scenario: Scenario,
    *,
    validation: Mapping[str, Any] | None = None,
) -> str:
    """Build leakage-controlled model-visible context for one scenario.

    Private experimental metadata such as the injected fault label,
    scenario-family name, seed, expected diagnosis, and injection
    configuration must not be exposed to the model.

    Validation information is restricted to an explicit whitelist of
    upstream readiness fields so arbitrary caller-provided metadata cannot
    accidentally leak into the prompt.

    The CURRENT SYSTEM STATUS block is deliberately stable because HOMEPOT
    Gate C requires this production context contract before LLM ingestion.
    """

    lines = [
        "[CURRENT SYSTEM STATUS]",
        f"Focus device: {scenario.device_id}",
        "",
        "[OPERATIONAL TELEMETRY]",
        (
            "timestamp,cpu_percent,memory_percent,disk_percent,"
            "network_latency_ms,error_rate,active_connections,queue_depth"
        ),
    ]

    for point in scenario.telemetry:
        values = (
            point.timestamp.isoformat(),
            point.cpu_percent,
            point.memory_percent,
            point.disk_percent,
            point.network_latency_ms,
            point.error_rate,
            point.active_connections,
            point.queue_depth,
        )

        lines.append(
            ",".join(
                "NA" if value is None else str(value)
                for value in values
            )
        )

    if validation is not None:
        safe_items = _safe_validation_items(validation)

        if safe_items:
            lines.extend(
                [
                    "",
                    "[VALIDATION STATUS]",
                ]
            )

            for key, value in safe_items:
                lines.append(f"{key}: {value}")

    return "\n".join(lines)