"""Build leakage-controlled model-visible diagnostic context."""

from __future__ import annotations

from typing import Any, Mapping

from .scenarios import Scenario


def build_model_context(
    scenario: Scenario,
    *,
    validation: Mapping[str, Any] | None = None,
) -> str:
    """Build the model-visible context for one controlled scenario.

    Private experimental metadata such as the injected fault label and
    scenario-family name must not be exposed to the model.
    """

    lines = [
        "[DEVICE]",
        f"Focus device: {scenario.device_id}",
        "",
        "[OPERATIONAL TELEMETRY]",
        (
            "timestamp,cpu_percent,memory_percent,disk_percent,"
            "network_latency_ms,error_rate,active_connections,queue_depth"
        ),
    ]

    for p in scenario.telemetry:
        vals = (
            p.timestamp.isoformat(),
            p.cpu_percent,
            p.memory_percent,
            p.disk_percent,
            p.network_latency_ms,
            p.error_rate,
            p.active_connections,
            p.queue_depth,
        )

        lines.append(
            ",".join(
                "NA" if value is None else str(value)
                for value in vals
            )
        )

    if validation is not None:
        lines.extend(
            [
                "",
                "[VALIDATION STATUS]",
            ]
        )

        for key, value in validation.items():
            lines.append(f"{key}: {value}")

    return "\n".join(lines)