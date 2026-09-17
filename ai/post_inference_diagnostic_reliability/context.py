"""Production model-visible context for diagnostic reliability assessment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from homepot.app.models.AnalyticsModel import DeviceMetrics


TELEMETRY_POINT_LIMIT = 24

MODEL_VISIBLE_VALIDATION_FIELDS = (
    "trust_mode",
    "trust_score",
    "actionable",
    "failed_gate_id",
)


def model_visible_validation(
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    """Restrict validation information to the experimentally approved fields."""
    return {
        key: validation[key]
        for key in MODEL_VISIBLE_VALIDATION_FIELDS
        if key in validation
    }


def build_diagnostic_context(
    *,
    device_id: str,
    telemetry: Sequence[DeviceMetrics],
    validation: Mapping[str, Any] | None = None,
) -> str:
    """Build the production diagnostic context using the frozen evidence format."""
    if not device_id.strip():
        raise ValueError("device_id must not be empty")

    if not telemetry:
        raise ValueError(
            "At least one telemetry observation is required"
        )

    lines = [
        "[CURRENT SYSTEM STATUS]",
        f"Focus device: {device_id}",
        "",
        "[OPERATIONAL TELEMETRY]",
        (
            "timestamp,cpu_percent,memory_percent,disk_percent,"
            "network_latency_ms,error_rate,active_connections,queue_depth"
        ),
    ]

    ordered = sorted(
        telemetry,
        key=lambda row: row.timestamp,
    )

    for point in ordered:
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
        safe_validation = model_visible_validation(
            validation
        )

        if safe_validation:
            lines.extend(
                [
                    "",
                    "[VALIDATION STATUS]",
                ]
            )

            for key in MODEL_VISIBLE_VALIDATION_FIELDS:
                if key in safe_validation:
                    lines.append(
                        f"{key}: {safe_validation[key]}"
                    )

    return "\n".join(lines)