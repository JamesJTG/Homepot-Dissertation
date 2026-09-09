"""Deterministic model-visible context construction."""

from __future__ import annotations

from typing import Mapping, Any

from .scenarios import Scenario


def build_model_context(
    scenario: Scenario,
    *,
    validation: Mapping[str, Any] | None = None,
) -> str:
    """Render raw evidence without exposing the private ground-truth label."""
    lines = [
        "[CURRENT SYSTEM STATUS]",
        f"Focus device: {scenario.device_id}",
        "",
        "[OPERATIONAL TELEMETRY]",
        "timestamp,cpu_percent,memory_percent,disk_percent,network_latency_ms,error_rate",
    ]
    for p in scenario.telemetry:
        vals = (
            p.timestamp.isoformat(),
            p.cpu_percent,
            p.memory_percent,
            p.disk_percent,
            p.network_latency_ms,
            p.error_rate,
        )
        lines.append(",".join("NA" if v is None else str(v) for v in vals))

    if validation is not None:
        lines.extend(
            [
                "",
                "[VALIDATION TRUST STATUS]",
                f"Trust mode: {validation.get('trust_mode', 'unknown')}",
                f"Trust score: {validation.get('trust_score', 'unknown')}",
                f"Actionable: {'yes' if validation.get('actionable') else 'no'}",
                f"Failed gate: {validation.get('failed_gate') or 'none'}",
            ]
        )

    return "\n".join(lines)
