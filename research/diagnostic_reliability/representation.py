"""Alternative leakage-controlled telemetry representations for research.

The production-compatible diagnostic context contains raw sequential telemetry.
This module provides a deterministic arithmetic summary for controlled
representation experiments.

The summary contains no diagnostic labels, thresholds, scenario identifiers,
fault-specific rules, qualitative interpretations, or private ground truth.
"""

from __future__ import annotations

from statistics import mean

from .scenarios import Scenario


SUMMARY_METRICS = (
    "cpu_percent",
    "memory_percent",
    "disk_percent",
    "network_latency_ms",
    "error_rate",
    "active_connections",
    "queue_depth",
)


def _values(
    scenario: Scenario,
    attribute: str,
) -> list[float]:
    """Return non-null numeric observations for one telemetry attribute."""

    result = []

    for point in scenario.telemetry:
        value = getattr(point, attribute)

        if value is not None:
            result.append(float(value))

    return result


def _format_number(
    value: float,
) -> str:
    """Format summary values consistently without qualitative interpretation."""

    return f"{value:.5f}"


def build_telemetry_summary(
    scenario: Scenario,
) -> str:
    """Build a generic arithmetic summary of model-visible telemetry."""

    lines = [
        "[DETERMINISTIC TELEMETRY SUMMARY]",
        (
            "metric,start,end,minimum,maximum,mean,"
            "first_half_mean,second_half_mean"
        ),
    ]

    for attribute in SUMMARY_METRICS:
        values = _values(
            scenario,
            attribute,
        )

        if not values:
            lines.append(
                f"{attribute},NA,NA,NA,NA,NA,NA,NA"
            )
            continue

        midpoint = max(
            1,
            len(values) // 2,
        )

        first_half = values[:midpoint]
        second_half = values[midpoint:]

        if not second_half:
            second_half = first_half

        summary_values = (
            values[0],
            values[-1],
            min(values),
            max(values),
            mean(values),
            mean(first_half),
            mean(second_half),
        )

        lines.append(
            ",".join(
                [
                    attribute,
                    *(
                        _format_number(value)
                        for value in summary_values
                    ),
                ]
            )
        )

    return "\n".join(lines)


def append_telemetry_summary(
    raw_context: str,
    scenario: Scenario,
) -> str:
    """Append the deterministic summary to an existing model context."""

    return (
        raw_context
        + "\n\n"
        + build_telemetry_summary(
            scenario
        )
    )