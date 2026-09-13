"""Tests for deterministic telemetry representations."""

from __future__ import annotations

from datetime import datetime, timezone

from research.diagnostic_reliability.context import (
    build_model_context,
)
from research.diagnostic_reliability.representation import (
    append_telemetry_summary,
    build_telemetry_summary,
)
from research.diagnostic_reliability.scenarios import (
    generate_scenario,
)
from research.diagnostic_reliability.taxonomy import (
    FaultLabel,
)


def _scenario():
    return generate_scenario(
        FaultLabel.MEMORY_PRESSURE,
        instance=1,
        seed=20260923,
        end_time=datetime(
            2026,
            9,
            12,
            23,
            30,
            tzinfo=timezone.utc,
        ),
        device_id="EXP_DEVICE_01",
    )


def test_summary_contains_all_generic_metrics():
    summary = build_telemetry_summary(
        _scenario()
    )

    expected = (
        "cpu_percent",
        "memory_percent",
        "disk_percent",
        "network_latency_ms",
        "error_rate",
        "active_connections",
        "queue_depth",
    )

    for metric in expected:
        assert metric in summary


def test_summary_contains_only_arithmetic_heading():
    summary = build_telemetry_summary(
        _scenario()
    )

    assert (
        "[DETERMINISTIC TELEMETRY SUMMARY]"
        in summary
    )

    assert (
        "metric,start,end,minimum,maximum,mean,"
        "first_half_mean,second_half_mean"
        in summary
    )


def test_summary_does_not_expose_private_metadata():
    scenario = _scenario()

    summary = build_telemetry_summary(
        scenario
    )

    assert scenario.scenario_id not in summary
    assert scenario.scenario_family not in summary

    assert (
        scenario.fault.value
        not in summary
    )


def test_summary_contains_no_qualitative_diagnostic_terms():
    summary = (
        build_telemetry_summary(
            _scenario()
        ).lower()
    )

    prohibited = (
        "high",
        "low",
        "severe",
        "dominant",
        "abnormal",
        "normal",
        "fault",
        "diagnosis",
        "increasing",
        "decreasing",
        "pressure",
        "saturation",
        "degradation",
        "failure",
    )

    for term in prohibited:
        assert term not in summary


def test_append_preserves_original_raw_context():
    scenario = _scenario()

    raw_context = build_model_context(
        scenario
    )

    extended = (
        append_telemetry_summary(
            raw_context,
            scenario,
        )
    )

    assert extended.startswith(
        raw_context
    )

    assert (
        "[OPERATIONAL TELEMETRY]"
        in extended
    )

    assert (
        "[DETERMINISTIC TELEMETRY SUMMARY]"
        in extended
    )