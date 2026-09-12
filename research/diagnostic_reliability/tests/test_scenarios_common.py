"""Tests for shared controlled-scenario behaviour."""

from __future__ import annotations

from datetime import datetime, timezone

from research.diagnostic_reliability.context import build_model_context
from research.diagnostic_reliability.scenarios import generate_scenario
from research.diagnostic_reliability.scenarios.cpu_saturation import (
    CS1_FAMILY,
)
from research.diagnostic_reliability.taxonomy import FaultLabel


FIXED_END = datetime(
    2026,
    9,
    11,
    12,
    0,
    0,
    tzinfo=timezone.utc,
)


def _reference_scenario():
    """Return a deterministic scenario for shared-behaviour tests."""

    return generate_scenario(
        fault=FaultLabel.CPU_SATURATION,
        scenario_family=CS1_FAMILY,
        seed=42,
        end_time=FIXED_END,
    )


def test_default_sampling_interval_is_2_3_seconds():
    """Default controlled scenarios should preserve the chosen 2.3 s cadence."""

    scenario = _reference_scenario()

    intervals = [
        (
            scenario.telemetry[idx].timestamp
            - scenario.telemetry[idx - 1].timestamp
        ).total_seconds()
        for idx in range(1, len(scenario.telemetry))
    ]

    assert all(
        interval == 2.3
        for interval in intervals
    )


def test_private_ground_truth_does_not_leak_into_model_context():
    """Private labels and scenario-family identifiers must remain hidden."""

    scenario = _reference_scenario()

    model_context = build_model_context(scenario)

    assert FaultLabel.CPU_SATURATION.value not in model_context
    assert CS1_FAMILY not in model_context

    assert "active_connections" in model_context
    assert "queue_depth" in model_context