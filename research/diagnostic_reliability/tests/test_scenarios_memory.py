"""Tests for controlled memory pressure scenario families."""

from __future__ import annotations

from datetime import datetime, timezone

from research.diagnostic_reliability.scenarios import generate_scenario
from research.diagnostic_reliability.scenarios.memory_pressure import (
    MP1_FAMILY,
    MP2_FAMILY,
    MP3_FAMILY,
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


def _memory_gradual(seed: int = 42):
    """Generate MP-1 gradual memory accumulation."""

    return generate_scenario(
        fault=FaultLabel.MEMORY_PRESSURE,
        scenario_family=MP1_FAMILY,
        seed=seed,
        end_time=FIXED_END,
    )


def _memory_sustained(seed: int = 42):
    """Generate MP-2 sustained high memory pressure."""

    return generate_scenario(
        fault=FaultLabel.MEMORY_PRESSURE,
        scenario_family=MP2_FAMILY,
        seed=seed,
        end_time=FIXED_END,
    )


def _memory_queue(seed: int = 42):
    """Generate MP-3 memory pressure with queue growth."""

    return generate_scenario(
        fault=FaultLabel.MEMORY_PRESSURE,
        scenario_family=MP3_FAMILY,
        seed=seed,
        end_time=FIXED_END,
    )


# ---------------------------------------------------------------------------
# MP-1: Gradual memory accumulation
# ---------------------------------------------------------------------------


def test_mp1_is_reproducible():
    assert _memory_gradual(seed=42) == _memory_gradual(seed=42)


def test_mp1_uses_expected_family_and_fault():
    scenario = _memory_gradual()

    assert scenario.scenario_family == MP1_FAMILY
    assert scenario.fault == FaultLabel.MEMORY_PRESSURE


def test_mp1_begins_below_memory_pressure_regime():
    scenario = _memory_gradual()

    early_memory = [
        point.memory_percent
        for point in scenario.telemetry[:5]
    ]

    assert max(early_memory) < 70.0


def test_mp1_memory_increases_over_time():
    scenario = _memory_gradual()

    early = [
        point.memory_percent
        for point in scenario.telemetry[:6]
    ]

    late = [
        point.memory_percent
        for point in scenario.telemetry[-6:]
    ]

    assert sum(late) / len(late) > sum(early) / len(early)


def test_mp1_reaches_severe_memory_pressure():
    scenario = _memory_gradual()

    memory = [
        point.memory_percent
        for point in scenario.telemetry
    ]

    assert max(memory) > 90.0


def test_mp1_late_memory_is_persistently_high():
    scenario = _memory_gradual()

    late = [
        point.memory_percent
        for point in scenario.telemetry[-6:]
    ]

    assert sum(value > 80.0 for value in late) >= 5


def test_mp1_cpu_remains_below_cpu_saturation():
    scenario = _memory_gradual()

    cpu = [
        point.cpu_percent
        for point in scenario.telemetry
    ]

    assert max(cpu) < 80.0


def test_mp1_latency_remains_below_network_degradation_regime():
    scenario = _memory_gradual()

    latency = [
        point.network_latency_ms
        for point in scenario.telemetry
    ]

    assert max(latency) < 300.0


def test_mp1_error_rate_remains_secondary():
    scenario = _memory_gradual()

    errors = [
        point.error_rate
        for point in scenario.telemetry
    ]

    assert max(errors) < 0.03


# ---------------------------------------------------------------------------
# MP-2: Sustained high memory pressure
# ---------------------------------------------------------------------------


def test_mp2_is_reproducible():
    assert _memory_sustained(seed=42) == _memory_sustained(seed=42)


def test_mp2_uses_expected_family_and_fault():
    scenario = _memory_sustained()

    assert scenario.scenario_family == MP2_FAMILY
    assert scenario.fault == FaultLabel.MEMORY_PRESSURE


def test_mp2_begins_in_high_memory_regime():
    scenario = _memory_sustained()

    early = [
        point.memory_percent
        for point in scenario.telemetry[:6]
    ]

    assert min(early) > 80.0


def test_mp2_memory_remains_persistently_high():
    scenario = _memory_sustained()

    memory = [
        point.memory_percent
        for point in scenario.telemetry
    ]

    assert sum(value > 80.0 for value in memory) >= 22


def test_mp2_does_not_require_gradual_accumulation():
    scenario = _memory_sustained()

    early = [
        point.memory_percent
        for point in scenario.telemetry[:6]
    ]

    late = [
        point.memory_percent
        for point in scenario.telemetry[-6:]
    ]

    early_mean = sum(early) / len(early)
    late_mean = sum(late) / len(late)

    assert abs(late_mean - early_mean) < 8.0


def test_mp2_cpu_remains_below_cpu_saturation():
    scenario = _memory_sustained()

    cpu = [
        point.cpu_percent
        for point in scenario.telemetry
    ]

    assert max(cpu) < 80.0


def test_mp2_latency_remains_below_network_degradation_regime():
    scenario = _memory_sustained()

    latency = [
        point.network_latency_ms
        for point in scenario.telemetry
    ]

    assert max(latency) < 300.0


def test_mp2_error_rate_remains_secondary():
    scenario = _memory_sustained()

    errors = [
        point.error_rate
        for point in scenario.telemetry
    ]

    assert max(errors) < 0.03


# ---------------------------------------------------------------------------
# MP-3: Memory pressure with queue growth
# ---------------------------------------------------------------------------


def test_mp3_is_reproducible():
    assert _memory_queue(seed=42) == _memory_queue(seed=42)


def test_mp3_uses_expected_family_and_fault():
    scenario = _memory_queue()

    assert scenario.scenario_family == MP3_FAMILY
    assert scenario.fault == FaultLabel.MEMORY_PRESSURE


def test_mp3_memory_reaches_pressure_regime():
    scenario = _memory_queue()

    memory = [
        point.memory_percent
        for point in scenario.telemetry
    ]

    assert max(memory) > 85.0


def test_mp3_queue_grows_substantially():
    scenario = _memory_queue()

    early_queue = [
        point.queue_depth
        for point in scenario.telemetry[:6]
    ]

    late_queue = [
        point.queue_depth
        for point in scenario.telemetry[-6:]
    ]

    assert sum(late_queue) / len(late_queue) > (
        sum(early_queue) / len(early_queue)
    )


def test_mp3_reaches_high_queue_depth():
    scenario = _memory_queue()

    queues = [
        point.queue_depth
        for point in scenario.telemetry
    ]

    assert max(queues) >= 25


def test_mp3_memory_and_queue_are_jointly_elevated_late():
    scenario = _memory_queue()

    late_points = scenario.telemetry[-6:]

    joint_points = [
        point
        for point in late_points
        if (
            point.memory_percent > 80.0
            and point.queue_depth >= 20
        )
    ]

    assert len(joint_points) >= 4


def test_mp3_cpu_remains_below_cpu_saturation():
    scenario = _memory_queue()

    cpu = [
        point.cpu_percent
        for point in scenario.telemetry
    ]

    assert max(cpu) < 80.0


def test_mp3_latency_remains_below_network_degradation_regime():
    scenario = _memory_queue()

    latency = [
        point.network_latency_ms
        for point in scenario.telemetry
    ]

    assert max(latency) < 350.0


def test_mp3_error_rate_does_not_dominate():
    scenario = _memory_queue()

    errors = [
        point.error_rate
        for point in scenario.telemetry
    ]

    assert max(errors) <= 0.03


# ---------------------------------------------------------------------------
# Cross-family checks
# ---------------------------------------------------------------------------


def test_mp1_and_mp2_have_different_temporal_shapes():
    gradual = _memory_gradual()
    sustained = _memory_sustained()

    gradual_early = sum(
        point.memory_percent
        for point in gradual.telemetry[:6]
    ) / 6

    sustained_early = sum(
        point.memory_percent
        for point in sustained.telemetry[:6]
    ) / 6

    assert sustained_early > gradual_early + 20.0


def test_mp3_has_more_queue_growth_than_mp1_and_mp2():
    gradual = _memory_gradual()
    sustained = _memory_sustained()
    queue_growth = _memory_queue()

    gradual_max_queue = max(
        point.queue_depth
        for point in gradual.telemetry
    )

    sustained_max_queue = max(
        point.queue_depth
        for point in sustained.telemetry
    )

    mp3_max_queue = max(
        point.queue_depth
        for point in queue_growth.telemetry
    )

    assert mp3_max_queue > gradual_max_queue
    assert mp3_max_queue > sustained_max_queue


def test_memory_families_share_same_private_fault_label():
    scenarios = [
        _memory_gradual(),
        _memory_sustained(),
        _memory_queue(),
    ]

    assert all(
        scenario.fault == FaultLabel.MEMORY_PRESSURE
        for scenario in scenarios
    )