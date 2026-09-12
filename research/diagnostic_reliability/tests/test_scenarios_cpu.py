"""Tests for controlled CPU saturation scenario families."""

from __future__ import annotations

from datetime import datetime, timezone

from research.diagnostic_reliability.scenarios import generate_scenario
from research.diagnostic_reliability.scenarios.cpu_saturation import (
    CS1_FAMILY,
    CS2_FAMILY,
    CS3_FAMILY,
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


def _cpu_gradual(seed: int = 42):
    """Generate the CS-1 gradual processing overload scenario."""

    return generate_scenario(
        fault=FaultLabel.CPU_SATURATION,
        scenario_family=CS1_FAMILY,
        seed=seed,
        end_time=FIXED_END,
    )


def _cpu_sudden(seed: int = 42):
    """Generate the CS-2 sudden sustained spike scenario."""

    return generate_scenario(
        fault=FaultLabel.CPU_SATURATION,
        scenario_family=CS2_FAMILY,
        seed=seed,
        end_time=FIXED_END,
    )


def _cpu_concurrency(seed: int = 42):
    """Generate the CS-3 high-concurrency pressure scenario."""

    return generate_scenario(
        fault=FaultLabel.CPU_SATURATION,
        scenario_family=CS3_FAMILY,
        seed=seed,
        end_time=FIXED_END,
    )


# ---------------------------------------------------------------------------
# CS-1: Gradual processing overload
# ---------------------------------------------------------------------------


def test_cs1_is_reproducible():
    first = _cpu_gradual(seed=42)
    second = _cpu_gradual(seed=42)

    assert first == second


def test_cs1_uses_expected_family():
    scenario = _cpu_gradual()

    assert scenario.scenario_family == CS1_FAMILY
    assert scenario.fault == FaultLabel.CPU_SATURATION


def test_cs1_cpu_rises_over_time():
    scenario = _cpu_gradual()

    early_cpu = [
        point.cpu_percent
        for point in scenario.telemetry[:6]
    ]
    late_cpu = [
        point.cpu_percent
        for point in scenario.telemetry[-6:]
    ]

    assert sum(late_cpu) / len(late_cpu) > sum(early_cpu) / len(early_cpu)


def test_cs1_finishes_in_cpu_saturation_range():
    scenario = _cpu_gradual()

    late_cpu = [
        point.cpu_percent
        for point in scenario.telemetry[-6:]
    ]

    assert max(late_cpu) >= 90.0
    assert sum(value >= 90.0 for value in late_cpu) >= 3


def test_cs1_queue_depth_increases():
    scenario = _cpu_gradual()

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


def test_cs1_network_latency_remains_secondary():
    scenario = _cpu_gradual()

    latencies = [
        point.network_latency_ms
        for point in scenario.telemetry
    ]

    assert max(latencies) < 180.0


def test_cs1_memory_remains_below_memory_pressure_regime():
    scenario = _cpu_gradual()

    memory = [
        point.memory_percent
        for point in scenario.telemetry
    ]

    assert max(memory) <= 70.2


# ---------------------------------------------------------------------------
# CS-2: Sudden sustained workload spike
# ---------------------------------------------------------------------------


def test_cs2_is_reproducible():
    first = _cpu_sudden(seed=42)
    second = _cpu_sudden(seed=42)

    assert first == second


def test_cs2_uses_expected_family():
    scenario = _cpu_sudden()

    assert scenario.scenario_family == CS2_FAMILY
    assert scenario.fault == FaultLabel.CPU_SATURATION


def test_cs2_has_abrupt_cpu_transition():
    scenario = _cpu_sudden()

    cpu = [
        point.cpu_percent
        for point in scenario.telemetry
    ]

    largest_jump = max(
        cpu[idx] - cpu[idx - 1]
        for idx in range(1, len(cpu))
    )

    assert largest_jump > 30.0


def test_cs2_pre_spike_cpu_is_normal():
    scenario = _cpu_sudden()

    early_cpu = [
        point.cpu_percent
        for point in scenario.telemetry[:8]
    ]

    assert max(early_cpu) < 70.0


def test_cs2_post_spike_cpu_is_sustained():
    scenario = _cpu_sudden()

    late_cpu = [
        point.cpu_percent
        for point in scenario.telemetry[-10:]
    ]

    assert min(late_cpu) > 90.0


def test_cs2_queue_increases_after_spike():
    scenario = _cpu_sudden()

    early_queue = [
        point.queue_depth
        for point in scenario.telemetry[:8]
    ]
    late_queue = [
        point.queue_depth
        for point in scenario.telemetry[-8:]
    ]

    assert sum(late_queue) / len(late_queue) > (
        sum(early_queue) / len(early_queue)
    )


def test_cs2_network_latency_remains_secondary():
    scenario = _cpu_sudden()

    latencies = [
        point.network_latency_ms
        for point in scenario.telemetry
    ]

    assert max(latencies) < 180.0


def test_cs2_memory_remains_below_memory_pressure_regime():
    scenario = _cpu_sudden()

    memory = [
        point.memory_percent
        for point in scenario.telemetry
    ]

    assert max(memory) <= 70.2


# ---------------------------------------------------------------------------
# CS-3: High-concurrency CPU pressure
# ---------------------------------------------------------------------------


def test_cs3_is_reproducible():
    first = _cpu_concurrency(seed=42)
    second = _cpu_concurrency(seed=42)

    assert first == second


def test_cs3_uses_expected_family():
    scenario = _cpu_concurrency()

    assert scenario.scenario_family == CS3_FAMILY
    assert scenario.fault == FaultLabel.CPU_SATURATION


def test_cs3_connections_increase_over_time():
    scenario = _cpu_concurrency()

    early_connections = [
        point.active_connections
        for point in scenario.telemetry[:6]
    ]
    late_connections = [
        point.active_connections
        for point in scenario.telemetry[-6:]
    ]

    assert sum(late_connections) / len(late_connections) > (
        sum(early_connections) / len(early_connections)
    )


def test_cs3_concurrency_rises_before_cpu_saturation():
    scenario = _cpu_concurrency()

    first_high_cpu = next(
        idx
        for idx, point in enumerate(scenario.telemetry)
        if point.cpu_percent >= 90.0
    )

    earlier_connections = [
        point.active_connections
        for point in scenario.telemetry[:first_high_cpu]
    ]

    assert max(earlier_connections) >= 35


def test_cs3_does_not_begin_saturated():
    scenario = _cpu_concurrency()

    early_cpu = [
        point.cpu_percent
        for point in scenario.telemetry[:6]
    ]

    assert max(early_cpu) < 70.0


def test_cs3_reaches_late_cpu_saturation():
    scenario = _cpu_concurrency()

    late_cpu = [
        point.cpu_percent
        for point in scenario.telemetry[-6:]
    ]

    assert max(late_cpu) >= 90.0


def test_cs3_queue_depth_increases():
    scenario = _cpu_concurrency()

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


def test_cs3_network_latency_remains_secondary():
    scenario = _cpu_concurrency()

    latencies = [
        point.network_latency_ms
        for point in scenario.telemetry
    ]

    assert max(latencies) < 180.0


def test_cs3_memory_remains_below_memory_pressure_regime():
    scenario = _cpu_concurrency()

    memory = [
        point.memory_percent
        for point in scenario.telemetry
    ]

    assert max(memory) <= 70.2


# ---------------------------------------------------------------------------
# Cross-family checks
# ---------------------------------------------------------------------------


def test_cs1_and_cs2_have_different_temporal_shapes():
    gradual = _cpu_gradual()
    sudden = _cpu_sudden()

    gradual_cpu = [
        point.cpu_percent
        for point in gradual.telemetry
    ]
    sudden_cpu = [
        point.cpu_percent
        for point in sudden.telemetry
    ]

    gradual_largest_jump = max(
        gradual_cpu[idx] - gradual_cpu[idx - 1]
        for idx in range(1, len(gradual_cpu))
    )

    sudden_largest_jump = max(
        sudden_cpu[idx] - sudden_cpu[idx - 1]
        for idx in range(1, len(sudden_cpu))
    )

    assert sudden_largest_jump > gradual_largest_jump


def test_cpu_families_share_same_private_fault_label():
    scenarios = [
        _cpu_gradual(),
        _cpu_sudden(),
        _cpu_concurrency(),
    ]

    assert all(
        scenario.fault == FaultLabel.CPU_SATURATION
        for scenario in scenarios
    )