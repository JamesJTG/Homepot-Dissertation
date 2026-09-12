"""Tests for controlled network degradation scenario families."""

from __future__ import annotations

from datetime import datetime, timezone

from research.diagnostic_reliability.scenarios import generate_scenario
from research.diagnostic_reliability.scenarios.network_degradation import (
    ND1_FAMILY,
    ND2_FAMILY,
    ND3_FAMILY,
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


def _network_persistent(seed: int = 42):
    """Generate the ND-1 persistent latency degradation scenario."""

    return generate_scenario(
        fault=FaultLabel.NETWORK_DEGRADATION,
        scenario_family=ND1_FAMILY,
        seed=seed,
        end_time=FIXED_END,
    )


def _network_intermittent(seed: int = 42):
    """Generate the ND-2 intermittent severe latency scenario."""

    return generate_scenario(
        fault=FaultLabel.NETWORK_DEGRADATION,
        scenario_family=ND2_FAMILY,
        seed=seed,
        end_time=FIXED_END,
    )


def _network_with_errors(seed: int = 42):
    """Generate the ND-3 degradation-with-request-errors scenario."""

    return generate_scenario(
        fault=FaultLabel.NETWORK_DEGRADATION,
        scenario_family=ND3_FAMILY,
        seed=seed,
        end_time=FIXED_END,
    )


# ---------------------------------------------------------------------------
# ND-1: Persistent latency degradation
# ---------------------------------------------------------------------------


def test_nd1_is_reproducible():
    first = _network_persistent(seed=42)
    second = _network_persistent(seed=42)

    assert first == second


def test_nd1_uses_expected_family():
    scenario = _network_persistent()

    assert scenario.scenario_family == ND1_FAMILY
    assert scenario.fault == FaultLabel.NETWORK_DEGRADATION


def test_nd1_begins_in_normal_latency_regime():
    scenario = _network_persistent()

    early_latency = [
        point.network_latency_ms
        for point in scenario.telemetry[:6]
    ]

    assert max(early_latency) < 150.0


def test_nd1_latency_increases_over_time():
    scenario = _network_persistent()

    early_latency = [
        point.network_latency_ms
        for point in scenario.telemetry[:6]
    ]
    late_latency = [
        point.network_latency_ms
        for point in scenario.telemetry[-6:]
    ]

    assert sum(late_latency) / len(late_latency) > (
        sum(early_latency) / len(early_latency)
    )


def test_nd1_reaches_severe_latency():
    scenario = _network_persistent()

    latencies = [
        point.network_latency_ms
        for point in scenario.telemetry
    ]

    assert max(latencies) > 700.0


def test_nd1_late_latency_is_persistently_high():
    scenario = _network_persistent()

    late_latency = [
        point.network_latency_ms
        for point in scenario.telemetry[-6:]
    ]

    assert sum(value > 500.0 for value in late_latency) >= 5


def test_nd1_cpu_does_not_enter_cpu_saturation_regime():
    scenario = _network_persistent()

    cpu = [
        point.cpu_percent
        for point in scenario.telemetry
    ]

    assert max(cpu) < 80.0


def test_nd1_memory_remains_below_memory_pressure_regime():
    scenario = _network_persistent()

    memory = [
        point.memory_percent
        for point in scenario.telemetry
    ]

    assert max(memory) <= 70.2


def test_nd1_error_rate_remains_secondary():
    scenario = _network_persistent()

    error_rates = [
        point.error_rate
        for point in scenario.telemetry
    ]

    assert max(error_rates) <= 0.02


# ---------------------------------------------------------------------------
# ND-2: Intermittent severe latency
# ---------------------------------------------------------------------------


def test_nd2_is_reproducible():
    first = _network_intermittent(seed=42)
    second = _network_intermittent(seed=42)

    assert first == second


def test_nd2_uses_expected_family():
    scenario = _network_intermittent()

    assert scenario.scenario_family == ND2_FAMILY
    assert scenario.fault == FaultLabel.NETWORK_DEGRADATION


def test_nd2_contains_repeated_severe_latency_spikes():
    scenario = _network_intermittent()

    latencies = [
        point.network_latency_ms
        for point in scenario.telemetry
    ]

    severe_points = [
        value
        for value in latencies
        if value > 500.0
    ]

    assert len(severe_points) >= 8


def test_nd2_contains_recovery_periods_between_spikes():
    scenario = _network_intermittent()

    latencies = [
        point.network_latency_ms
        for point in scenario.telemetry
    ]

    severe_indices = [
        idx
        for idx, value in enumerate(latencies)
        if value > 500.0
    ]

    assert severe_indices

    gaps_between_severe_points = [
        severe_indices[idx] - severe_indices[idx - 1]
        for idx in range(1, len(severe_indices))
    ]

    assert any(gap > 1 for gap in gaps_between_severe_points)


def test_nd2_recovers_below_severe_latency_between_episodes():
    scenario = _network_intermittent()

    recovery_points = [
        scenario.telemetry[idx].network_latency_ms
        for idx in (8, 9, 14, 15, 20, 21)
    ]

    assert all(value < 250.0 for value in recovery_points)


def test_nd2_is_not_persistently_severe_at_the_end():
    scenario = _network_intermittent()

    late_latency = [
        point.network_latency_ms
        for point in scenario.telemetry[-6:]
    ]

    assert any(value < 250.0 for value in late_latency)
    assert any(value > 500.0 for value in late_latency)


def test_nd2_cpu_does_not_enter_cpu_saturation_regime():
    scenario = _network_intermittent()

    cpu = [
        point.cpu_percent
        for point in scenario.telemetry
    ]

    assert max(cpu) < 80.0


def test_nd2_memory_remains_below_memory_pressure_regime():
    scenario = _network_intermittent()

    memory = [
        point.memory_percent
        for point in scenario.telemetry
    ]

    assert max(memory) <= 70.2


def test_nd2_error_rate_remains_secondary():
    scenario = _network_intermittent()

    error_rates = [
        point.error_rate
        for point in scenario.telemetry
    ]

    assert max(error_rates) <= 0.02


# ---------------------------------------------------------------------------
# ND-3: Network degradation with request errors
# ---------------------------------------------------------------------------


def test_nd3_is_reproducible():
    first = _network_with_errors(seed=42)
    second = _network_with_errors(seed=42)

    assert first == second


def test_nd3_uses_expected_family():
    scenario = _network_with_errors()

    assert scenario.scenario_family == ND3_FAMILY
    assert scenario.fault == FaultLabel.NETWORK_DEGRADATION


def test_nd3_begins_with_low_error_rate():
    scenario = _network_with_errors()

    early_errors = [
        point.error_rate
        for point in scenario.telemetry[:6]
    ]

    assert max(early_errors) < 0.02


def test_nd3_network_degradation_appears_before_high_request_errors():
    scenario = _network_with_errors()

    first_degraded_latency = next(
        idx
        for idx, point in enumerate(scenario.telemetry)
        if point.network_latency_ms >= 150.0
    )

    first_high_error = next(
        idx
        for idx, point in enumerate(scenario.telemetry)
        if point.error_rate >= 0.03
    )

    assert first_degraded_latency < first_high_error


def test_nd3_reaches_severe_latency():
    scenario = _network_with_errors()

    latencies = [
        point.network_latency_ms
        for point in scenario.telemetry
    ]

    assert max(latencies) > 700.0


def test_nd3_request_error_rate_becomes_elevated():
    scenario = _network_with_errors()

    error_rates = [
        point.error_rate
        for point in scenario.telemetry
    ]

    assert max(error_rates) > 0.05


def test_nd3_late_period_contains_both_high_latency_and_errors():
    scenario = _network_with_errors()

    late_points = scenario.telemetry[-6:]

    overlapping_points = [
        point
        for point in late_points
        if (
            point.network_latency_ms > 500.0
            and point.error_rate > 0.03
        )
    ]

    assert len(overlapping_points) >= 4


def test_nd3_cpu_does_not_enter_cpu_saturation_regime():
    scenario = _network_with_errors()

    cpu = [
        point.cpu_percent
        for point in scenario.telemetry
    ]

    assert max(cpu) < 80.0


def test_nd3_memory_remains_below_memory_pressure_regime():
    scenario = _network_with_errors()

    memory = [
        point.memory_percent
        for point in scenario.telemetry
    ]

    assert max(memory) <= 70.2


# ---------------------------------------------------------------------------
# Cross-family checks
# ---------------------------------------------------------------------------


def test_nd1_and_nd2_have_different_temporal_shapes():
    persistent = _network_persistent()
    intermittent = _network_intermittent()

    persistent_latency = [
        point.network_latency_ms
        for point in persistent.telemetry
    ]
    intermittent_latency = [
        point.network_latency_ms
        for point in intermittent.telemetry
    ]

    persistent_late_mean = sum(persistent_latency[-6:]) / 6
    intermittent_late_mean = sum(intermittent_latency[-6:]) / 6

    assert persistent_late_mean > intermittent_late_mean


def test_nd3_has_more_error_overlap_than_nd1_and_nd2():
    persistent = _network_persistent()
    intermittent = _network_intermittent()
    with_errors = _network_with_errors()

    persistent_max_error = max(
        point.error_rate
        for point in persistent.telemetry
    )
    intermittent_max_error = max(
        point.error_rate
        for point in intermittent.telemetry
    )
    nd3_max_error = max(
        point.error_rate
        for point in with_errors.telemetry
    )

    assert nd3_max_error > persistent_max_error
    assert nd3_max_error > intermittent_max_error
    assert nd3_max_error > 0.05


def test_network_families_share_same_private_fault_label():
    scenarios = [
        _network_persistent(),
        _network_intermittent(),
        _network_with_errors(),
    ]

    assert all(
        scenario.fault == FaultLabel.NETWORK_DEGRADATION
        for scenario in scenarios
    )