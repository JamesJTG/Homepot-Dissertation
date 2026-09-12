"""Tests for controlled application failure scenario families."""

from __future__ import annotations

from datetime import datetime, timezone

from research.diagnostic_reliability.scenarios import generate_scenario
from research.diagnostic_reliability.scenarios.application_failure import (
    AF1_FAMILY,
    AF2_FAMILY,
    AF3_FAMILY,
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


def _service_process_failure(seed: int = 42):
    """Generate AF-1 service process failure."""

    return generate_scenario(
        fault=FaultLabel.APPLICATION_ERROR,
        scenario_family=AF1_FAMILY,
        seed=seed,
        end_time=FIXED_END,
    )


def _dependency_failure(seed: int = 42):
    """Generate AF-2 dependency/service failure."""

    return generate_scenario(
        fault=FaultLabel.APPLICATION_ERROR,
        scenario_family=AF2_FAMILY,
        seed=seed,
        end_time=FIXED_END,
    )


def _application_error_burst(seed: int = 42):
    """Generate AF-3 application error burst."""

    return generate_scenario(
        fault=FaultLabel.APPLICATION_ERROR,
        scenario_family=AF3_FAMILY,
        seed=seed,
        end_time=FIXED_END,
    )


# ---------------------------------------------------------------------------
# AF-1: Service process failure
# ---------------------------------------------------------------------------


def test_af1_is_reproducible():
    assert (
        _service_process_failure(seed=42)
        == _service_process_failure(seed=42)
    )


def test_af1_uses_expected_family_and_fault():
    scenario = _service_process_failure()

    assert scenario.scenario_family == AF1_FAMILY
    assert scenario.fault == FaultLabel.APPLICATION_ERROR


def test_af1_begins_with_low_error_rate():
    scenario = _service_process_failure()

    early_errors = [
        point.error_rate
        for point in scenario.telemetry[:6]
    ]

    assert max(early_errors) < 0.02


def test_af1_error_rate_rises_after_failure():
    scenario = _service_process_failure()

    early = [
        point.error_rate
        for point in scenario.telemetry[:6]
    ]

    late = [
        point.error_rate
        for point in scenario.telemetry[-6:]
    ]

    assert sum(late) / len(late) > (
        sum(early) / len(early)
    )


def test_af1_reaches_severe_application_error_rate():
    scenario = _service_process_failure()

    errors = [
        point.error_rate
        for point in scenario.telemetry
    ]

    assert max(errors) > 0.10


def test_af1_connections_decline_after_failure():
    scenario = _service_process_failure()

    early_connections = [
        point.active_connections
        for point in scenario.telemetry[:6]
    ]

    late_connections = [
        point.active_connections
        for point in scenario.telemetry[-6:]
    ]

    assert (
        sum(late_connections) / len(late_connections)
        <
        sum(early_connections) / len(early_connections)
    )


def test_af1_queue_grows_after_failure():
    scenario = _service_process_failure()

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


def test_af1_does_not_enter_cpu_saturation():
    scenario = _service_process_failure()

    assert max(
        point.cpu_percent
        for point in scenario.telemetry
    ) < 80.0


def test_af1_does_not_enter_memory_pressure():
    scenario = _service_process_failure()

    assert max(
        point.memory_percent
        for point in scenario.telemetry
    ) <= 70.2


def test_af1_does_not_enter_severe_network_regime():
    scenario = _service_process_failure()

    assert max(
        point.network_latency_ms
        for point in scenario.telemetry
    ) < 300.0


# ---------------------------------------------------------------------------
# AF-2: Dependency / service failure
# ---------------------------------------------------------------------------


def test_af2_is_reproducible():
    assert (
        _dependency_failure(seed=42)
        == _dependency_failure(seed=42)
    )


def test_af2_uses_expected_family_and_fault():
    scenario = _dependency_failure()

    assert scenario.scenario_family == AF2_FAMILY
    assert scenario.fault == FaultLabel.APPLICATION_ERROR


def test_af2_begins_with_low_error_rate():
    scenario = _dependency_failure()

    early_errors = [
        point.error_rate
        for point in scenario.telemetry[:5]
    ]

    assert max(early_errors) < 0.02


def test_af2_errors_become_persistently_elevated():
    scenario = _dependency_failure()

    late_errors = [
        point.error_rate
        for point in scenario.telemetry[-8:]
    ]

    assert sum(value > 0.05 for value in late_errors) >= 7


def test_af2_reaches_high_application_error_rate():
    scenario = _dependency_failure()

    assert max(
        point.error_rate
        for point in scenario.telemetry
    ) > 0.10


def test_af2_has_moderate_latency_degradation():
    scenario = _dependency_failure()

    late_latency = [
        point.network_latency_ms
        for point in scenario.telemetry[-6:]
    ]

    assert sum(late_latency) / len(late_latency) > 150.0


def test_af2_does_not_reach_severe_network_latency():
    scenario = _dependency_failure()

    assert max(
        point.network_latency_ms
        for point in scenario.telemetry
    ) < 400.0


def test_af2_queue_accumulates():
    scenario = _dependency_failure()

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


def test_af2_does_not_enter_cpu_saturation():
    scenario = _dependency_failure()

    assert max(
        point.cpu_percent
        for point in scenario.telemetry
    ) < 80.0


def test_af2_does_not_enter_memory_pressure():
    scenario = _dependency_failure()

    assert max(
        point.memory_percent
        for point in scenario.telemetry
    ) <= 70.2


# ---------------------------------------------------------------------------
# AF-3: Intermittent application error bursts
# ---------------------------------------------------------------------------


def test_af3_is_reproducible():
    assert (
        _application_error_burst(seed=42)
        == _application_error_burst(seed=42)
    )


def test_af3_uses_expected_family_and_fault():
    scenario = _application_error_burst()

    assert scenario.scenario_family == AF3_FAMILY
    assert scenario.fault == FaultLabel.APPLICATION_ERROR


def test_af3_contains_multiple_high_error_bursts():
    scenario = _application_error_burst()

    errors = [
        point.error_rate
        for point in scenario.telemetry
    ]

    high_indices = [
        idx
        for idx, value in enumerate(errors)
        if value > 0.08
    ]

    assert len(high_indices) >= 8


def test_af3_contains_recovery_periods():
    scenario = _application_error_burst()

    errors = [
        point.error_rate
        for point in scenario.telemetry
    ]

    high_indices = [
        idx
        for idx, value in enumerate(errors)
        if value > 0.08
    ]

    gaps = [
        high_indices[idx] - high_indices[idx - 1]
        for idx in range(1, len(high_indices))
    ]

    assert any(gap > 1 for gap in gaps)


def test_af3_error_rate_recovers_between_bursts():
    scenario = _application_error_burst()

    recovery_indices = (
        8,
        9,
        14,
        15,
        20,
        21,
        22,
        23,
    )

    recovery_errors = [
        scenario.telemetry[idx].error_rate
        for idx in recovery_indices
    ]

    assert all(value < 0.03 for value in recovery_errors)


def test_af3_latency_remains_secondary():
    scenario = _application_error_burst()

    assert max(
        point.network_latency_ms
        for point in scenario.telemetry
    ) < 250.0


def test_af3_does_not_enter_cpu_saturation():
    scenario = _application_error_burst()

    assert max(
        point.cpu_percent
        for point in scenario.telemetry
    ) < 80.0


def test_af3_does_not_enter_memory_pressure():
    scenario = _application_error_burst()

    assert max(
        point.memory_percent
        for point in scenario.telemetry
    ) <= 70.2


# ---------------------------------------------------------------------------
# Cross-family checks
# ---------------------------------------------------------------------------


def test_af1_and_af3_have_different_error_temporal_shapes():
    service_failure = _service_process_failure()
    error_burst = _application_error_burst()

    af1_late_errors = [
        point.error_rate
        for point in service_failure.telemetry[-6:]
    ]

    af3_late_errors = [
        point.error_rate
        for point in error_burst.telemetry[-6:]
    ]

    assert (
        sum(af1_late_errors) / len(af1_late_errors)
        >
        sum(af3_late_errors) / len(af3_late_errors)
    )


def test_af2_has_more_latency_effect_than_af3():
    dependency = _dependency_failure()
    error_burst = _application_error_burst()

    af2_late_latency = [
        point.network_latency_ms
        for point in dependency.telemetry[-6:]
    ]

    af3_late_latency = [
        point.network_latency_ms
        for point in error_burst.telemetry[-6:]
    ]

    assert (
        sum(af2_late_latency) / len(af2_late_latency)
        >
        sum(af3_late_latency) / len(af3_late_latency)
    )


def test_application_families_share_same_private_fault_label():
    scenarios = [
        _service_process_failure(),
        _dependency_failure(),
        _application_error_burst(),
    ]

    assert all(
        scenario.fault == FaultLabel.APPLICATION_ERROR
        for scenario in scenarios
    )