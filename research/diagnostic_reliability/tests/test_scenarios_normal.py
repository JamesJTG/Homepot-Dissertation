"""Tests for controlled normal-operation scenario families."""

from __future__ import annotations

from datetime import datetime, timezone

from research.diagnostic_reliability.scenarios import generate_scenario
from research.diagnostic_reliability.scenarios.normal import (
    N1_FAMILY,
    N2_FAMILY,
    N3_FAMILY,
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


def _stable_normal(seed: int = 42):
    """Generate N-1 stable normal operation."""

    return generate_scenario(
        fault=FaultLabel.NORMAL,
        scenario_family=N1_FAMILY,
        seed=seed,
        end_time=FIXED_END,
    )


def _high_valid_workload(seed: int = 42):
    """Generate N-2 high-but-valid workload."""

    return generate_scenario(
        fault=FaultLabel.NORMAL,
        scenario_family=N2_FAMILY,
        seed=seed,
        end_time=FIXED_END,
    )


def _benign_transient(seed: int = 42):
    """Generate N-3 brief benign transient."""

    return generate_scenario(
        fault=FaultLabel.NORMAL,
        scenario_family=N3_FAMILY,
        seed=seed,
        end_time=FIXED_END,
    )


# ---------------------------------------------------------------------------
# N-1: Stable normal operation
# ---------------------------------------------------------------------------


def test_n1_is_reproducible():
    assert _stable_normal(seed=42) == _stable_normal(seed=42)


def test_n1_uses_expected_family_and_fault():
    scenario = _stable_normal()

    assert scenario.scenario_family == N1_FAMILY
    assert scenario.fault == FaultLabel.NORMAL


def test_n1_cpu_remains_normal():
    scenario = _stable_normal()

    assert max(
        point.cpu_percent
        for point in scenario.telemetry
    ) < 65.0


def test_n1_memory_remains_normal():
    scenario = _stable_normal()

    assert max(
        point.memory_percent
        for point in scenario.telemetry
    ) < 65.0


def test_n1_latency_remains_normal():
    scenario = _stable_normal()

    assert max(
        point.network_latency_ms
        for point in scenario.telemetry
    ) < 150.0


def test_n1_error_rate_remains_low():
    scenario = _stable_normal()

    assert max(
        point.error_rate
        for point in scenario.telemetry
    ) < 0.02


def test_n1_queue_remains_low():
    scenario = _stable_normal()

    assert max(
        point.queue_depth
        for point in scenario.telemetry
    ) <= 10


# ---------------------------------------------------------------------------
# N-2: High-but-valid workload
# ---------------------------------------------------------------------------


def test_n2_is_reproducible():
    assert (
        _high_valid_workload(seed=42)
        == _high_valid_workload(seed=42)
    )


def test_n2_uses_expected_family_and_fault():
    scenario = _high_valid_workload()

    assert scenario.scenario_family == N2_FAMILY
    assert scenario.fault == FaultLabel.NORMAL


def test_n2_has_higher_cpu_than_n1():
    normal = _stable_normal()
    high_workload = _high_valid_workload()

    normal_cpu = [
        point.cpu_percent
        for point in normal.telemetry
    ]

    high_cpu = [
        point.cpu_percent
        for point in high_workload.telemetry
    ]

    assert (
        sum(high_cpu) / len(high_cpu)
        >
        sum(normal_cpu) / len(normal_cpu)
    )


def test_n2_cpu_is_elevated_but_below_saturation():
    scenario = _high_valid_workload()

    cpu = [
        point.cpu_percent
        for point in scenario.telemetry
    ]

    assert sum(cpu) / len(cpu) > 65.0
    assert max(cpu) < 80.0


def test_n2_memory_is_high_but_stable():
    scenario = _high_valid_workload()

    memory = [
        point.memory_percent
        for point in scenario.telemetry
    ]

    early = memory[:6]
    late = memory[-6:]

    assert sum(memory) / len(memory) > 75.0
    assert max(memory) > 80.0
    assert max(memory) < 90.0

    early_mean = sum(early) / len(early)
    late_mean = sum(late) / len(late)

    assert late_mean - early_mean < 10.0


def test_n2_has_higher_connection_load_than_n1():
    normal = _stable_normal()
    high_workload = _high_valid_workload()

    normal_connections = [
        point.active_connections
        for point in normal.telemetry
    ]

    high_connections = [
        point.active_connections
        for point in high_workload.telemetry
    ]

    assert (
        sum(high_connections) / len(high_connections)
        >
        sum(normal_connections) / len(normal_connections)
    )


def test_n2_latency_is_elevated_but_not_network_failure():
    scenario = _high_valid_workload()

    latency = [
        point.network_latency_ms
        for point in scenario.telemetry
    ]

    assert sum(latency) / len(latency) > 90.0
    assert max(latency) < 250.0


def test_n2_error_rate_remains_low():
    scenario = _high_valid_workload()

    assert max(
        point.error_rate
        for point in scenario.telemetry
    ) < 0.03


def test_n2_remains_labelled_normal():
    scenario = _high_valid_workload()

    assert scenario.fault == FaultLabel.NORMAL


# ---------------------------------------------------------------------------
# N-3: Brief benign transient
# ---------------------------------------------------------------------------


def test_n3_is_reproducible():
    assert _benign_transient(seed=42) == _benign_transient(seed=42)


def test_n3_uses_expected_family_and_fault():
    scenario = _benign_transient()

    assert scenario.scenario_family == N3_FAMILY
    assert scenario.fault == FaultLabel.NORMAL


def test_n3_contains_brief_high_cpu_observations():
    scenario = _benign_transient()

    high_cpu = [
        point.cpu_percent
        for point in scenario.telemetry
        if point.cpu_percent > 80.0
    ]

    assert len(high_cpu) >= 2


def test_n3_can_cross_severe_cpu_threshold_without_becoming_fault():
    scenario = _benign_transient(seed=42)

    assert max(
        point.cpu_percent
        for point in scenario.telemetry
    ) > 90.0

    assert scenario.fault == FaultLabel.NORMAL


def test_n3_high_cpu_is_not_persistent():
    scenario = _benign_transient()

    high_cpu = [
        point
        for point in scenario.telemetry
        if point.cpu_percent > 80.0
    ]

    assert len(high_cpu) <= 4


def test_n3_contains_temporary_latency_elevation():
    scenario = _benign_transient()

    elevated_latency = [
        point.network_latency_ms
        for point in scenario.telemetry
        if point.network_latency_ms > 250.0
    ]

    assert len(elevated_latency) >= 2


def test_n3_severe_latency_is_not_persistent():
    scenario = _benign_transient()

    elevated_latency = [
        point
        for point in scenario.telemetry
        if point.network_latency_ms > 250.0
    ]

    assert len(elevated_latency) <= 4


def test_n3_error_rate_is_temporarily_elevated():
    scenario = _benign_transient()

    elevated_errors = [
        point.error_rate
        for point in scenario.telemetry
        if point.error_rate > 0.02
    ]

    assert len(elevated_errors) >= 2


def test_n3_does_not_reach_application_failure_error_levels():
    scenario = _benign_transient()

    assert max(
        point.error_rate
        for point in scenario.telemetry
    ) < 0.05


def test_n3_recovers_after_transient():
    scenario = _benign_transient()

    late = scenario.telemetry[-5:]

    assert max(
        point.cpu_percent
        for point in late
    ) < 60.0

    assert max(
        point.network_latency_ms
        for point in late
    ) < 130.0

    assert max(
        point.error_rate
        for point in late
    ) < 0.02


def test_n3_does_not_enter_memory_pressure():
    scenario = _benign_transient()

    assert max(
        point.memory_percent
        for point in scenario.telemetry
    ) < 75.0


def test_n3_remains_labelled_normal_despite_transient():
    scenario = _benign_transient()

    assert scenario.fault == FaultLabel.NORMAL


# ---------------------------------------------------------------------------
# Cross-family checks
# ---------------------------------------------------------------------------


def test_n1_and_n2_have_different_workload_levels():
    stable = _stable_normal()
    high_workload = _high_valid_workload()

    stable_cpu = sum(
        point.cpu_percent
        for point in stable.telemetry
    ) / len(stable.telemetry)

    high_cpu = sum(
        point.cpu_percent
        for point in high_workload.telemetry
    ) / len(high_workload.telemetry)

    assert high_cpu > stable_cpu + 15.0


def test_n3_has_higher_peak_cpu_than_n2():
    high_workload = _high_valid_workload()
    transient = _benign_transient()

    assert max(
        point.cpu_percent
        for point in transient.telemetry
    ) > max(
        point.cpu_percent
        for point in high_workload.telemetry
    )


def test_n3_has_higher_peak_latency_than_n2():
    high_workload = _high_valid_workload()
    transient = _benign_transient()

    assert max(
        point.network_latency_ms
        for point in transient.telemetry
    ) > max(
        point.network_latency_ms
        for point in high_workload.telemetry
    )


def test_all_normal_families_share_normal_ground_truth():
    scenarios = [
        _stable_normal(),
        _high_valid_workload(),
        _benign_transient(),
    ]

    assert all(
        scenario.fault == FaultLabel.NORMAL
        for scenario in scenarios
    )