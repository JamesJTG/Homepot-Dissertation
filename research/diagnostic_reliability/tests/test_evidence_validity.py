"""Experimental-validity tests for evidence conditions and model context.

These tests protect the boundary between private experimental ground truth and
the evidence that is permitted to reach the diagnostic model.

They also verify that evidence-quality transformations operate consistently
across every controlled scenario family without changing the underlying fault.
"""

from datetime import datetime, timezone

import pytest

from research.diagnostic_reliability.context import build_model_context
from research.diagnostic_reliability.evidence_conditions import (
    apply_evidence_condition,
    expected_sufficiency,
)
from research.diagnostic_reliability.scenarios import (
    AF1_FAMILY,
    AF2_FAMILY,
    AF3_FAMILY,
    CS1_FAMILY,
    CS2_FAMILY,
    CS3_FAMILY,
    MP1_FAMILY,
    MP2_FAMILY,
    MP3_FAMILY,
    N1_FAMILY,
    N2_FAMILY,
    N3_FAMILY,
    ND1_FAMILY,
    ND2_FAMILY,
    ND3_FAMILY,
    generate_scenario,
)
from research.diagnostic_reliability.taxonomy import (
    EvidenceCondition,
    EvidenceSufficiency,
    FaultLabel,
)


FIXED_END_TIME = datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)


SCENARIO_FAMILIES = [
    (FaultLabel.CPU_SATURATION, CS1_FAMILY),
    (FaultLabel.CPU_SATURATION, CS2_FAMILY),
    (FaultLabel.CPU_SATURATION, CS3_FAMILY),
    (FaultLabel.NETWORK_DEGRADATION, ND1_FAMILY),
    (FaultLabel.NETWORK_DEGRADATION, ND2_FAMILY),
    (FaultLabel.NETWORK_DEGRADATION, ND3_FAMILY),
    (FaultLabel.MEMORY_PRESSURE, MP1_FAMILY),
    (FaultLabel.MEMORY_PRESSURE, MP2_FAMILY),
    (FaultLabel.MEMORY_PRESSURE, MP3_FAMILY),
    (FaultLabel.APPLICATION_FAILURE, AF1_FAMILY),
    (FaultLabel.APPLICATION_FAILURE, AF2_FAMILY),
    (FaultLabel.APPLICATION_FAILURE, AF3_FAMILY),
    (FaultLabel.NORMAL, N1_FAMILY),
    (FaultLabel.NORMAL, N2_FAMILY),
    (FaultLabel.NORMAL, N3_FAMILY),
]


ALL_CONDITIONS = [
    EvidenceCondition.COMPLETE_FRESH,
    EvidenceCondition.STALE,
    EvidenceCondition.INCOMPLETE,
    EvidenceCondition.DISCONTINUOUS,
]


def _scenario(fault: FaultLabel, family: str):
    return generate_scenario(
        fault,
        scenario_family=family,
        instance=1,
        seed=42,
        end_time=FIXED_END_TIME,
    )


@pytest.mark.parametrize("fault,family", SCENARIO_FAMILIES)
@pytest.mark.parametrize("condition", ALL_CONDITIONS)
def test_evidence_condition_preserves_private_ground_truth(
    fault,
    family,
    condition,
):
    original = _scenario(fault, family)
    transformed = apply_evidence_condition(original, condition)

    assert transformed.fault == original.fault
    assert transformed.scenario_family == original.scenario_family
    assert transformed.device_id == original.device_id
    assert transformed.seed == original.seed


@pytest.mark.parametrize("fault,family", SCENARIO_FAMILIES)
def test_complete_fresh_does_not_modify_scenario(fault, family):
    original = _scenario(fault, family)
    transformed = apply_evidence_condition(
        original,
        EvidenceCondition.COMPLETE_FRESH,
    )

    assert transformed == original


@pytest.mark.parametrize("fault,family", SCENARIO_FAMILIES)
def test_stale_changes_only_timestamps(fault, family):
    original = _scenario(fault, family)
    transformed = apply_evidence_condition(
        original,
        EvidenceCondition.STALE,
    )

    assert len(transformed.telemetry) == len(original.telemetry)

    for before, after in zip(original.telemetry, transformed.telemetry):
        assert after.timestamp < before.timestamp

        assert after.cpu_percent == before.cpu_percent
        assert after.memory_percent == before.memory_percent
        assert after.disk_percent == before.disk_percent
        assert after.network_latency_ms == before.network_latency_ms
        assert after.error_rate == before.error_rate
        assert after.active_connections == before.active_connections
        assert after.queue_depth == before.queue_depth


@pytest.mark.parametrize("fault,family", SCENARIO_FAMILIES)
def test_incomplete_changes_only_completeness_field(fault, family):
    original = _scenario(fault, family)
    transformed = apply_evidence_condition(
        original,
        EvidenceCondition.INCOMPLETE,
    )

    assert len(transformed.telemetry) == len(original.telemetry)
    assert any(
        point.disk_percent is None
        for point in transformed.telemetry
    )

    for before, after in zip(original.telemetry, transformed.telemetry):
        assert after.timestamp == before.timestamp
        assert after.cpu_percent == before.cpu_percent
        assert after.memory_percent == before.memory_percent
        assert after.network_latency_ms == before.network_latency_ms
        assert after.error_rate == before.error_rate
        assert after.active_connections == before.active_connections
        assert after.queue_depth == before.queue_depth


@pytest.mark.parametrize("fault,family", SCENARIO_FAMILIES)
def test_discontinuous_changes_only_timestamps(fault, family):
    original = _scenario(fault, family)
    transformed = apply_evidence_condition(
        original,
        EvidenceCondition.DISCONTINUOUS,
    )

    assert len(transformed.telemetry) == len(original.telemetry)

    changed_timestamp_count = 0

    for before, after in zip(original.telemetry, transformed.telemetry):
        if after.timestamp != before.timestamp:
            changed_timestamp_count += 1

        assert after.cpu_percent == before.cpu_percent
        assert after.memory_percent == before.memory_percent
        assert after.disk_percent == before.disk_percent
        assert after.network_latency_ms == before.network_latency_ms
        assert after.error_rate == before.error_rate
        assert after.active_connections == before.active_connections
        assert after.queue_depth == before.queue_depth

    assert changed_timestamp_count > 0


@pytest.mark.parametrize("fault,family", SCENARIO_FAMILIES)
@pytest.mark.parametrize("condition", ALL_CONDITIONS)
def test_model_context_does_not_expose_private_scenario_metadata(
    fault,
    family,
    condition,
):
    original = _scenario(fault, family)
    transformed = apply_evidence_condition(original, condition)

    context = build_model_context(transformed)

    assert transformed.scenario_id not in context
    assert transformed.scenario_family not in context

    context_lower = context.lower()

    assert "scenario_id" not in context_lower
    assert "scenario_family" not in context_lower
    assert "scenario id" not in context_lower
    assert "scenario family" not in context_lower
    assert "seed:" not in context_lower
    assert "seed=" not in context_lower


def test_model_context_does_not_add_fault_metadata():
    scenario = _scenario(
        FaultLabel.CPU_SATURATION,
        CS1_FAMILY,
    )

    context = build_model_context(scenario)

    assert scenario.fault.value not in context
    assert "fault:" not in context.lower()
    assert "ground truth" not in context.lower()
    assert "expected diagnosis" not in context.lower()
    assert "injected fault" not in context.lower()


def test_expected_sufficiency_contract_for_all_conditions():
    assert (
        expected_sufficiency(EvidenceCondition.COMPLETE_FRESH)
        is EvidenceSufficiency.SUFFICIENT
    )

    for condition in (
        EvidenceCondition.STALE,
        EvidenceCondition.INCOMPLETE,
        EvidenceCondition.DISCONTINUOUS,
    ):
        assert (
            expected_sufficiency(condition)
            is EvidenceSufficiency.INSUFFICIENT
        )