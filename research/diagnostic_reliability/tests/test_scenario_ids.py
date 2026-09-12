"""Tests for controlled scenario identifier construction."""

from __future__ import annotations

import pytest

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
from research.diagnostic_reliability.scenarios.common import build_scenario_id
from research.diagnostic_reliability.taxonomy import FaultLabel


def test_build_scenario_id_contains_fault_family_and_instance():
    scenario_id = build_scenario_id(
        fault=FaultLabel.CPU_SATURATION,
        scenario_family=CS2_FAMILY,
        instance=7,
    )

    assert scenario_id == "CPU_SATURATION_CS-2_007"


def test_different_families_with_same_instance_have_unique_ids():
    first = generate_scenario(
        fault=FaultLabel.CPU_SATURATION,
        scenario_family=CS1_FAMILY,
        instance=1,
        seed=42,
    )

    second = generate_scenario(
        fault=FaultLabel.CPU_SATURATION,
        scenario_family=CS2_FAMILY,
        instance=1,
        seed=42,
    )

    assert first.scenario_id != second.scenario_id


def test_same_family_different_instances_have_unique_ids():
    first = generate_scenario(
        fault=FaultLabel.NORMAL,
        scenario_family=N2_FAMILY,
        instance=1,
        seed=42,
    )

    second = generate_scenario(
        fault=FaultLabel.NORMAL,
        scenario_family=N2_FAMILY,
        instance=2,
        seed=42,
    )

    assert first.scenario_id != second.scenario_id


def test_all_current_families_generate_unique_ids_for_same_instance():
    cases = (
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
    )

    scenario_ids = {
        generate_scenario(
            fault=fault,
            scenario_family=family,
            instance=1,
            seed=42,
        ).scenario_id
        for fault, family in cases
    }

    assert len(scenario_ids) == len(cases)


def test_scenario_id_is_reproducible():
    first = generate_scenario(
        fault=FaultLabel.NETWORK_DEGRADATION,
        scenario_family=ND3_FAMILY,
        instance=12,
        seed=42,
    )

    second = generate_scenario(
        fault=FaultLabel.NETWORK_DEGRADATION,
        scenario_family=ND3_FAMILY,
        instance=12,
        seed=99,
    )

    assert first.scenario_id == second.scenario_id
    assert first.scenario_id == "NETWORK_DEGRADATION_ND-3_012"


def test_seed_is_not_encoded_in_scenario_id():
    first = generate_scenario(
        fault=FaultLabel.MEMORY_PRESSURE,
        scenario_family=MP1_FAMILY,
        instance=5,
        seed=1,
    )

    second = generate_scenario(
        fault=FaultLabel.MEMORY_PRESSURE,
        scenario_family=MP1_FAMILY,
        instance=5,
        seed=999,
    )

    assert first.scenario_id == second.scenario_id


def test_build_scenario_id_rejects_zero_instance():
    with pytest.raises(ValueError, match="instance must be at least 1"):
        build_scenario_id(
            fault=FaultLabel.NORMAL,
            scenario_family=N1_FAMILY,
            instance=0,
        )


def test_build_scenario_id_rejects_invalid_family_format():
    with pytest.raises(ValueError, match="scenario_family must begin"):
        build_scenario_id(
            fault=FaultLabel.NORMAL,
            scenario_family="INVALID_FAMILY",
            instance=1,
        )