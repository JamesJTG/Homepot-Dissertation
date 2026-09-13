"""Tests for deterministic temporal sampling."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from research.diagnostic_reliability.scenarios import (
    generate_scenario,
)
from research.diagnostic_reliability.taxonomy import (
    FaultLabel,
)
from research.diagnostic_reliability.temporal_sampling import (
    downsample_scenario,
    evenly_spaced_indices,
)


def _scenario():
    return generate_scenario(
        FaultLabel.NETWORK_DEGRADATION,
        instance=1,
        seed=20260924,
        end_time=datetime(
            2026,
            9,
            13,
            0,
            0,
            tzinfo=timezone.utc,
        ),
        device_id="EXP_DEVICE_01",
    )


def test_24_observations_reduce_to_8():
    scenario = _scenario()

    assert len(
        scenario.telemetry
    ) == 24

    reduced = downsample_scenario(
        scenario,
        n_selected=8,
    )

    assert len(
        reduced.telemetry
    ) == 8


def test_expected_evenly_spaced_indices_for_24_to_8():
    indices = evenly_spaced_indices(
        24,
        8,
    )

    assert indices == (
        0,
        3,
        7,
        10,
        13,
        16,
        20,
        23,
    )


def test_first_and_last_observations_are_preserved():
    scenario = _scenario()

    reduced = downsample_scenario(
        scenario,
        n_selected=8,
    )

    assert (
        reduced.telemetry[0]
        == scenario.telemetry[0]
    )

    assert (
        reduced.telemetry[-1]
        == scenario.telemetry[-1]
    )


def test_selected_observations_are_existing_points():
    scenario = _scenario()

    reduced = downsample_scenario(
        scenario,
        n_selected=8,
    )

    for point in reduced.telemetry:
        assert point in scenario.telemetry


def test_temporal_order_is_preserved():
    scenario = _scenario()

    reduced = downsample_scenario(
        scenario,
        n_selected=8,
    )

    timestamps = [
        point.timestamp
        for point in reduced.telemetry
    ]

    assert timestamps == sorted(
        timestamps
    )

    assert len(
        timestamps
    ) == len(
        set(timestamps)
    )


def test_scenario_metadata_is_unchanged():
    scenario = _scenario()

    reduced = downsample_scenario(
        scenario,
        n_selected=8,
    )

    assert (
        reduced.scenario_id
        == scenario.scenario_id
    )

    assert (
        reduced.scenario_family
        == scenario.scenario_family
    )

    assert (
        reduced.device_id
        == scenario.device_id
    )

    assert (
        reduced.fault
        == scenario.fault
    )

    assert (
        reduced.seed
        == scenario.seed
    )


def test_original_scenario_is_not_modified():
    scenario = _scenario()

    original_telemetry = (
        scenario.telemetry
    )

    reduced = downsample_scenario(
        scenario,
        n_selected=8,
    )

    assert (
        scenario.telemetry
        == original_telemetry
    )

    assert len(
        scenario.telemetry
    ) == 24

    assert (
        reduced.telemetry
        is not scenario.telemetry
    )


def test_selecting_all_observations_preserves_sequence():
    scenario = _scenario()

    reduced = downsample_scenario(
        scenario,
        n_selected=24,
    )

    assert (
        reduced.telemetry
        == scenario.telemetry
    )


def test_invalid_selection_larger_than_source_is_rejected():
    scenario = _scenario()

    with pytest.raises(
        ValueError,
        match=(
            "n_selected cannot exceed "
            "n_observations"
        ),
    ):
        downsample_scenario(
            scenario,
            n_selected=25,
        )


def test_zero_selected_observations_is_rejected():
    scenario = _scenario()

    with pytest.raises(
        ValueError,
        match=(
            "n_selected must be at least 1"
        ),
    ):
        downsample_scenario(
            scenario,
            n_selected=0,
        )