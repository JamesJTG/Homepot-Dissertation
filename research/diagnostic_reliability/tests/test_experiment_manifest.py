"""Tests for the frozen A8.5 main-experiment manifest."""

from __future__ import annotations

from collections import Counter

from research.diagnostic_reliability.experiment_manifest import (
    FAMILY_MAP,
    build_manifest,
)
from research.diagnostic_reliability.taxonomy import FaultLabel


def test_manifest_contains_150_scenarios() -> None:
    manifest = build_manifest()

    assert len(manifest) == 150


def test_manifest_contains_30_instances_per_fault() -> None:
    manifest = build_manifest()

    counts = Counter(
        entry.fault
        for entry in manifest
    )

    assert counts == {
        fault.value: 30
        for fault in FaultLabel
    }


def test_manifest_contains_10_instances_per_family() -> None:
    manifest = build_manifest()

    counts = Counter(
        entry.scenario_family
        for entry in manifest
    )

    assert len(counts) == 15

    assert all(
        count == 10
        for count in counts.values()
    )


def test_all_expected_families_are_present() -> None:
    manifest = build_manifest()

    expected = {
        family
        for families in FAMILY_MAP.values()
        for family in families
    }

    actual = {
        entry.scenario_family
        for entry in manifest
    }

    assert actual == expected


def test_manifest_indices_are_unique_and_sequential() -> None:
    manifest = build_manifest()

    assert [
        entry.manifest_index
        for entry in manifest
    ] == list(
        range(
            1,
            151,
        )
    )


def test_seeds_are_unique() -> None:
    manifest = build_manifest()

    seeds = [
        entry.seed
        for entry in manifest
    ]

    assert len(seeds) == len(set(seeds))


def test_manifest_is_deterministic() -> None:
    first = build_manifest()
    second = build_manifest()

    assert first == second


def test_manifest_contains_no_unknown_ground_truth() -> None:
    manifest = build_manifest()

    assert all(
        entry.fault != "UNKNOWN"
        for entry in manifest
    )