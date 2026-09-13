"""Tests for A8.5 main-experiment orchestration helpers."""

from __future__ import annotations

import pytest

from research.diagnostic_reliability.main_experiment import (
    build_run_metadata,
    case_id_for,
    generation_record_from_response,
    selected_manifest_entries,
    validate_expected_envelope,
    verify_existing_case_groups,
)
from research.diagnostic_reliability.taxonomy import (
    EvidenceCondition,
)


def _case(
    *,
    case_id: str = "scenario::COMPLETE_FRESH",
) -> dict:
    return {
        "case_id": case_id,
        "manifest_index": 1,
        "evidence_condition": (
            EvidenceCondition
            .COMPLETE_FRESH
            .value
        ),
        "model_context_sha256": (
            "abc123"
        ),
    }


def _complete_envelope() -> dict:
    return {
        "failed_gate_id": None,
        "actionable": True,
        "gates": {
            "A": {},
            "B": {},
            "C": {},
            "D": {},
            "E": {},
        },
    }


def _gate_b_failure() -> dict:
    return {
        "failed_gate_id": "B",
        "actionable": False,
        "gates": {
            "A": {},
            "B": {},
        },
    }


def test_case_id_contains_evidence_condition() -> None:
    assert (
        case_id_for(
            "CPU_SATURATION_CS-1_001",
            EvidenceCondition.COMPLETE_FRESH,
        )
        ==
        "CPU_SATURATION_CS-1_001::COMPLETE_FRESH"
    )


def test_complete_fresh_accepts_full_envelope() -> None:
    validate_expected_envelope(
        condition=(
            EvidenceCondition
            .COMPLETE_FRESH
        ),
        envelope=(
            _complete_envelope()
        ),
    )


@pytest.mark.parametrize(
    "condition",
    [
        EvidenceCondition.STALE,
        EvidenceCondition.INCOMPLETE,
        EvidenceCondition.DISCONTINUOUS,
    ],
)
def test_degraded_q_conditions_accept_gate_b_failure(
    condition,
) -> None:
    validate_expected_envelope(
        condition=condition,
        envelope=(
            _gate_b_failure()
        ),
    )


def test_complete_fresh_rejects_gate_b_failure() -> None:
    with pytest.raises(
        RuntimeError,
        match="COMPLETE_FRESH",
    ):
        validate_expected_envelope(
            condition=(
                EvidenceCondition
                .COMPLETE_FRESH
            ),
            envelope=(
                _gate_b_failure()
            ),
        )


def test_degraded_condition_rejects_passing_envelope() -> None:
    with pytest.raises(
        RuntimeError,
        match="expected to fail Gate B",
    ):
        validate_expected_envelope(
            condition=(
                EvidenceCondition
                .STALE
            ),
            envelope=(
                _complete_envelope()
            ),
        )


def test_selected_manifest_entries_supports_one_scenario() -> None:
    entries = (
        selected_manifest_entries(
            max_scenarios=1
        )
    )

    assert len(
        entries
    ) == 1

    assert (
        entries[0]
        .manifest_index
        == 1
    )


def test_selected_manifest_entries_rejects_zero() -> None:
    with pytest.raises(
        ValueError,
        match="at least 1",
    ):
        selected_manifest_entries(
            max_scenarios=0
        )


def test_generation_record_parses_valid_response() -> None:
    record = (
        generation_record_from_response(
            case=_case(),
            generation_id=1,
            raw_response=(
                '{"diagnosis":"NORMAL",'
                '"confidence":0.8,'
                '"explanation":"Stable evidence."}'
            ),
            latency_seconds=1.25,
            attempt=1,
        )
    )

    assert (
        record[
            "status"
        ]
        == "ok"
    )

    assert (
        record[
            "diagnosis"
        ]
        == "NORMAL"
    )

    assert (
        record[
            "confidence"
        ]
        == 0.8
    )


def test_generation_record_preserves_parse_failure() -> None:
    record = (
        generation_record_from_response(
            case=_case(),
            generation_id=2,
            raw_response=(
                "not-json"
            ),
            latency_seconds=0.75,
            attempt=1,
        )
    )

    assert (
        record[
            "status"
        ]
        == "parse_error"
    )

    assert (
        record[
            "raw_response"
        ]
        == "not-json"
    )

    assert (
        record[
            "diagnosis"
        ]
        is None
    )

    assert (
        record[
            "parser_error_type"
        ]
        is not None
    )


def test_verify_existing_case_groups_accepts_empty_run() -> None:
    entries = (
        selected_manifest_entries(
            max_scenarios=1
        )
    )

    index = (
        verify_existing_case_groups(
            records=[],
            selected_entries=entries,
        )
    )

    assert index == {}


def test_verify_existing_case_groups_rejects_partial_q_group() -> None:
    entries = (
        selected_manifest_entries(
            max_scenarios=1
        )
    )

    records = [
        {
            "case_id": (
                "scenario::COMPLETE_FRESH"
            ),
            "manifest_index": 1,
            "evidence_condition": (
                "COMPLETE_FRESH"
            ),
            "base_scenario_end_time": (
                "2026-09-13T12:00:00+00:00"
            ),
        }
    ]

    with pytest.raises(
        RuntimeError,
        match="Partial evidence-case group",
    ):
        verify_existing_case_groups(
            records=records,
            selected_entries=entries,
        )


def test_verify_existing_case_groups_accepts_four_paired_q_cases() -> None:
    entries = (
        selected_manifest_entries(
            max_scenarios=1
        )
    )

    base_time = (
        "2026-09-13T12:00:00+00:00"
    )

    records = [
        {
            "case_id": (
                f"scenario::{condition.value}"
            ),
            "manifest_index": 1,
            "evidence_condition": (
                condition.value
            ),
            "base_scenario_end_time": (
                base_time
            ),
        }
        for condition in EvidenceCondition
    ]

    index = (
        verify_existing_case_groups(
            records=records,
            selected_entries=entries,
        )
    )

    assert len(
        index
    ) == 4


def test_verify_existing_case_groups_rejects_different_pair_timestamps() -> None:
    entries = (
        selected_manifest_entries(
            max_scenarios=1
        )
    )

    records = []

    for index, condition in enumerate(
        EvidenceCondition
    ):
        records.append(
            {
                "case_id": (
                    f"scenario::{condition.value}"
                ),
                "manifest_index": 1,
                "evidence_condition": (
                    condition.value
                ),
                "base_scenario_end_time": (
                    f"time-{index}"
                ),
            }
        )

    with pytest.raises(
        RuntimeError,
        match="do not share one base scenario",
    ):
        verify_existing_case_groups(
            records=records,
            selected_entries=entries,
        )


def test_run_metadata_records_selected_experiment_size() -> None:
    entries = (
        selected_manifest_entries(
            max_scenarios=1
        )
    )

    metadata = (
        build_run_metadata(
            selected_entries=entries,
            git={
                "commit": "abc123",
                "dirty": False,
            },
        )
    )

    assert (
        metadata[
            "selected_scenario_count"
        ]
        == 1
    )

    assert (
        metadata[
            "selected_evidence_case_count"
        ]
        == 4
    )

    assert (
        metadata[
            "selected_llm_generation_count"
        ]
        == 28
    )

    assert (
        metadata[
            "ground_truth_model_visible"
        ]
        is False
    )