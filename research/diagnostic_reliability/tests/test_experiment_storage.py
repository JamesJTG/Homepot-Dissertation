"""Tests for durable A8.5 experiment storage."""

from __future__ import annotations

import json

import pytest

from research.diagnostic_reliability.experiment_storage import (
    append_jsonl,
    append_jsonl_batch,
    completed_generation_keys,
    ensure_run_directory,
    experiment_paths,
    generation_key,
    index_generations,
    index_unique,
    load_jsonl,
    sha256_json,
    sha256_text,
    write_immutable_json,
)


def test_experiment_paths_are_inside_run_directory(
    tmp_path,
) -> None:
    paths = experiment_paths(
        tmp_path / "run"
    )

    assert paths.metadata.parent == paths.run_dir
    assert paths.cases.parent == paths.run_dir
    assert paths.generations.parent == paths.run_dir
    assert paths.technical_errors.parent == paths.run_dir


def test_ensure_run_directory_creates_directory(
    tmp_path,
) -> None:
    paths = ensure_run_directory(
        tmp_path / "run"
    )

    assert paths.run_dir.is_dir()


def test_sha256_text_is_deterministic() -> None:
    assert (
        sha256_text("abc")
        ==
        sha256_text("abc")
    )


def test_sha256_json_ignores_dictionary_key_order() -> None:
    left = {
        "a": 1,
        "b": 2,
    }

    right = {
        "b": 2,
        "a": 1,
    }

    assert (
        sha256_json(left)
        ==
        sha256_json(right)
    )


def test_write_immutable_json_creates_file(
    tmp_path,
) -> None:
    path = (
        tmp_path
        / "metadata.json"
    )

    write_immutable_json(
        path,
        {
            "model": "gemma3:12b",
            "temperature": 0.4,
        },
    )

    stored = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    assert stored["model"] == "gemma3:12b"
    assert stored["temperature"] == 0.4


def test_write_immutable_json_accepts_identical_existing_content(
    tmp_path,
) -> None:
    path = (
        tmp_path
        / "metadata.json"
    )

    document = {
        "a": 1,
        "b": 2,
    }

    write_immutable_json(
        path,
        document,
    )

    write_immutable_json(
        path,
        document,
    )


def test_write_immutable_json_rejects_changed_existing_content(
    tmp_path,
) -> None:
    path = (
        tmp_path
        / "metadata.json"
    )

    write_immutable_json(
        path,
        {
            "model": "gemma3:12b",
        },
    )

    with pytest.raises(
        RuntimeError,
        match="differs",
    ):
        write_immutable_json(
            path,
            {
                "model": "llama3.2",
            },
        )


def test_append_and_load_jsonl(
    tmp_path,
) -> None:
    path = (
        tmp_path
        / "records.jsonl"
    )

    append_jsonl(
        path,
        {
            "id": 1,
        },
    )

    append_jsonl(
        path,
        {
            "id": 2,
        },
    )

    assert load_jsonl(
        path
    ) == [
        {
            "id": 1,
        },
        {
            "id": 2,
        },
    ]


def test_append_jsonl_batch_preserves_record_order(
    tmp_path,
) -> None:
    path = (
        tmp_path
        / "cases.jsonl"
    )

    append_jsonl_batch(
        path,
        [
            {
                "case_id": "Q1",
            },
            {
                "case_id": "Q2",
            },
            {
                "case_id": "Q3",
            },
            {
                "case_id": "Q4",
            },
        ],
    )

    records = load_jsonl(
        path
    )

    assert [
        record["case_id"]
        for record in records
    ] == [
        "Q1",
        "Q2",
        "Q3",
        "Q4",
    ]


def test_load_jsonl_returns_empty_for_missing_file(
    tmp_path,
) -> None:
    assert load_jsonl(
        tmp_path
        / "missing.jsonl"
    ) == []


def test_load_jsonl_rejects_malformed_record(
    tmp_path,
) -> None:
    path = (
        tmp_path
        / "broken.jsonl"
    )

    path.write_text(
        '{"id": 1}\n{"id":',
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="Malformed JSONL",
    ):
        load_jsonl(
            path
        )


def test_index_unique_rejects_duplicates() -> None:
    records = [
        {
            "case_id": "case-1",
        },
        {
            "case_id": "case-1",
        },
    ]

    with pytest.raises(
        RuntimeError,
        match="Duplicate",
    ):
        index_unique(
            records,
            key="case_id",
            source_name="cases",
        )


def test_generation_key_is_stable() -> None:
    assert (
        generation_key(
            case_id="CASE-001",
            generation_id=7,
        )
        ==
        "CASE-001::generation-07"
    )


def test_generation_key_rejects_zero_generation() -> None:
    with pytest.raises(
        ValueError,
        match="at least 1",
    ):
        generation_key(
            case_id="CASE-001",
            generation_id=0,
        )


def test_index_generations_rejects_duplicate_observation() -> None:
    records = [
        {
            "case_id": "case-1",
            "generation_id": 1,
            "status": "ok",
        },
        {
            "case_id": "case-1",
            "generation_id": 1,
            "status": "ok",
        },
    ]

    with pytest.raises(
        RuntimeError,
        match="Duplicate generation",
    ):
        index_generations(
            records
        )


def test_completed_generations_include_parse_errors() -> None:
    records = [
        {
            "case_id": "case-1",
            "generation_id": 1,
            "status": "ok",
        },
        {
            "case_id": "case-1",
            "generation_id": 2,
            "status": "parse_error",
        },
    ]

    assert completed_generation_keys(
        records
    ) == {
        "case-1::generation-01",
        "case-1::generation-02",
    }


def test_completed_generations_reject_unknown_status() -> None:
    records = [
        {
            "case_id": "case-1",
            "generation_id": 1,
            "status": "technical_error",
        },
    ]

    with pytest.raises(
        RuntimeError,
        match="Unexpected generation status",
    ):
        completed_generation_keys(
            records
        )