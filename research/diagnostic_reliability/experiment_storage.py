"""Durable storage primitives for the A8.5 main experiment.

The main diagnostic-reliability experiment is intentionally long-running.
Experimental state must therefore survive interruption without silently
duplicating observations or replacing already collected evidence.

Three research artefacts are maintained inside a run directory:

- ``metadata.json``:
    immutable provenance for the experiment run;

- ``cases.jsonl``:
    frozen prepared evidence cases, including validation output and the exact
    leakage-controlled model context used for generation;

- ``generations.jsonl``:
    one durable record for each observed LLM generation;

- ``technical_errors.jsonl``:
    operational failures such as temporarily unavailable Ollama calls. These
    are not treated as model generations.

JSON Lines is used for append-only experimental observations. Each append is
flushed and synchronised to disk before execution continues.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


METADATA_FILENAME = "metadata.json"
CASES_FILENAME = "cases.jsonl"
GENERATIONS_FILENAME = "generations.jsonl"
TECHNICAL_ERRORS_FILENAME = "technical_errors.jsonl"


@dataclass(frozen=True)
class ExperimentPaths:
    """Filesystem paths belonging to one experiment run."""

    run_dir: Path
    metadata: Path
    cases: Path
    generations: Path
    technical_errors: Path


def experiment_paths(
    run_dir: str | Path,
) -> ExperimentPaths:
    """Return all durable paths associated with one experiment run."""

    root = Path(run_dir)

    return ExperimentPaths(
        run_dir=root,
        metadata=root / METADATA_FILENAME,
        cases=root / CASES_FILENAME,
        generations=root / GENERATIONS_FILENAME,
        technical_errors=root / TECHNICAL_ERRORS_FILENAME,
    )


def canonical_json(
    value: Any,
) -> str:
    """Return deterministic compact JSON suitable for hashing."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def sha256_text(
    text: str,
) -> str:
    """Return a hexadecimal SHA-256 digest for text."""

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def sha256_json(
    value: Any,
) -> str:
    """Return a deterministic SHA-256 digest for JSON-compatible data."""

    return sha256_text(
        canonical_json(value)
    )


def ensure_run_directory(
    run_dir: str | Path,
) -> ExperimentPaths:
    """Create the run directory and return its paths."""

    paths = experiment_paths(
        run_dir
    )

    paths.run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return paths


def write_immutable_json(
    path: str | Path,
    document: Mapping[str, Any],
) -> Path:
    """Create an immutable JSON document or verify an existing copy.

    Existing content must be semantically identical. A run can therefore be
    resumed safely, while accidental changes to frozen metadata cause an
    explicit failure rather than silently altering the experiment.
    """

    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    serialised = json.dumps(
        document,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    ) + "\n"

    if output_path.exists():
        existing = output_path.read_text(
            encoding="utf-8"
        )

        if existing != serialised:
            raise RuntimeError(
                "Existing immutable experiment document differs "
                f"from the requested content: {output_path}"
            )

        return output_path

    temporary = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    try:
        with temporary.open(
            "w",
            encoding="utf-8",
        ) as handle:
            handle.write(
                serialised
            )
            handle.flush()
            os.fsync(
                handle.fileno()
            )

        os.replace(
            temporary,
            output_path,
        )

    finally:
        if temporary.exists():
            temporary.unlink()

    return output_path


def _serialise_jsonl_record(
    record: Mapping[str, Any],
) -> str:
    """Serialise one JSONL observation."""

    return (
        json.dumps(
            record,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        + "\n"
    )


def append_jsonl(
    path: str | Path,
    record: Mapping[str, Any],
) -> None:
    """Durably append one JSON record."""

    append_jsonl_batch(
        path,
        [record],
    )


def append_jsonl_batch(
    path: str | Path,
    records: Iterable[Mapping[str, Any]],
) -> None:
    """Durably append a batch of JSON records using one filesystem write.

    Scenario preparation uses this function to append all evidence conditions
    belonging to one scenario together after the entire scenario has passed
    preparation checks.
    """

    materialised = list(
        records
    )

    if not materialised:
        return

    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = "".join(
        _serialise_jsonl_record(
            record
        )
        for record in materialised
    )

    with output_path.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            payload
        )
        handle.flush()
        os.fsync(
            handle.fileno()
        )


def load_jsonl(
    path: str | Path,
) -> list[dict[str, Any]]:
    """Load and validate a JSONL experiment artefact."""

    input_path = Path(path)

    if not input_path.exists():
        return []

    records: list[dict[str, Any]] = []

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, raw_line in enumerate(
            handle,
            start=1,
        ):
            line = raw_line.strip()

            if not line:
                continue

            try:
                payload = json.loads(
                    line
                )
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "Malformed JSONL record in "
                    f"{input_path} at line {line_number}."
                ) from exc

            if not isinstance(
                payload,
                dict,
            ):
                raise RuntimeError(
                    "JSONL records must be objects: "
                    f"{input_path}, line {line_number}."
                )

            records.append(
                payload
            )

    return records


def index_unique(
    records: Iterable[Mapping[str, Any]],
    *,
    key: str,
    source_name: str,
) -> dict[Any, Mapping[str, Any]]:
    """Index records by a field while rejecting duplicate identities."""

    index: dict[
        Any,
        Mapping[str, Any],
    ] = {}

    for record in records:
        if key not in record:
            raise RuntimeError(
                f"{source_name} record is missing required key {key!r}."
            )

        identity = record[
            key
        ]

        if identity in index:
            raise RuntimeError(
                f"Duplicate {key}={identity!r} detected in {source_name}."
            )

        index[
            identity
        ] = record

    return index


def generation_key(
    *,
    case_id: str,
    generation_id: int,
) -> str:
    """Return the stable identity of one required LLM generation."""

    if not case_id:
        raise ValueError(
            "case_id must not be empty"
        )

    if generation_id < 1:
        raise ValueError(
            "generation_id must be at least 1"
        )

    return (
        f"{case_id}::generation-"
        f"{generation_id:02d}"
    )


def index_generations(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    """Index generation records and reject duplicate generation identities."""

    index: dict[
        str,
        Mapping[str, Any],
    ] = {}

    for record in records:
        try:
            case_id = str(
                record["case_id"]
            )
            generation_id = int(
                record["generation_id"]
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise RuntimeError(
                "Generation record is missing a valid "
                "case_id or generation_id."
            ) from exc

        identity = generation_key(
            case_id=case_id,
            generation_id=generation_id,
        )

        if identity in index:
            raise RuntimeError(
                "Duplicate generation observation "
                f"detected: {identity}"
            )

        index[
            identity
        ] = record

    return index


def completed_generation_keys(
    records: Iterable[Mapping[str, Any]],
) -> set[str]:
    """Return identities already represented by observed model responses.

    Both successfully parsed outputs and parse failures count as completed
    generations because both are genuine LLM observations.

    Technical call failures are deliberately stored separately and therefore
    never appear here.
    """

    index = index_generations(
        records
    )

    completed: set[str] = set()

    for identity, record in index.items():
        status = record.get(
            "status"
        )

        if status not in {
            "ok",
            "parse_error",
        }:
            raise RuntimeError(
                "Unexpected generation status "
                f"{status!r} for {identity}."
            )

        completed.add(
            identity
        )

    return completed