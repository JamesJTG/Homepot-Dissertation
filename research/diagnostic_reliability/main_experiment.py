"""A8.5 resumable main diagnostic-reliability experiment.

This module executes the frozen main experiment defined after the A8.4
research-assurance pilot.

Experimental hierarchy:

    fault
        -> scenario family
            -> scenario instance
                -> evidence condition
                    -> repeated LLM generation

The runner preserves the distinction between:

- F: private operational ground truth;
- Q: controlled evidence-quality condition;
- E: HOMEPOT Validation Envelope evidence-readiness information;
- model-visible context;
- raw LLM generations.

Ground truth and other private research metadata are persisted for later
analysis but are never passed automatically into the model-visible context.

Persistence is intentionally incremental. Completed LLM observations are not
repeated on resume. Parse failures count as genuine model observations, while
technical Ollama failures do not and may therefore be retried.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import platform
import subprocess
import time
from typing import Any, Mapping

from sqlalchemy import text

from homepot.database import get_database_service

from .context import build_model_context
from .evidence_conditions import (
    apply_evidence_condition,
    expected_sufficiency,
)
from .experiment_config import MAIN_EXPERIMENT
from .experiment_manifest import (
    ManifestEntry,
    build_manifest,
    manifest_document,
    write_manifest,
)
from .experiment_storage import (
    append_jsonl,
    append_jsonl_batch,
    completed_generation_keys,
    ensure_run_directory,
    generation_key,
    index_unique,
    load_jsonl,
    sha256_json,
    sha256_text,
    write_immutable_json,
)
from .ollama_adapter import ExperimentalOllamaAdapter
from .parser import parse_diagnosis
from .prompt import (
    DIAGNOSTIC_PROMPT,
    SYSTEM_PROMPT,
)
from .scenarios import generate_scenario
from .taxonomy import (
    EvidenceCondition,
    FaultLabel,
)
from .validation_integration import (
    RESEARCH_DEVICE_ID,
    envelope_summary,
    evaluate_validation_envelope,
    materialise_scenario,
    model_visible_validation,
    provision_experimental_device,
)


RUNNER_VERSION = "A8.5-main-v2"

EXPECTED_EXPERIMENT_DATABASE = "homepot_dissertation_db"

DEFAULT_MANIFEST_PATH = Path(
    "research/diagnostic_reliability/"
    "outputs/main_experiment_manifest.json"
)

DEFAULT_MAIN_RUN_DIR = Path(
    "research/diagnostic_reliability/"
    "outputs/main_experiment"
)

TECHNICAL_MAX_ATTEMPTS = 3
TECHNICAL_RETRY_BACKOFF_SECONDS = 2.0


def utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""

    return datetime.now(
        timezone.utc
    )


def utc_now_iso() -> str:
    """Return the current UTC timestamp as ISO 8601 text."""

    return utc_now().isoformat()


def validate_experiment_database_name(
    database_name: str,
) -> None:
    """Reject any database other than the dedicated experiment database."""

    if database_name != EXPECTED_EXPERIMENT_DATABASE:
        raise RuntimeError(
            "Refusing to execute the diagnostic-reliability experiment "
            f"against database {database_name!r}. "
            "Expected dedicated experimental database "
            f"{EXPECTED_EXPERIMENT_DATABASE!r}."
        )


async def verify_experiment_database(
    db,
) -> str:
    """Return the connected database name after enforcing the safety guard."""

    async with db.get_session() as session:
        database_name = (
            await session.execute(
                text(
                    "SELECT current_database()"
                )
            )
        ).scalar_one()

    database_name = str(
        database_name
    )

    validate_experiment_database_name(
        database_name
    )

    return database_name


def case_id_for(
    scenario_id: str,
    condition: EvidenceCondition,
) -> str:
    """Return the stable identity of one scenario-by-Q evidence case."""

    if not scenario_id:
        raise ValueError(
            "scenario_id must not be empty"
        )

    return (
        f"{scenario_id}::"
        f"{condition.value}"
    )


def _git_output(
    *args: str,
) -> str:
    """Run a read-only Git command and return stripped stdout."""

    result = subprocess.run(
        [
            "git",
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    return result.stdout.strip()


def git_provenance() -> dict[str, Any]:
    """Return Git provenance and reject an uncommitted experiment tree."""

    commit = _git_output(
        "rev-parse",
        "HEAD",
    )

    status = _git_output(
        "status",
        "--porcelain",
    )

    dirty = bool(
        status
    )

    return {
        "commit": commit,
        "dirty": dirty,
    }


def leakage_checks(
    *,
    context: str,
    scenario,
) -> dict[str, bool]:
    """Verify private research metadata is absent from model-visible context."""

    fault_value = (
        scenario.fault.value
    )

    return {
        "scenario_id_absent": (
            scenario.scenario_id
            not in context
        ),
        "scenario_family_absent": (
            scenario.scenario_family
            not in context
        ),
        "seed_absent": (
            str(scenario.seed)
            not in context
        ),
        "private_fault_metadata_absent": (
            (
                f"fault: {fault_value}"
                not in context
            )
            and
            (
                f"fault={fault_value}"
                not in context
            )
        ),
    }


def validate_expected_envelope(
    *,
    condition: EvidenceCondition,
    envelope: Mapping[str, Any],
) -> None:
    """Verify that Q manipulations affect the real envelope as intended.

    COMPLETE_FRESH is expected to pass the full Validation Envelope.

    STALE, INCOMPLETE and DISCONTINUOUS are controlled Gate B degradations and
    are therefore expected to stop the envelope at Gate B.

    Exact trust-score values are deliberately not asserted because they are
    production rule outputs rather than research-defined constants.
    """

    failed_gate_id = envelope.get(
        "failed_gate_id"
    )

    actionable = envelope.get(
        "actionable"
    )

    gates = envelope.get(
        "gates",
        {},
    )

    if condition is EvidenceCondition.COMPLETE_FRESH:
        if failed_gate_id is not None:
            raise RuntimeError(
                "COMPLETE_FRESH evidence unexpectedly failed "
                f"the Validation Envelope at {failed_gate_id!r}."
            )

        if actionable is not True:
            raise RuntimeError(
                "COMPLETE_FRESH evidence was unexpectedly "
                "non-actionable."
            )

        expected_gates = {
            "A",
            "B",
            "C",
            "D",
            "E",
        }

        if set(
            gates
        ) != expected_gates:
            raise RuntimeError(
                "COMPLETE_FRESH evidence did not evaluate "
                "the complete A-E Validation Envelope."
            )

        return

    if condition in {
        EvidenceCondition.STALE,
        EvidenceCondition.INCOMPLETE,
        EvidenceCondition.DISCONTINUOUS,
    }:
        if failed_gate_id != "B":
            raise RuntimeError(
                f"{condition.value} evidence was expected to "
                "fail Gate B but failed "
                f"{failed_gate_id!r}."
            )

        if actionable is not False:
            raise RuntimeError(
                f"{condition.value} evidence unexpectedly "
                "remained actionable."
            )

        if "A" not in gates or "B" not in gates:
            raise RuntimeError(
                f"{condition.value} evidence did not evaluate "
                "the expected A -> B validation path."
            )

        return

    raise ValueError(
        f"Unsupported evidence condition: {condition}"
    )


def selected_manifest_entries(
    *,
    max_scenarios: int | None,
) -> list[ManifestEntry]:
    """Return the requested prefix of the frozen scenario manifest."""

    manifest = build_manifest()

    if max_scenarios is None:
        return manifest

    if max_scenarios < 1:
        raise ValueError(
            "max_scenarios must be at least 1"
        )

    if max_scenarios > len(
        manifest
    ):
        raise ValueError(
            "max_scenarios cannot exceed the frozen "
            f"manifest size of {len(manifest)}"
        )

    return manifest[
        :max_scenarios
    ]


def build_run_metadata(
    *,
    selected_entries: list[ManifestEntry],
    git: Mapping[str, Any],
    database_name: str,
) -> dict[str, Any]:
    """Return immutable provenance for one experiment run."""

    frozen_manifest = (
        manifest_document()
    )

    selected_document = [
        asdict(
            entry
        )
        for entry in selected_entries
    ]

    return {
        "runner_version": (
            RUNNER_VERSION
        ),
        "git": dict(
            git
        ),
        "database": {
            "name": database_name,
            "expected_name": (
                EXPECTED_EXPERIMENT_DATABASE
            ),
        },
        "python_version": (
            platform.python_version()
        ),
        "experiment_config": (
            asdict(
                MAIN_EXPERIMENT
            )
        ),
        "manifest_version": (
            frozen_manifest[
                "manifest_version"
            ]
        ),
        "manifest_sha256": (
            sha256_json(
                frozen_manifest
            )
        ),
        "selected_manifest_sha256": (
            sha256_json(
                selected_document
            )
        ),
        "selected_manifest_indices": [
            entry.manifest_index
            for entry in selected_entries
        ],
        "selected_scenario_count": (
            len(
                selected_entries
            )
        ),
        "selected_evidence_case_count": (
            len(
                selected_entries
            )
            * len(
                EvidenceCondition
            )
        ),
        "selected_llm_generation_count": (
            len(
                selected_entries
            )
            * len(
                EvidenceCondition
            )
            * MAIN_EXPERIMENT
            .generations_per_evidence_case
        ),
        "prompt_sha256": (
            sha256_text(
                DIAGNOSTIC_PROMPT
            )
        ),
        "system_prompt_sha256": (
            sha256_text(
                SYSTEM_PROMPT
            )
        ),
        "technical_retry_policy": {
            "max_attempts": (
                TECHNICAL_MAX_ATTEMPTS
            ),
            "backoff_seconds": (
                TECHNICAL_RETRY_BACKOFF_SECONDS
            ),
            "parse_failures_retried": (
                False
            ),
        },
        "ground_truth_model_visible": (
            False
        ),
    }


def verify_existing_case_groups(
    *,
    records: list[dict[str, Any]],
    selected_entries: list[ManifestEntry],
) -> dict[str, dict[str, Any]]:
    """Validate persisted case completeness and return an index by case ID.

    A scenario must have either zero persisted Q cases or all four. The runner
    writes the four cases as a batch, so any partial group indicates an
    interrupted/corrupt case artefact that should be investigated rather than
    silently repaired.
    """

    index = index_unique(
        records,
        key="case_id",
        source_name="cases",
    )

    selected_indices = {
        entry.manifest_index
        for entry in selected_entries
    }

    expected_conditions = {
        condition.value
        for condition in EvidenceCondition
    }

    grouped: dict[
        int,
        list[Mapping[str, Any]],
    ] = {}

    for record in records:
        try:
            manifest_index = int(
                record[
                    "manifest_index"
                ]
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise RuntimeError(
                "Persisted case is missing a valid manifest_index."
            ) from exc

        if manifest_index not in selected_indices:
            raise RuntimeError(
                "Run directory contains a case outside the "
                f"selected manifest: {manifest_index}"
            )

        grouped.setdefault(
            manifest_index,
            [],
        ).append(
            record
        )

    for entry in selected_entries:
        group = grouped.get(
            entry.manifest_index,
            [],
        )

        if not group:
            continue

        if len(
            group
        ) != len(
            EvidenceCondition
        ):
            raise RuntimeError(
                "Partial evidence-case group detected for "
                f"manifest index {entry.manifest_index}. "
                "Expected either zero or four persisted cases."
            )

        conditions = {
            record.get(
                "evidence_condition"
            )
            for record in group
        }

        if conditions != expected_conditions:
            raise RuntimeError(
                "Persisted evidence conditions do not match "
                "the frozen Q set for manifest index "
                f"{entry.manifest_index}."
            )

        base_times = {
            record.get(
                "base_scenario_end_time"
            )
            for record in group
        }

        if len(
            base_times
        ) != 1:
            raise RuntimeError(
                "Paired Q cases do not share one base scenario "
                f"timestamp for manifest index {entry.manifest_index}."
            )

    return {
        str(
            case_id
        ): dict(
            record
        )
        for case_id, record in index.items()
    }


async def prepare_scenario_cases(
    *,
    db,
    device_id: str,
    device_int_id: int,
    entry: ManifestEntry,
) -> list[dict[str, Any]]:
    """Prepare and validate the four paired Q cases for one scenario."""

    base_end_time = utc_now()

    fault = FaultLabel(
        entry.fault
    )

    base_scenario = generate_scenario(
        fault,
        instance=entry.instance,
        seed=entry.seed,
        device_id=device_id,
        scenario_family=entry.scenario_family,
        n_points=(
            MAIN_EXPERIMENT
            .telemetry_points_per_scenario
        ),
        end_time=base_end_time,
    )

    if (
        len(
            base_scenario.telemetry
        )
        !=
        MAIN_EXPERIMENT
        .telemetry_points_per_scenario
    ):
        raise RuntimeError(
            "Generated scenario has an unexpected number "
            f"of telemetry observations: "
            f"{len(base_scenario.telemetry)}"
        )

    records: list[
        dict[str, Any]
    ] = []

    for condition in EvidenceCondition:
        scenario = apply_evidence_condition(
            base_scenario,
            condition,
        )

        if scenario.fault is not fault:
            raise RuntimeError(
                "Evidence transformation changed private "
                "ground truth."
            )

        pre_validation_context = (
            build_model_context(
                scenario
            )
        )

        async with db.get_session() as session:
            await materialise_scenario(
                session,
                scenario,
                device_int_id=device_int_id,
            )

            envelope_result = (
                await evaluate_validation_envelope(
                    session,
                    device_id=device_id,
                    device_int_id=device_int_id,
                    assembled_context=(
                        pre_validation_context
                    ),
                )
            )

        envelope = envelope_summary(
            envelope_result
        )

        validate_expected_envelope(
            condition=condition,
            envelope=envelope,
        )

        visible_validation = (
            model_visible_validation(
                envelope_result
            )
        )

        model_context = (
            build_model_context(
                scenario,
                validation=(
                    visible_validation
                ),
            )
        )

        leakage = leakage_checks(
            context=model_context,
            scenario=scenario,
        )

        if not all(
            leakage.values()
        ):
            failed = [
                name
                for name, passed
                in leakage.items()
                if not passed
            ]

            raise RuntimeError(
                "Leakage-control failure for "
                f"{scenario.scenario_id}: "
                f"{failed}"
            )

        timestamps = [
            point.timestamp.isoformat()
            for point in scenario.telemetry
        ]

        record = {
            "schema_version": (
                "A8.5-case-v1"
            ),
            "case_id": (
                case_id_for(
                    base_scenario.scenario_id,
                    condition,
                )
            ),
            "manifest_index": (
                entry.manifest_index
            ),
            "private_research": {
                "scenario_id": (
                    base_scenario.scenario_id
                ),
                "scenario_family": (
                    entry.scenario_family
                ),
                "fault": (
                    fault.value
                ),
                "instance": (
                    entry.instance
                ),
                "seed": (
                    entry.seed
                ),
            },
            "evidence_condition": (
                condition.value
            ),
            "expected_sufficiency": (
                expected_sufficiency(
                    condition
                ).value
            ),
            "base_scenario_end_time": (
                base_end_time.isoformat()
            ),
            "evidence_start_time": (
                timestamps[0]
            ),
            "evidence_end_time": (
                timestamps[-1]
            ),
            "telemetry_observations": (
                len(
                    scenario.telemetry
                )
            ),
            "scenario_sha256": (
                sha256_json(
                    scenario.to_dict()
                )
            ),
            "validation": (
                envelope
            ),
            "model_visible_validation": (
                visible_validation
            ),
            "model_context": (
                model_context
            ),
            "model_context_sha256": (
                sha256_text(
                    model_context
                )
            ),
            "model_context_characters": (
                len(
                    model_context
                )
            ),
            "model_context_lines": (
                len(
                    model_context.splitlines()
                )
            ),
            "leakage_checks": (
                leakage
            ),
            "prepared_at_utc": (
                utc_now_iso()
            ),
        }

        records.append(
            record
        )

    return records


def generation_record_from_response(
    *,
    case: Mapping[str, Any],
    generation_id: int,
    raw_response: str,
    latency_seconds: float,
    attempt: int,
) -> dict[str, Any]:
    """Convert one observed model response into a durable record.

    Parse failure is deliberately retained as an observed generation.
    """

    common = {
        "schema_version": (
            "A8.5-generation-v1"
        ),
        "case_id": (
            str(
                case[
                    "case_id"
                ]
            )
        ),
        "manifest_index": (
            int(
                case[
                    "manifest_index"
                ]
            )
        ),
        "evidence_condition": (
            str(
                case[
                    "evidence_condition"
                ]
            )
        ),
        "generation_id": (
            generation_id
        ),
        "model": (
            MAIN_EXPERIMENT.model
        ),
        "temperature": (
            MAIN_EXPERIMENT.temperature
        ),
        "raw_response": (
            raw_response
        ),
        "latency_seconds": (
            latency_seconds
        ),
        "technical_attempt": (
            attempt
        ),
        "model_context_sha256": (
            str(
                case[
                    "model_context_sha256"
                ]
            )
        ),
        "observed_at_utc": (
            utc_now_iso()
        ),
    }

    try:
        parsed = parse_diagnosis(
            raw_response
        )

    except Exception as exc:
        return {
            **common,
            "status": (
                "parse_error"
            ),
            "diagnosis": None,
            "confidence": None,
            "explanation": None,
            "parser_error_type": (
                type(
                    exc
                ).__name__
            ),
            "parser_error_message": (
                str(
                    exc
                )
            ),
        }

    return {
        **common,
        "status": (
            "ok"
        ),
        "diagnosis": (
            parsed.diagnosis.value
        ),
        "confidence": (
            parsed.confidence
        ),
        "explanation": (
            parsed.explanation
        ),
        "parser_error_type": None,
        "parser_error_message": None,
    }


async def execute_generation(
    *,
    adapter: ExperimentalOllamaAdapter,
    case: Mapping[str, Any],
    generation_id: int,
    technical_errors_path: Path,
) -> dict[str, Any]:
    """Execute one required generation with technical retry handling."""

    case_id = str(
        case[
            "case_id"
        ]
    )

    context = str(
        case[
            "model_context"
        ]
    )

    expected_context_hash = str(
        case[
            "model_context_sha256"
        ]
    )

    actual_context_hash = (
        sha256_text(
            context
        )
    )

    if (
        actual_context_hash
        != expected_context_hash
    ):
        raise RuntimeError(
            "Persisted model context hash mismatch "
            f"for {case_id}."
        )

    for attempt in range(
        1,
        TECHNICAL_MAX_ATTEMPTS + 1,
    ):
        started = (
            time.perf_counter()
        )

        try:
            raw_response = (
                adapter.generate(
                    DIAGNOSTIC_PROMPT,
                    context,
                    SYSTEM_PROMPT,
                )
            )

        except Exception as exc:
            latency = (
                time.perf_counter()
                - started
            )

            append_jsonl(
                technical_errors_path,
                {
                    "schema_version": (
                        "A8.5-technical-error-v1"
                    ),
                    "case_id": (
                        case_id
                    ),
                    "manifest_index": (
                        int(
                            case[
                                "manifest_index"
                            ]
                        )
                    ),
                    "evidence_condition": (
                        str(
                            case[
                                "evidence_condition"
                            ]
                        )
                    ),
                    "generation_id": (
                        generation_id
                    ),
                    "attempt": (
                        attempt
                    ),
                    "max_attempts": (
                        TECHNICAL_MAX_ATTEMPTS
                    ),
                    "model": (
                        MAIN_EXPERIMENT.model
                    ),
                    "temperature": (
                        MAIN_EXPERIMENT.temperature
                    ),
                    "latency_seconds": (
                        latency
                    ),
                    "error_type": (
                        type(
                            exc
                        ).__name__
                    ),
                    "error_message": (
                        str(
                            exc
                        )
                    ),
                    "occurred_at_utc": (
                        utc_now_iso()
                    ),
                },
            )

            if (
                attempt
                >= TECHNICAL_MAX_ATTEMPTS
            ):
                raise RuntimeError(
                    "Technical generation failure after "
                    f"{TECHNICAL_MAX_ATTEMPTS} attempts for "
                    f"{case_id}, generation "
                    f"{generation_id}."
                ) from exc

            await asyncio.sleep(
                TECHNICAL_RETRY_BACKOFF_SECONDS
            )

            continue

        latency = (
            time.perf_counter()
            - started
        )

        return (
            generation_record_from_response(
                case=case,
                generation_id=(
                    generation_id
                ),
                raw_response=(
                    raw_response
                ),
                latency_seconds=(
                    latency
                ),
                attempt=attempt,
            )
        )

    raise AssertionError(
        "Unreachable generation retry state"
    )


def scenario_case_records(
    *,
    case_index: Mapping[str, Mapping[str, Any]],
    manifest_index: int,
) -> list[dict[str, Any]]:
    """Return one scenario's Q cases in frozen EvidenceCondition order."""

    matches = [
        dict(
            record
        )
        for record in case_index.values()
        if int(
            record[
                "manifest_index"
            ]
        )
        == manifest_index
    ]

    if not matches:
        return []

    order = {
        condition.value: index
        for index, condition in enumerate(
            EvidenceCondition
        )
    }

    matches.sort(
        key=lambda record: order[
            str(
                record[
                    "evidence_condition"
                ]
            )
        ]
    )

    return matches


async def run_experiment(
    *,
    run_dir: Path,
    max_scenarios: int | None,
    prepare_only: bool,
) -> int:
    """Execute or resume the selected A8.5 experiment run."""

    selected_entries = (
        selected_manifest_entries(
            max_scenarios=(
                max_scenarios
            )
        )
    )

    # Ensure the human-readable frozen manifest artefact exists and matches
    # the current frozen manifest definition.
    write_manifest(
        DEFAULT_MANIFEST_PATH
    )

    git = git_provenance()

    if git[
        "dirty"
    ]:
        raise RuntimeError(
            "Refusing to execute the main experiment from a "
            "dirty Git working tree. Commit the research code "
            "before collecting experimental observations."
        )

    db = await get_database_service()

    database_name = (
        await verify_experiment_database(
            db
        )
    )

    paths = ensure_run_directory(
        run_dir
    )

    metadata = build_run_metadata(
        selected_entries=(
            selected_entries
        ),
        git=git,
        database_name=(
            database_name
        ),
    )

    write_immutable_json(
        paths.metadata,
        metadata,
    )

    case_records = load_jsonl(
        paths.cases
    )

    case_index = (
        verify_existing_case_groups(
            records=case_records,
            selected_entries=(
                selected_entries
            ),
        )
    )

    generation_records = (
        load_jsonl(
            paths.generations
        )
    )

    completed = (
        completed_generation_keys(
            generation_records
        )
    )

    print(
        "=" * 80
    )
    print(
        "A8.5 DIAGNOSTIC-RELIABILITY MAIN EXPERIMENT"
    )
    print(
        "=" * 80
    )
    print(
        f"Run directory: {paths.run_dir}"
    )
    print(
        f"Database: {database_name}"
    )
    print(
        f"Model: {MAIN_EXPERIMENT.model}"
    )
    print(
        f"Temperature: {MAIN_EXPERIMENT.temperature}"
    )
    print(
        f"Selected scenarios: {len(selected_entries)}"
    )
    print(
        "Expected evidence cases: "
        f"{len(selected_entries) * len(EvidenceCondition)}"
    )
    print(
        "Expected generations: "
        f"{len(selected_entries) * len(EvidenceCondition) * MAIN_EXPERIMENT.generations_per_evidence_case}"
    )
    print(
        f"Existing cases: {len(case_index)}"
    )
    print(
        f"Completed generations: {len(completed)}"
    )
    print(
        f"Prepare only: {prepare_only}"
    )

    async with db.get_session() as session:
        device = (
            await provision_experimental_device(
                session
            )
        )

        device_int_id = (
            device.id
        )

        device_id = (
            device.device_id
        )

    if (
        device_id
        != RESEARCH_DEVICE_ID
    ):
        raise RuntimeError(
            "Unexpected research device identity."
        )

    adapter: (
        ExperimentalOllamaAdapter
        | None
    ) = None

    if not prepare_only:
        adapter = (
            ExperimentalOllamaAdapter(
                temperature=(
                    MAIN_EXPERIMENT
                    .temperature
                ),
                model=(
                    MAIN_EXPERIMENT
                    .model
                ),
            )
        )

        if not adapter.check_health():
            raise RuntimeError(
                "Ollama service is not reachable."
            )

    total_selected = len(
        selected_entries
    )

    for scenario_position, entry in enumerate(
        selected_entries,
        start=1,
    ):
        existing_scenario_cases = (
            scenario_case_records(
                case_index=case_index,
                manifest_index=(
                    entry.manifest_index
                ),
            )
        )

        if not existing_scenario_cases:
            print()
            print(
                f"[{scenario_position}/{total_selected}] "
                f"Preparing manifest index "
                f"{entry.manifest_index}: "
                f"{entry.fault} / "
                f"{entry.scenario_family} / "
                f"instance {entry.instance}"
            )

            prepared = (
                await prepare_scenario_cases(
                    db=db,
                    device_id=device_id,
                    device_int_id=(
                        device_int_id
                    ),
                    entry=entry,
                )
            )

            append_jsonl_batch(
                paths.cases,
                prepared,
            )

            for record in prepared:
                case_index[
                    str(
                        record[
                            "case_id"
                        ]
                    )
                ] = (
                    record
                )

            existing_scenario_cases = (
                prepared
            )

            print(
                "  Prepared and persisted "
                f"{len(prepared)} paired Q cases."
            )

        else:
            print()
            print(
                f"[{scenario_position}/{total_selected}] "
                f"Manifest index {entry.manifest_index}: "
                "using previously persisted Q cases."
            )

        if prepare_only:
            continue

        if adapter is None:
            raise AssertionError(
                "Adapter was not initialised."
            )

        for case in existing_scenario_cases:
            case_id = str(
                case[
                    "case_id"
                ]
            )

            condition = str(
                case[
                    "evidence_condition"
                ]
            )

            print(
                f"  Case {condition}"
            )

            for generation_id in range(
                1,
                MAIN_EXPERIMENT
                .generations_per_evidence_case
                + 1,
            ):
                identity = (
                    generation_key(
                        case_id=case_id,
                        generation_id=(
                            generation_id
                        ),
                    )
                )

                if identity in completed:
                    print(
                        f"    Generation "
                        f"{generation_id}: "
                        "already complete"
                    )
                    continue

                print(
                    f"    Generation "
                    f"{generation_id}: running..."
                )

                record = (
                    await execute_generation(
                        adapter=adapter,
                        case=case,
                        generation_id=(
                            generation_id
                        ),
                        technical_errors_path=(
                            paths.technical_errors
                        ),
                    )
                )

                append_jsonl(
                    paths.generations,
                    record,
                )

                completed.add(
                    identity
                )

                print(
                    f"      status="
                    f"{record['status']}",
                    end="",
                )

                if (
                    record[
                        "status"
                    ]
                    == "ok"
                ):
                    print(
                        ", diagnosis="
                        f"{record['diagnosis']}, "
                        "confidence="
                        f"{record['confidence']:.3f}, "
                        "latency="
                        f"{record['latency_seconds']:.2f}s"
                    )

                else:
                    print(
                        ", parse_error="
                        f"{record['parser_error_type']}, "
                        "latency="
                        f"{record['latency_seconds']:.2f}s"
                    )

    final_case_records = (
        load_jsonl(
            paths.cases
        )
    )

    final_generation_records = (
        load_jsonl(
            paths.generations
        )
    )

    final_case_index = (
        verify_existing_case_groups(
            records=final_case_records,
            selected_entries=(
                selected_entries
            ),
        )
    )

    final_completed = (
        completed_generation_keys(
            final_generation_records
        )
    )

    expected_cases = (
        len(
            selected_entries
        )
        * len(
            EvidenceCondition
        )
    )

    expected_generations = (
        expected_cases
        * MAIN_EXPERIMENT
        .generations_per_evidence_case
    )

    print()
    print(
        "=" * 80
    )
    print(
        "RUN STATUS"
    )
    print(
        "=" * 80
    )
    print(
        f"Persisted cases: "
        f"{len(final_case_index)}/"
        f"{expected_cases}"
    )

    if prepare_only:
        print(
            "Generation execution skipped "
            "because --prepare-only was used."
        )

    else:
        print(
            f"Completed generations: "
            f"{len(final_completed)}/"
            f"{expected_generations}"
        )

        if (
            len(
                final_completed
            )
            != expected_generations
        ):
            raise RuntimeError(
                "Experiment stopped without completing "
                "all selected generations."
            )

    if (
        len(
            final_case_index
        )
        != expected_cases
    ):
        raise RuntimeError(
            "Experiment stopped without preparing "
            "all selected evidence cases."
        )

    print(
        "\nA8.5 run completed successfully."
    )

    return 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run or resume the frozen A8.5 "
            "diagnostic-reliability experiment."
        )
    )

    parser.add_argument(
        "--run-dir",
        type=Path,
        default=(
            DEFAULT_MAIN_RUN_DIR
        ),
        help=(
            "Directory containing durable run artefacts."
        ),
    )

    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=None,
        help=(
            "Run only the first N frozen manifest scenarios. "
            "Use this for a separate dry-run directory."
        ),
    )

    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help=(
            "Prepare and persist validation-controlled evidence "
            "cases without making LLM calls."
        ),
    )

    return parser.parse_args()


def main() -> int:
    """Command-line entry point."""

    args = parse_args()

    return asyncio.run(
        run_experiment(
            run_dir=args.run_dir,
            max_scenarios=(
                args.max_scenarios
            ),
            prepare_only=(
                args.prepare_only
            ),
        )
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )