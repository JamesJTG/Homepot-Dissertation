"""A8.4 temporal-density ablation pilot.

This pilot investigates whether the density of sequential telemetry presented
to the original HOMEPOT experimental LLM contributes to poor diagnostic
discrimination.

Two downstream model-visible representations are compared:

FULL_RAW
    The complete original 24-observation telemetry sequence.

DOWNSAMPLED_RAW
    Eight deterministic, approximately evenly spaced observations selected
    from the same original sequence while preserving the first and final
    observations.

The controlled scenario generators are not modified.

The real HOMEPOT A→E Validation Envelope is evaluated against the complete
24-observation scenario. Its resulting validation/readiness information is
then held fixed across both downstream LLM conditions.

No telemetry values are averaged, interpolated, synthesised, or changed.

This is a small descriptive pilot, not a final performance experiment.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from datetime import datetime, timezone
import json

from homepot.database import get_database_service

from .context import build_model_context
from .evidence_conditions import apply_evidence_condition
from .ollama_adapter import ExperimentalOllamaAdapter
from .runner import run_repeated_generations
from .scenarios import generate_scenario
from .taxonomy import (
    EvidenceCondition,
    FaultLabel,
)
from .temporal_sampling import (
    downsample_scenario,
    evenly_spaced_indices,
)
from .uncertainty import (
    diagnosis_distribution,
    diagnostic_entropy,
    modal_fraction,
)
from .validation_integration import (
    RESEARCH_DEVICE_ID,
    envelope_summary,
    evaluate_validation_envelope,
    materialise_scenario,
    model_visible_validation,
    provision_experimental_device,
)


TEMPERATURE = 0.4
N_GENERATIONS = 3

FULL_OBSERVATIONS = 24
DOWNSAMPLED_OBSERVATIONS = 8

PILOT_FAULTS = (
    FaultLabel.NORMAL,
    FaultLabel.CPU_SATURATION,
    FaultLabel.MEMORY_PRESSURE,
    FaultLabel.NETWORK_DEGRADATION,
    FaultLabel.APPLICATION_FAILURE,
)

REPRESENTATIONS = (
    "FULL_RAW",
    "DOWNSAMPLED_RAW",
)


def _summarise_case(
    results,
    ground_truth: FaultLabel,
) -> dict:
    """Summarise repeated generations for one representation."""

    labels = [
        result.parsed.diagnosis
        for result in results
    ]

    confidences = [
        result.parsed.confidence
        for result in results
    ]

    counts = Counter(
        label.value
        for label in labels
    )

    modal_count = max(
        counts.values()
    )

    modal_labels = sorted(
        label
        for label, count
        in counts.items()
        if count == modal_count
    )

    modal_diagnosis = (
        modal_labels[0]
        if len(modal_labels) == 1
        else "TIE"
    )

    correct_generations = sum(
        result.parsed.diagnosis.value
        == ground_truth.value
        for result in results
    )

    return {
        "diagnosis_distribution": (
            diagnosis_distribution(
                labels
            )
        ),
        "diagnostic_entropy": (
            diagnostic_entropy(
                labels
            )
        ),
        "modal_fraction": (
            modal_fraction(
                labels
            )
        ),
        "modal_diagnosis": (
            modal_diagnosis
        ),
        "mean_self_confidence": (
            sum(confidences)
            / len(confidences)
        ),
        "correct_generations": (
            correct_generations
        ),
        "generation_accuracy": (
            correct_generations
            / len(results)
        ),
        "generations": [
            {
                "generation_id": (
                    result.generation_id
                ),
                "diagnosis": (
                    result.parsed.diagnosis.value
                ),
                "confidence": (
                    result.parsed.confidence
                ),
                "correct": (
                    result.parsed.diagnosis.value
                    == ground_truth.value
                ),
                "explanation": (
                    result.parsed.explanation
                ),
            }
            for result in results
        ],
    }


def _aggregate_representation(
    cases: list[dict],
    representation: str,
) -> dict:
    """Aggregate descriptive metrics for one telemetry representation."""

    total_generations = (
        len(cases)
        * N_GENERATIONS
    )

    total_correct = sum(
        case[
            "representations"
        ][representation][
            "correct_generations"
        ]
        for case in cases
    )

    modal_correct_cases = sum(
        (
            case[
                "representations"
            ][representation][
                "modal_diagnosis"
            ]
            ==
            case[
                "research_only"
            ][
                "private_ground_truth"
            ]
        )
        for case in cases
    )

    disagreement_cases = sum(
        case[
            "representations"
        ][representation][
            "modal_fraction"
        ]
        < 1.0
        for case in cases
    )

    return {
        "total_generations": (
            total_generations
        ),
        "correct_generations": (
            total_correct
        ),
        "descriptive_generation_accuracy": (
            total_correct
            / total_generations
        ),
        "modal_correct_cases": (
            modal_correct_cases
        ),
        "total_cases": (
            len(cases)
        ),
        "cases_with_any_disagreement": (
            disagreement_cases
        ),
        "mean_modal_fraction": (
            sum(
                case[
                    "representations"
                ][representation][
                    "modal_fraction"
                ]
                for case in cases
            )
            / len(cases)
        ),
        "mean_diagnostic_entropy": (
            sum(
                case[
                    "representations"
                ][representation][
                    "diagnostic_entropy"
                ]
                for case in cases
            )
            / len(cases)
        ),
        "mean_self_confidence": (
            sum(
                case[
                    "representations"
                ][representation][
                    "mean_self_confidence"
                ]
                for case in cases
            )
            / len(cases)
        ),
    }


def _leakage_checks(
    *,
    context: str,
    scenario,
) -> dict[str, bool]:
    """Check that private research metadata is absent."""

    return {
        "scenario_id_absent": (
            scenario.scenario_id
            not in context
        ),
        "scenario_family_absent": (
            scenario.scenario_family
            not in context
        ),
        "private_fault_metadata_absent": (
            (
                f"fault: "
                f"{scenario.fault.value}"
            )
            not in context
            and
            (
                f"fault="
                f"{scenario.fault.value}"
            )
            not in context
        ),
    }


async def main() -> int:
    """Run the full-versus-downsampled temporal-density pilot."""

    selected_indices = (
        evenly_spaced_indices(
            FULL_OBSERVATIONS,
            DOWNSAMPLED_OBSERVATIONS,
        )
    )

    db = await get_database_service()

    async with db.get_session() as session:
        device = (
            await provision_experimental_device(
                session
            )
        )

        device_int_id = device.id
        device_id = device.device_id

    if device_id != RESEARCH_DEVICE_ID:
        raise RuntimeError(
            "Unexpected research device identity."
        )

    adapter = ExperimentalOllamaAdapter(
        temperature=TEMPERATURE
    )

    if not adapter.check_health():
        raise RuntimeError(
            "Ollama service is not reachable."
        )

    report = {
        "pilot": (
            "A8.4_temporal_density_ablation"
        ),
        "temperature": (
            TEMPERATURE
        ),
        "n_generations_per_case_per_representation": (
            N_GENERATIONS
        ),
        "evidence_condition": (
            EvidenceCondition
            .COMPLETE_FRESH
            .value
        ),
        "full_observations": (
            FULL_OBSERVATIONS
        ),
        "downsampled_observations": (
            DOWNSAMPLED_OBSERVATIONS
        ),
        "selected_indices": list(
            selected_indices
        ),
        "representations": list(
            REPRESENTATIONS
        ),
        "cases": [],
    }

    for index, fault in enumerate(
        PILOT_FAULTS,
        start=1,
    ):
        base = generate_scenario(
            fault,
            instance=1,
            seed=20260920 + index,
            end_time=datetime.now(
                timezone.utc
            ),
            device_id=device_id,
        )

        full_scenario = (
            apply_evidence_condition(
                base,
                EvidenceCondition
                .COMPLETE_FRESH,
            )
        )

        if (
            len(full_scenario.telemetry)
            != FULL_OBSERVATIONS
        ):
            raise RuntimeError(
                f"Expected "
                f"{FULL_OBSERVATIONS} "
                "telemetry observations but "
                f"received "
                f"{len(full_scenario.telemetry)}."
            )

        reduced_scenario = (
            downsample_scenario(
                full_scenario,
                n_selected=(
                    DOWNSAMPLED_OBSERVATIONS
                ),
            )
        )

        print(
            f"\nPreparing "
            f"{fault.value} "
            f"({full_scenario.scenario_family})"
        )

        print(
            f"  observations: "
            f"{len(full_scenario.telemetry)} "
            f"-> "
            f"{len(reduced_scenario.telemetry)}"
        )

        print(
            f"  selected indices: "
            f"{selected_indices}"
        )

        # Upstream validation is deliberately evaluated only against the
        # complete original scenario.
        pre_validation_context = (
            build_model_context(
                full_scenario
            )
        )

        async with db.get_session() as session:
            await materialise_scenario(
                session,
                full_scenario,
                device_int_id=(
                    device_int_id
                ),
            )

            envelope_result = (
                await evaluate_validation_envelope(
                    session,
                    device_id=device_id,
                    device_int_id=(
                        device_int_id
                    ),
                    assembled_context=(
                        pre_validation_context
                    ),
                )
            )

        envelope = envelope_summary(
            envelope_result
        )

        if not envelope["actionable"]:
            raise RuntimeError(
                f"COMPLETE_FRESH case "
                f"{full_scenario.scenario_id} "
                "was not actionable."
            )

        if (
            envelope["failed_gate_id"]
            is not None
        ):
            raise RuntimeError(
                f"COMPLETE_FRESH case "
                f"{full_scenario.scenario_id} "
                "failed the Validation "
                "Envelope at "
                f"{envelope['failed_gate_id']}."
            )

        expected_gates = {
            "A",
            "B",
            "C",
            "D",
            "E",
        }

        if set(
            envelope["gates"]
        ) != expected_gates:
            raise RuntimeError(
                "COMPLETE_FRESH case did "
                "not evaluate the complete "
                "A→E Validation Envelope."
            )

        visible_validation = (
            model_visible_validation(
                envelope_result
            )
        )

        full_context = (
            build_model_context(
                full_scenario,
                validation=(
                    visible_validation
                ),
            )
        )

        reduced_context = (
            build_model_context(
                reduced_scenario,
                validation=(
                    visible_validation
                ),
            )
        )

        contexts = {
            "FULL_RAW": (
                full_context
            ),
            "DOWNSAMPLED_RAW": (
                reduced_context
            ),
        }

        case_report = {
            "research_only": {
                "scenario_id": (
                    full_scenario.scenario_id
                ),
                "scenario_family": (
                    full_scenario.scenario_family
                ),
                "private_ground_truth": (
                    full_scenario.fault.value
                ),
                "seed": (
                    full_scenario.seed
                ),
            },
            "validation": {
                "trust_mode": (
                    envelope[
                        "trust_mode"
                    ]
                ),
                "trust_score": (
                    envelope[
                        "trust_score"
                    ]
                ),
                "actionable": (
                    envelope[
                        "actionable"
                    ]
                ),
                "failed_gate_id": (
                    envelope[
                        "failed_gate_id"
                    ]
                ),
                "passed_gate_ids": (
                    envelope[
                        "passed_gate_ids"
                    ]
                ),
            },
            "telemetry": {
                "full_count": (
                    len(
                        full_scenario.telemetry
                    )
                ),
                "downsampled_count": (
                    len(
                        reduced_scenario.telemetry
                    )
                ),
                "selected_indices": list(
                    selected_indices
                ),
                "first_timestamp_preserved": (
                    reduced_scenario.telemetry[
                        0
                    ].timestamp
                    ==
                    full_scenario.telemetry[
                        0
                    ].timestamp
                ),
                "last_timestamp_preserved": (
                    reduced_scenario.telemetry[
                        -1
                    ].timestamp
                    ==
                    full_scenario.telemetry[
                        -1
                    ].timestamp
                ),
            },
            "context_sizes": {
                name: {
                    "characters": (
                        len(context)
                    ),
                    "lines": (
                        len(
                            context.splitlines()
                        )
                    ),
                }
                for name, context
                in contexts.items()
            },
            "leakage_checks": {},
            "representations": {},
        }

        for representation in REPRESENTATIONS:
            context = (
                contexts[
                    representation
                ]
            )

            leakage = (
                _leakage_checks(
                    context=context,
                    scenario=(
                        full_scenario
                    ),
                )
            )

            case_report[
                "leakage_checks"
            ][representation] = (
                leakage
            )

            if not all(
                leakage.values()
            ):
                raise RuntimeError(
                    "Leakage-control failure "
                    f"for "
                    f"{full_scenario.scenario_id} "
                    f"under "
                    f"{representation}."
                )

            print(
                f"  {representation}: "
                f"{len(context)} chars, "
                f"{len(context.splitlines())} lines"
            )

            print(
                f"    running "
                f"{N_GENERATIONS} "
                f"generations at "
                f"temperature="
                f"{TEMPERATURE}..."
            )

            results = (
                run_repeated_generations(
                    adapter.generate,
                    context=context,
                    n_generations=(
                        N_GENERATIONS
                    ),
                )
            )

            case_report[
                "representations"
            ][representation] = (
                _summarise_case(
                    results,
                    ground_truth=(
                        full_scenario.fault
                    ),
                )
            )

        report["cases"].append(
            case_report
        )

    report["summary"] = {
        representation: (
            _aggregate_representation(
                report["cases"],
                representation,
            )
        )
        for representation
        in REPRESENTATIONS
    }

    full_summary = (
        report[
            "summary"
        ][
            "FULL_RAW"
        ]
    )

    reduced_summary = (
        report[
            "summary"
        ][
            "DOWNSAMPLED_RAW"
        ]
    )

    report["comparison"] = {
        "generation_accuracy_delta": (
            reduced_summary[
                "descriptive_generation_accuracy"
            ]
            -
            full_summary[
                "descriptive_generation_accuracy"
            ]
        ),
        "modal_correct_cases_delta": (
            reduced_summary[
                "modal_correct_cases"
            ]
            -
            full_summary[
                "modal_correct_cases"
            ]
        ),
        "mean_entropy_delta": (
            reduced_summary[
                "mean_diagnostic_entropy"
            ]
            -
            full_summary[
                "mean_diagnostic_entropy"
            ]
        ),
        "mean_modal_fraction_delta": (
            reduced_summary[
                "mean_modal_fraction"
            ]
            -
            full_summary[
                "mean_modal_fraction"
            ]
        ),
        "mean_self_confidence_delta": (
            reduced_summary[
                "mean_self_confidence"
            ]
            -
            full_summary[
                "mean_self_confidence"
            ]
        ),
    }

    print(
        "\n"
        + "=" * 80
    )

    print(
        "A8.4 TEMPORAL-DENSITY ABLATION PILOT"
    )

    print(
        "=" * 80
    )

    print(
        json.dumps(
            report,
            indent=2,
            default=str,
        )
    )

    print(
        "\nCASE COMPARISON"
    )

    print(
        "-" * 80
    )

    for case in report["cases"]:
        truth = (
            case[
                "research_only"
            ][
                "private_ground_truth"
            ]
        )

        full = (
            case[
                "representations"
            ][
                "FULL_RAW"
            ]
        )

        reduced = (
            case[
                "representations"
            ][
                "DOWNSAMPLED_RAW"
            ]
        )

        print(
            f"{truth}: "
            f"FULL modal="
            f"{full['modal_diagnosis']} "
            f"accuracy="
            f"{full['generation_accuracy']:.3f} "
            f"entropy="
            f"{full['diagnostic_entropy']:.3f}"
        )

        print(
            f"{'':4}"
            f"8-ROW modal="
            f"{reduced['modal_diagnosis']} "
            f"accuracy="
            f"{reduced['generation_accuracy']:.3f} "
            f"entropy="
            f"{reduced['diagnostic_entropy']:.3f}"
        )

    print(
        "\nAGGREGATE COMPARISON"
    )

    print(
        "-" * 80
    )

    for representation in REPRESENTATIONS:
        summary = (
            report[
                "summary"
            ][representation]
        )

        print(
            f"{representation}: "
            f"accuracy="
            f"{summary['descriptive_generation_accuracy']:.3f}, "
            f"modal_correct="
            f"{summary['modal_correct_cases']}/"
            f"{summary['total_cases']}, "
            f"mean_entropy="
            f"{summary['mean_diagnostic_entropy']:.3f}, "
            f"mean_modal_fraction="
            f"{summary['mean_modal_fraction']:.3f}, "
            f"mean_confidence="
            f"{summary['mean_self_confidence']:.3f}"
        )

    comparison = (
        report[
            "comparison"
        ]
    )

    print(
        "\nDOWNSAMPLED_RAW minus FULL_RAW"
    )

    print(
        "-" * 80
    )

    print(
        "Generation accuracy delta: "
        f"{comparison['generation_accuracy_delta']:+.3f}"
    )

    print(
        "Modal-correct cases delta: "
        f"{comparison['modal_correct_cases_delta']:+d}"
    )

    print(
        "Mean entropy delta: "
        f"{comparison['mean_entropy_delta']:+.3f}"
    )

    print(
        "Mean modal fraction delta: "
        f"{comparison['mean_modal_fraction_delta']:+.3f}"
    )

    print(
        "Mean self-confidence delta: "
        f"{comparison['mean_self_confidence_delta']:+.3f}"
    )

    print(
        "\nThis pilot is descriptive only. "
        "Its purpose is to test whether "
        "temporal input density plausibly "
        "contributes to the observed "
        "diagnostic difficulty."
    )

    print(
        "\nA8.4 TEMPORAL-DENSITY ABLATION: COMPLETE"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(
            main()
        )
    )