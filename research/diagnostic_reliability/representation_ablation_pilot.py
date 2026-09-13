"""A8.4 representation ablation pilot.

This pilot investigates whether the raw sequential telemetry representation
contributes to the poor diagnostic discrimination observed in the A8.4d
confirmation pilot.

Two model-visible representations are compared while holding all other
experimental components constant:

A. RAW
   Existing leakage-controlled context containing the full sequential
   telemetry table.

B. RAW_PLUS_SUMMARY
   The same raw context plus a deterministic arithmetic telemetry summary.

The summary contains no diagnostic labels, private ground truth, scenario
identifiers, class-specific thresholds, or qualitative diagnostic
interpretation.

The real HOMEPOT A→E Validation Envelope is evaluated once per scenario using
the existing production-compatible raw context. The resulting upstream
validation/readiness information is then held fixed across both downstream
representation conditions.

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
from .representation import append_telemetry_summary
from .runner import run_repeated_generations
from .scenarios import generate_scenario
from .taxonomy import (
    EvidenceCondition,
    FaultLabel,
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

PILOT_FAULTS = (
    FaultLabel.NORMAL,
    FaultLabel.CPU_SATURATION,
    FaultLabel.MEMORY_PRESSURE,
    FaultLabel.NETWORK_DEGRADATION,
    FaultLabel.APPLICATION_FAILURE,
)

REPRESENTATIONS = (
    "RAW",
    "RAW_PLUS_SUMMARY",
)


def _summarise_case(
    results,
    ground_truth: FaultLabel,
) -> dict:
    """Summarise repeated generations for one representation condition."""

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
    """Aggregate descriptive metrics for one representation."""

    summaries = [
        case["representations"][
            representation
        ]
        for case in cases
    ]

    total_generations = (
        len(summaries)
        * N_GENERATIONS
    )

    total_correct = sum(
        summary[
            "correct_generations"
        ]
        for summary in summaries
    )

    modal_correct_cases = sum(
        summary[
            "modal_diagnosis"
        ]
        == case[
            "research_only"
        ][
            "private_ground_truth"
        ]
        for case, summary
        in zip(
            cases,
            summaries,
        )
    )

    cases_with_disagreement = sum(
        summary[
            "modal_fraction"
        ]
        < 1.0
        for summary in summaries
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
            len(summaries)
        ),
        "cases_with_any_disagreement": (
            cases_with_disagreement
        ),
        "mean_modal_fraction": (
            sum(
                summary[
                    "modal_fraction"
                ]
                for summary
                in summaries
            )
            / len(summaries)
        ),
        "mean_diagnostic_entropy": (
            sum(
                summary[
                    "diagnostic_entropy"
                ]
                for summary
                in summaries
            )
            / len(summaries)
        ),
        "mean_self_confidence": (
            sum(
                summary[
                    "mean_self_confidence"
                ]
                for summary
                in summaries
            )
            / len(summaries)
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
    """Run the raw-versus-summary representation ablation."""

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
            "A8.4_representation_ablation"
        ),
        "temperature": TEMPERATURE,
        "n_generations_per_case_per_representation": (
            N_GENERATIONS
        ),
        "evidence_condition": (
            EvidenceCondition
            .COMPLETE_FRESH
            .value
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

        scenario = (
            apply_evidence_condition(
                base,
                EvidenceCondition
                .COMPLETE_FRESH,
            )
        )

        print(
            f"\nPreparing "
            f"{fault.value} "
            f"({scenario.scenario_family})"
        )

        # The upstream Validation Envelope remains attached to the existing
        # raw production-compatible context. Representation is varied only
        # after upstream validation has completed.
        pre_validation_context = (
            build_model_context(
                scenario
            )
        )

        async with db.get_session() as session:
            await materialise_scenario(
                session,
                scenario,
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
                f"{scenario.scenario_id} "
                "was not actionable."
            )

        if (
            envelope["failed_gate_id"]
            is not None
        ):
            raise RuntimeError(
                f"COMPLETE_FRESH case "
                f"{scenario.scenario_id} "
                "failed the Validation "
                f"Envelope at "
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

        raw_context = (
            build_model_context(
                scenario,
                validation=(
                    visible_validation
                ),
            )
        )

        summary_context = (
            append_telemetry_summary(
                raw_context,
                scenario,
            )
        )

        contexts = {
            "RAW": raw_context,
            "RAW_PLUS_SUMMARY": (
                summary_context
            ),
        }

        case_report = {
            "research_only": {
                "scenario_id": (
                    scenario.scenario_id
                ),
                "scenario_family": (
                    scenario.scenario_family
                ),
                "private_ground_truth": (
                    scenario.fault.value
                ),
                "seed": (
                    scenario.seed
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
            "model_visible_validation": (
                visible_validation
            ),
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
            context = contexts[
                representation
            ]

            leakage = (
                _leakage_checks(
                    context=context,
                    scenario=scenario,
                )
            )

            case_report[
                "leakage_checks"
            ][representation] = leakage

            if not all(
                leakage.values()
            ):
                raise RuntimeError(
                    "Leakage-control failure "
                    f"for "
                    f"{scenario.scenario_id} "
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
                        scenario.fault
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

    raw_summary = (
        report["summary"][
            "RAW"
        ]
    )

    extended_summary = (
        report["summary"][
            "RAW_PLUS_SUMMARY"
        ]
    )

    report["comparison"] = {
        "generation_accuracy_delta": (
            extended_summary[
                "descriptive_generation_accuracy"
            ]
            -
            raw_summary[
                "descriptive_generation_accuracy"
            ]
        ),
        "modal_correct_cases_delta": (
            extended_summary[
                "modal_correct_cases"
            ]
            -
            raw_summary[
                "modal_correct_cases"
            ]
        ),
        "mean_entropy_delta": (
            extended_summary[
                "mean_diagnostic_entropy"
            ]
            -
            raw_summary[
                "mean_diagnostic_entropy"
            ]
        ),
        "mean_modal_fraction_delta": (
            extended_summary[
                "mean_modal_fraction"
            ]
            -
            raw_summary[
                "mean_modal_fraction"
            ]
        ),
        "mean_self_confidence_delta": (
            extended_summary[
                "mean_self_confidence"
            ]
            -
            raw_summary[
                "mean_self_confidence"
            ]
        ),
    }

    print(
        "\n"
        + "=" * 80
    )

    print(
        "A8.4 REPRESENTATION ABLATION PILOT"
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

        raw = (
            case[
                "representations"
            ][
                "RAW"
            ]
        )

        extended = (
            case[
                "representations"
            ][
                "RAW_PLUS_SUMMARY"
            ]
        )

        print(
            f"{truth}: "
            f"RAW modal="
            f"{raw['modal_diagnosis']} "
            f"accuracy="
            f"{raw['generation_accuracy']:.3f} "
            f"entropy="
            f"{raw['diagnostic_entropy']:.3f}"
        )

        print(
            f"{'':4}"
            f"SUMMARY modal="
            f"{extended['modal_diagnosis']} "
            f"accuracy="
            f"{extended['generation_accuracy']:.3f} "
            f"entropy="
            f"{extended['diagnostic_entropy']:.3f}"
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
        "\nRAW_PLUS_SUMMARY minus RAW"
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
        "representation plausibly contributes "
        "to the observed diagnostic difficulty."
    )

    print(
        "\nA8.4 REPRESENTATION ABLATION: COMPLETE"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(
            main()
        )
    )