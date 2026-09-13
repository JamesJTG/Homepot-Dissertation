"""A8.4d confirmation pilot for the refined diagnostic task.

This pilot checks whether the fixed semantic diagnostic taxonomy resolves the
task-definition ambiguity identified during the initial temperature pilot.

It is deliberately small and is not intended to estimate final diagnostic
performance.

Design:

- one representative COMPLETE_FRESH scenario per injected fault class;
- provisional temperature 0.4;
- three repeated generations per scenario;
- real HOMEPOT A→E Validation Envelope;
- leakage-controlled model context;
- fixed-support diagnostic entropy.

The pilot asks whether the model now demonstrates meaningful discrimination
between the experimental diagnostic classes while retaining some stochastic
variation for downstream uncertainty measurement.
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


def _summarise_case(
    results,
    ground_truth: FaultLabel,
) -> dict:
    """Summarise repeated generations for one confirmation case."""

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


async def main() -> int:
    """Run the refined-task confirmation pilot."""

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
            "A8.4d_task_validity_confirmation"
        ),
        "temperature": TEMPERATURE,
        "n_generations_per_case": (
            N_GENERATIONS
        ),
        "evidence_condition": (
            EvidenceCondition
            .COMPLETE_FRESH
            .value
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
            f"\nPreparing {fault.value} "
            f"({scenario.scenario_family})"
        )

        # Gate C validates the model context before envelope-derived
        # validation information exists.
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

        actual_gates = set(
            envelope["gates"]
        )

        if (
            actual_gates
            != expected_gates
        ):
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

        model_context = (
            build_model_context(
                scenario,
                validation=(
                    visible_validation
                ),
            )
        )

        leakage_checks = {
            "scenario_id_absent": (
                scenario.scenario_id
                not in model_context
            ),
            "scenario_family_absent": (
                scenario.scenario_family
                not in model_context
            ),
            "private_fault_metadata_absent": (
                (
                    f"fault: "
                    f"{scenario.fault.value}"
                )
                not in model_context
                and
                (
                    f"fault="
                    f"{scenario.fault.value}"
                )
                not in model_context
            ),
        }

        if not all(
            leakage_checks.values()
        ):
            raise RuntimeError(
                "Leakage-control failure "
                f"for "
                f"{scenario.scenario_id}"
            )

        print(
            f"  envelope="
            f"{envelope['trust_mode']} "
            f"score="
            f"{envelope['trust_score']:.3f}"
        )

        print(
            f"  running "
            f"{N_GENERATIONS} "
            f"generations at "
            f"temperature="
            f"{TEMPERATURE}..."
        )

        results = (
            run_repeated_generations(
                adapter.generate,
                context=model_context,
                n_generations=(
                    N_GENERATIONS
                ),
            )
        )

        case_summary = (
            _summarise_case(
                results,
                ground_truth=(
                    scenario.fault
                ),
            )
        )

        report["cases"].append(
            {
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
                "leakage_checks": (
                    leakage_checks
                ),
                "result": (
                    case_summary
                ),
            }
        )

    total_generations = (
        len(PILOT_FAULTS)
        * N_GENERATIONS
    )

    total_correct = sum(
        case["result"][
            "correct_generations"
        ]
        for case in report["cases"]
    )

    cases_with_disagreement = sum(
        case["result"][
            "modal_fraction"
        ]
        < 1.0
        for case in report["cases"]
    )

    modal_correct_cases = sum(
        (
            case["result"][
                "modal_diagnosis"
            ]
            ==
            case["research_only"][
                "private_ground_truth"
            ]
        )
        for case in report["cases"]
    )

    report["summary"] = {
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
            len(PILOT_FAULTS)
        ),
        "cases_with_any_disagreement": (
            cases_with_disagreement
        ),
        "mean_modal_fraction": (
            sum(
                case["result"][
                    "modal_fraction"
                ]
                for case
                in report["cases"]
            )
            / len(report["cases"])
        ),
        "mean_diagnostic_entropy": (
            sum(
                case["result"][
                    "diagnostic_entropy"
                ]
                for case
                in report["cases"]
            )
            / len(report["cases"])
        ),
        "mean_self_confidence": (
            sum(
                case["result"][
                    "mean_self_confidence"
                ]
                for case
                in report["cases"]
            )
            / len(report["cases"])
        ),
    }

    print(
        "\n"
        + "=" * 72
    )
    print(
        "TASK-VALIDITY CONFIRMATION PILOT"
    )
    print(
        "=" * 72
    )

    print(
        json.dumps(
            report,
            indent=2,
            default=str,
        )
    )

    print(
        "\nCASE SUMMARY"
    )
    print(
        "-" * 72
    )

    for case in report["cases"]:
        research = (
            case["research_only"]
        )

        result = (
            case["result"]
        )

        print(
            f"{research['private_ground_truth']}: "
            f"modal="
            f"{result['modal_diagnosis']}, "
            f"modal_fraction="
            f"{result['modal_fraction']:.3f}, "
            f"entropy="
            f"{result['diagnostic_entropy']:.3f}, "
            f"generation_accuracy="
            f"{result['generation_accuracy']:.3f}, "
            f"mean_confidence="
            f"{result['mean_self_confidence']:.3f}"
        )

    summary = report[
        "summary"
    ]

    print(
        "\nOVERALL"
    )
    print(
        "-" * 72
    )

    print(
        "Descriptive generation accuracy: "
        f"{summary['descriptive_generation_accuracy']:.3f}"
    )

    print(
        "Modal-correct cases: "
        f"{summary['modal_correct_cases']}/"
        f"{summary['total_cases']}"
    )

    print(
        "Cases with disagreement: "
        f"{summary['cases_with_any_disagreement']}/"
        f"{summary['total_cases']}"
    )

    print(
        "Mean modal fraction: "
        f"{summary['mean_modal_fraction']:.3f}"
    )

    print(
        "Mean fixed-support entropy: "
        f"{summary['mean_diagnostic_entropy']:.3f}"
    )

    print(
        "Mean self-confidence: "
        f"{summary['mean_self_confidence']:.3f}"
    )

    print(
        "\nThis pilot is descriptive only. "
        "It must not be interpreted as a "
        "final performance estimate."
    )

    print(
        "\nA8.4d CONFIRMATION PILOT: COMPLETE"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(
            main()
        )
    )