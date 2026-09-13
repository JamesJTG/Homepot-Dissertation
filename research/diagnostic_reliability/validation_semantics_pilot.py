"""A8.4 validation-semantics ablation pilot.

This pilot investigates whether ambiguity in the meaning of HOMEPOT validation
status contributes to poor diagnostic discrimination.

Two system-prompt conditions are compared:

BASELINE
    The current experimental SYSTEM_PROMPT unchanged.

VALIDATION_CLARIFIED
    The same system prompt plus an explicit architectural definition stating
    that validation status describes evidence readiness, not operational
    health or diagnostic correctness.

All other experimental components are held constant.

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

PROMPT_CONDITIONS = (
    "BASELINE",
    "VALIDATION_CLARIFIED",
)


VALIDATION_SEMANTICS = """
Validation status describes the readiness and integrity of the supplied
evidence for inference. It does not describe whether the device itself is
healthy or faulty.

A grounded trust mode, trust_score of 1.0, actionable status, or absence of a
failed validation gate means that the supplied evidence passed the configured
readiness checks. These values must not be interpreted as evidence for NORMAL
operation and do not imply that any generated diagnosis is correct.

Determine operational health from the telemetry evidence itself. Use
validation status only to judge whether that evidence is suitable to support
diagnostic inference.
""".strip()


VALIDATION_CLARIFIED_SYSTEM_PROMPT = (
    SYSTEM_PROMPT.rstrip()
    + "\n\n"
    + VALIDATION_SEMANTICS
)


def _run_generations(
    adapter: ExperimentalOllamaAdapter,
    *,
    context: str,
    system_prompt: str,
) -> list[dict]:
    """Run repeated generations using an explicitly selected system prompt."""

    results = []

    for generation_id in range(
        1,
        N_GENERATIONS + 1,
    ):
        raw = adapter.generate(
            DIAGNOSTIC_PROMPT,
            context,
            system_prompt,
        )

        parsed = parse_diagnosis(
            raw
        )

        results.append(
            {
                "generation_id": (
                    generation_id
                ),
                "diagnosis": (
                    parsed.diagnosis
                ),
                "confidence": (
                    parsed.confidence
                ),
                "explanation": (
                    parsed.explanation
                ),
            }
        )

    return results


def _summarise_case(
    generations: list[dict],
    *,
    ground_truth: FaultLabel,
) -> dict:
    """Summarise one prompt-condition result."""

    labels = [
        item["diagnosis"]
        for item in generations
    ]

    confidences = [
        item["confidence"]
        for item in generations
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
        label.value
        == ground_truth.value
        for label in labels
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
            / len(generations)
        ),
        "generations": [
            {
                "generation_id": (
                    item[
                        "generation_id"
                    ]
                ),
                "diagnosis": (
                    item[
                        "diagnosis"
                    ].value
                ),
                "confidence": (
                    item[
                        "confidence"
                    ]
                ),
                "correct": (
                    item[
                        "diagnosis"
                    ].value
                    == ground_truth.value
                ),
                "explanation": (
                    item[
                        "explanation"
                    ]
                ),
            }
            for item in generations
        ],
    }


def _aggregate_condition(
    cases: list[dict],
    condition: str,
) -> dict:
    """Aggregate descriptive metrics for one prompt condition."""

    total_generations = (
        len(cases)
        * N_GENERATIONS
    )

    total_correct = sum(
        case[
            "conditions"
        ][condition][
            "correct_generations"
        ]
        for case in cases
    )

    modal_correct_cases = sum(
        (
            case[
                "conditions"
            ][condition][
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

    cases_with_disagreement = sum(
        case[
            "conditions"
        ][condition][
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
            cases_with_disagreement
        ),
        "mean_modal_fraction": (
            sum(
                case[
                    "conditions"
                ][condition][
                    "modal_fraction"
                ]
                for case in cases
            )
            / len(cases)
        ),
        "mean_diagnostic_entropy": (
            sum(
                case[
                    "conditions"
                ][condition][
                    "diagnostic_entropy"
                ]
                for case in cases
            )
            / len(cases)
        ),
        "mean_self_confidence": (
            sum(
                case[
                    "conditions"
                ][condition][
                    "mean_self_confidence"
                ]
                for case in cases
            )
            / len(cases)
        ),
    }


async def main() -> int:
    """Run the validation-semantics ablation pilot."""

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

    system_prompts = {
        "BASELINE": (
            SYSTEM_PROMPT
        ),
        "VALIDATION_CLARIFIED": (
            VALIDATION_CLARIFIED_SYSTEM_PROMPT
        ),
    }

    report = {
        "pilot": (
            "A8.4_validation_semantics_ablation"
        ),
        "temperature": (
            TEMPERATURE
        ),
        "n_generations_per_case_per_condition": (
            N_GENERATIONS
        ),
        "evidence_condition": (
            EvidenceCondition
            .COMPLETE_FRESH
            .value
        ),
        "representation": (
            "RAW"
        ),
        "conditions": list(
            PROMPT_CONDITIONS
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
            envelope[
                "failed_gate_id"
            ]
            is not None
        ):
            raise RuntimeError(
                f"COMPLETE_FRESH case "
                f"{scenario.scenario_id} "
                "failed the Validation Envelope."
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
            },
            "leakage_checks": (
                leakage_checks
            ),
            "conditions": {},
        }

        for condition in PROMPT_CONDITIONS:
            print(
                f"  {condition}: "
                f"running "
                f"{N_GENERATIONS} "
                f"generations at "
                f"temperature="
                f"{TEMPERATURE}..."
            )

            generations = (
                _run_generations(
                    adapter,
                    context=(
                        model_context
                    ),
                    system_prompt=(
                        system_prompts[
                            condition
                        ]
                    ),
                )
            )

            case_report[
                "conditions"
            ][condition] = (
                _summarise_case(
                    generations,
                    ground_truth=(
                        scenario.fault
                    ),
                )
            )

        report[
            "cases"
        ].append(
            case_report
        )

    report["summary"] = {
        condition: (
            _aggregate_condition(
                report["cases"],
                condition,
            )
        )
        for condition
        in PROMPT_CONDITIONS
    }

    baseline = (
        report[
            "summary"
        ][
            "BASELINE"
        ]
    )

    clarified = (
        report[
            "summary"
        ][
            "VALIDATION_CLARIFIED"
        ]
    )

    report["comparison"] = {
        "generation_accuracy_delta": (
            clarified[
                "descriptive_generation_accuracy"
            ]
            -
            baseline[
                "descriptive_generation_accuracy"
            ]
        ),
        "modal_correct_cases_delta": (
            clarified[
                "modal_correct_cases"
            ]
            -
            baseline[
                "modal_correct_cases"
            ]
        ),
        "mean_entropy_delta": (
            clarified[
                "mean_diagnostic_entropy"
            ]
            -
            baseline[
                "mean_diagnostic_entropy"
            ]
        ),
        "mean_modal_fraction_delta": (
            clarified[
                "mean_modal_fraction"
            ]
            -
            baseline[
                "mean_modal_fraction"
            ]
        ),
        "mean_self_confidence_delta": (
            clarified[
                "mean_self_confidence"
            ]
            -
            baseline[
                "mean_self_confidence"
            ]
        ),
    }

    print(
        "\n"
        + "=" * 80
    )

    print(
        "A8.4 VALIDATION-SEMANTICS ABLATION PILOT"
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

    for case in report[
        "cases"
    ]:
        truth = (
            case[
                "research_only"
            ][
                "private_ground_truth"
            ]
        )

        baseline_case = (
            case[
                "conditions"
            ][
                "BASELINE"
            ]
        )

        clarified_case = (
            case[
                "conditions"
            ][
                "VALIDATION_CLARIFIED"
            ]
        )

        print(
            f"{truth}: "
            f"BASELINE modal="
            f"{baseline_case['modal_diagnosis']} "
            f"accuracy="
            f"{baseline_case['generation_accuracy']:.3f} "
            f"entropy="
            f"{baseline_case['diagnostic_entropy']:.3f}"
        )

        print(
            f"{'':4}"
            f"CLARIFIED modal="
            f"{clarified_case['modal_diagnosis']} "
            f"accuracy="
            f"{clarified_case['generation_accuracy']:.3f} "
            f"entropy="
            f"{clarified_case['diagnostic_entropy']:.3f}"
        )

    print(
        "\nAGGREGATE COMPARISON"
    )

    print(
        "-" * 80
    )

    for condition in PROMPT_CONDITIONS:
        summary = (
            report[
                "summary"
            ][condition]
        )

        print(
            f"{condition}: "
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
        "\nVALIDATION_CLARIFIED minus BASELINE"
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
        "validation-status ambiguity plausibly "
        "contributes to diagnostic error."
    )

    print(
        "\nA8.4 VALIDATION-SEMANTICS ABLATION: COMPLETE"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(
            main()
        )
    )