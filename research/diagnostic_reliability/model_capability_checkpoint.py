"""A8.4 controlled LLM model-capability checkpoint.

This checkpoint investigates whether the diagnostic difficulty observed during
the pilot is primarily associated with the capability of the original HOMEPOT
LLM.

Two locally available models are compared under otherwise identical
experimental conditions:

- llama3.2
- gemma3:12b

The comparison holds constant:

- controlled scenario;
- private ground truth;
- 24-observation raw telemetry representation;
- COMPLETE_FRESH evidence condition;
- real HOMEPOT A→E Validation Envelope;
- model-visible validation information;
- diagnostic task specification;
- validation-semantics clarification;
- generation temperature;
- number of repeated generations.

The validation-semantics clarification states only the architectural meaning
of HOMEPOT validation information. It does not reveal scenario ground truth,
fault labels, generator rules, thresholds, or expected diagnoses.

This is a small research-assurance checkpoint, not a final performance
benchmark and not a decision to replace HOMEPOT's configured production model.
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
from .prompt import SYSTEM_PROMPT
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

MODELS = (
    "llama3.2",
    "gemma3:12b",
)

PILOT_FAULTS = (
    FaultLabel.NORMAL,
    FaultLabel.CPU_SATURATION,
    FaultLabel.MEMORY_PRESSURE,
    FaultLabel.NETWORK_DEGRADATION,
    FaultLabel.APPLICATION_FAILURE,
)


CAPABILITY_SYSTEM_PROMPT = SYSTEM_PROMPT


def _summarise_generations(
    results,
    ground_truth: FaultLabel,
) -> dict:
    """Summarise repeated generations for one model and scenario."""

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
        for label, count in counts.items()
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


def _aggregate_model(
    cases: list[dict],
    model: str,
) -> dict:
    """Aggregate descriptive checkpoint metrics for one model."""

    total_generations = (
        len(cases)
        * N_GENERATIONS
    )

    total_correct = sum(
        case[
            "models"
        ][model][
            "correct_generations"
        ]
        for case in cases
    )

    modal_correct_cases = sum(
        (
            case[
                "models"
            ][model][
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
            "models"
        ][model][
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
                    "models"
                ][model][
                    "modal_fraction"
                ]
                for case in cases
            )
            / len(cases)
        ),
        "mean_diagnostic_entropy": (
            sum(
                case[
                    "models"
                ][model][
                    "diagnostic_entropy"
                ]
                for case in cases
            )
            / len(cases)
        ),
        "mean_self_confidence": (
            sum(
                case[
                    "models"
                ][model][
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
    """Confirm that private research metadata is absent from model context."""

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
                f"fault: {scenario.fault.value}"
                not in context
            )
            and
            (
                f"fault={scenario.fault.value}"
                not in context
            )
        ),
    }


async def main() -> int:
    """Run the controlled model-capability checkpoint."""

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

    adapters = {
        model: ExperimentalOllamaAdapter(
            temperature=TEMPERATURE,
            model=model,
        )
        for model in MODELS
    }

    for model, adapter in adapters.items():
        if not adapter.check_health():
            raise RuntimeError(
                f"Ollama service is not reachable "
                f"for model {model!r}."
            )

    report = {
        "checkpoint": (
            "A8.4_model_capability"
        ),
        "models": list(
            MODELS
        ),
        "temperature": (
            TEMPERATURE
        ),
        "n_generations_per_case_per_model": (
            N_GENERATIONS
        ),
        "representation": (
            "FULL_RAW_24_OBSERVATIONS"
        ),
        "evidence_condition": (
            EvidenceCondition
            .COMPLETE_FRESH
            .value
        ),
        "validation_semantics_clarified": (
            True
        ),
        "cases": [],
    }

    for index, fault in enumerate(
        PILOT_FAULTS,
        start=1,
    ):
        scenario = generate_scenario(
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
                scenario,
                EvidenceCondition
                .COMPLETE_FRESH,
            )
        )

        if len(scenario.telemetry) != 24:
            raise RuntimeError(
                "Expected 24 telemetry observations "
                f"but received "
                f"{len(scenario.telemetry)}."
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

        if envelope["failed_gate_id"] is not None:
            raise RuntimeError(
                f"COMPLETE_FRESH case "
                f"{scenario.scenario_id} "
                "failed the Validation Envelope "
                f"at {envelope['failed_gate_id']}."
            )

        expected_gates = {
            "A",
            "B",
            "C",
            "D",
            "E",
        }

        if set(envelope["gates"]) != expected_gates:
            raise RuntimeError(
                "COMPLETE_FRESH case did not "
                "evaluate the complete A→E "
                "Validation Envelope."
            )

        visible_validation = (
            model_visible_validation(
                envelope_result
            )
        )

        context = build_model_context(
            scenario,
            validation=visible_validation,
        )

        leakage = _leakage_checks(
            context=context,
            scenario=scenario,
        )

        if not all(leakage.values()):
            raise RuntimeError(
                "Leakage-control failure for "
                f"{scenario.scenario_id}."
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
                    envelope["trust_mode"]
                ),
                "trust_score": (
                    envelope["trust_score"]
                ),
                "actionable": (
                    envelope["actionable"]
                ),
                "failed_gate_id": (
                    envelope["failed_gate_id"]
                ),
                "passed_gate_ids": (
                    envelope["passed_gate_ids"]
                ),
            },
            "context": {
                "telemetry_observations": (
                    len(scenario.telemetry)
                ),
                "characters": (
                    len(context)
                ),
                "lines": (
                    len(
                        context.splitlines()
                    )
                ),
            },
            "leakage_checks": (
                leakage
            ),
            "models": {},
        }

        for model in MODELS:
            print(
                f"  {model}: running "
                f"{N_GENERATIONS} generations "
                f"at temperature={TEMPERATURE}..."
            )

            def generate_with_clarified_validation(
                prompt: str,
                frozen_context: str,
                _system_prompt: str,
                *,
                adapter=adapters[model],
            ) -> str:
                """Use the normal task with clarified validation semantics."""

                return adapter.generate(
                    prompt,
                    frozen_context,
                    CAPABILITY_SYSTEM_PROMPT,
                )

            results = (
                run_repeated_generations(
                    generate_with_clarified_validation,
                    context=context,
                    n_generations=(
                        N_GENERATIONS
                    ),
                )
            )

            case_report[
                "models"
            ][model] = (
                _summarise_generations(
                    results,
                    ground_truth=scenario.fault,
                )
            )

        report["cases"].append(
            case_report
        )

    report["summary"] = {
        model: _aggregate_model(
            report["cases"],
            model,
        )
        for model in MODELS
    }

    baseline = (
        report[
            "summary"
        ][
            "llama3.2"
        ]
    )

    stronger = (
        report[
            "summary"
        ][
            "gemma3:12b"
        ]
    )

    report["comparison"] = {
        "generation_accuracy_delta": (
            stronger[
                "descriptive_generation_accuracy"
            ]
            -
            baseline[
                "descriptive_generation_accuracy"
            ]
        ),
        "modal_correct_cases_delta": (
            stronger[
                "modal_correct_cases"
            ]
            -
            baseline[
                "modal_correct_cases"
            ]
        ),
        "mean_entropy_delta": (
            stronger[
                "mean_diagnostic_entropy"
            ]
            -
            baseline[
                "mean_diagnostic_entropy"
            ]
        ),
        "mean_modal_fraction_delta": (
            stronger[
                "mean_modal_fraction"
            ]
            -
            baseline[
                "mean_modal_fraction"
            ]
        ),
        "mean_self_confidence_delta": (
            stronger[
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
        "A8.4 MODEL-CAPABILITY CHECKPOINT"
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

        print(
            f"\n{truth}"
        )

        for model in MODELS:
            result = (
                case[
                    "models"
                ][model]
            )

            print(
                f"  {model}: "
                f"modal="
                f"{result['modal_diagnosis']}, "
                f"accuracy="
                f"{result['generation_accuracy']:.3f}, "
                f"entropy="
                f"{result['diagnostic_entropy']:.3f}, "
                f"confidence="
                f"{result['mean_self_confidence']:.3f}"
            )

    print(
        "\nAGGREGATE COMPARISON"
    )

    print(
        "-" * 80
    )

    for model in MODELS:
        summary = (
            report[
                "summary"
            ][model]
        )

        print(
            f"{model}: "
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
        "\ngemma3:12b minus llama3.2"
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
        "\nThis checkpoint is descriptive only. "
        "It tests whether model capability is "
        "a plausible explanation for the "
        "observed diagnostic difficulty."
    )

    print(
        "\nA8.4 MODEL-CAPABILITY CHECKPOINT: COMPLETE"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(
            main()
        )
    )