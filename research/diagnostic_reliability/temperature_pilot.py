"""A8.4a pilot: compare candidate LLM generation temperatures.

The purpose of this pilot is to identify a defensible non-zero temperature
for repeated-generation uncertainty measurement.

Temperature is not selected by maximising diagnostic accuracy. The pilot
instead examines whether each candidate produces:

- valid structured diagnoses;
- sufficient stability for repeated operational diagnosis;
- enough stochastic variation for disagreement-based uncertainty to exist.

Only COMPLETE_FRESH evidence is used here so that evidence quality does not
confound the temperature comparison.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from datetime import datetime, timezone
import json

from sqlalchemy import select

from ai.gates.base import GateStatus
from homepot.database import get_database_service
from homepot.models import Device, HealthState, LifecycleState, Site

from .context import build_model_context
from .evidence_conditions import apply_evidence_condition
from .ollama_adapter import ExperimentalOllamaAdapter
from .runner import run_repeated_generations
from .scenarios import generate_scenario
from .taxonomy import EvidenceCondition, FaultLabel
from .uncertainty import (
    diagnosis_distribution,
    diagnostic_entropy,
    modal_fraction,
)
from .validation_integration import (
    evaluate_gate_b,
    materialise_scenario,
)


SITE_ID = "research-diagnostic-reliability"
DEVICE_ID = "EXP_DEVICE_01"

TEMPERATURES = (0.2, 0.4, 0.7)
N_GENERATIONS = 5

PILOT_FAULTS = (
    FaultLabel.NORMAL,
    FaultLabel.CPU_SATURATION,
    FaultLabel.MEMORY_PRESSURE,
    FaultLabel.NETWORK_DEGRADATION,
    FaultLabel.APPLICATION_FAILURE,
)


async def _ensure_experiment_device(session):
    """Create or restore the dedicated dissertation research device."""

    site = (
        await session.execute(
            select(Site).where(Site.site_id == SITE_ID)
        )
    ).scalar_one_or_none()

    if site is None:
        site = Site(
            site_id=SITE_ID,
            name="Diagnostic Reliability Research",
            description="Dedicated dissertation experiment site",
            location="Controlled experimental environment",
            is_active=True,
        )
        session.add(site)
        await session.flush()

    device = (
        await session.execute(
            select(Device).where(Device.device_id == DEVICE_ID)
        )
    ).scalar_one_or_none()

    if device is None:
        device = Device(
            device_id=DEVICE_ID,
            name="Diagnostic Reliability Experiment Device",
            device_type="POS",
            site_id=site.id,
            lifecycle_state=LifecycleState.ACTIVE.value,
            health_state=HealthState.HEALTHY.value,
            is_active=True,
            is_monitored=True,
            is_simulated=True,
            config={
                "device_source": "simulation",
                "research_only": True,
            },
        )
        session.add(device)
        await session.flush()

    else:
        device.is_active = True
        device.lifecycle_state = LifecycleState.ACTIVE.value
        device.health_state = HealthState.HEALTHY.value

    await session.commit()

    return device


def _summarise_case(results, ground_truth: FaultLabel) -> dict:
    """Summarise repeated generations for one scenario-temperature pair."""

    labels = [result.parsed.diagnosis for result in results]
    confidences = [
        result.parsed.confidence
        for result in results
    ]

    counts = Counter(label.value for label in labels)

    modal_count = max(counts.values())
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

    correct = sum(
        result.parsed.diagnosis.value == ground_truth.value
        for result in results
    )

    return {
        "diagnosis_distribution": diagnosis_distribution(labels),
        "diagnostic_entropy": diagnostic_entropy(labels),
        "modal_fraction": modal_fraction(labels),
        "modal_diagnosis": modal_diagnosis,
        "mean_self_confidence": (
            sum(confidences) / len(confidences)
        ),
        "correct_generations": correct,
        "generation_accuracy": correct / len(results),
        "generations": [
            {
                "generation_id": result.generation_id,
                "diagnosis": result.parsed.diagnosis.value,
                "confidence": result.parsed.confidence,
                "correct": (
                    result.parsed.diagnosis.value
                    == ground_truth.value
                ),
            }
            for result in results
        ],
    }


async def main() -> int:
    """Run the candidate-temperature pilot."""

    db = await get_database_service()

    async with db.get_session() as session:
        device = await _ensure_experiment_device(session)
        device_int_id = device.id

    scenarios = {}

    for index, fault in enumerate(PILOT_FAULTS, start=1):
        base = generate_scenario(
            fault,
            instance=1,
            seed=20260912 + index,
            end_time=datetime.now(timezone.utc),
            device_id=DEVICE_ID,
        )

        scenarios[fault] = apply_evidence_condition(
            base,
            EvidenceCondition.COMPLETE_FRESH,
        )

    report = {
        "pilot": "A8.4a_temperature_comparison",
        "evidence_condition": EvidenceCondition.COMPLETE_FRESH.value,
        "candidate_temperatures": list(TEMPERATURES),
        "n_generations_per_case": N_GENERATIONS,
        "cases": [],
    }

    for fault, scenario in scenarios.items():
        print(
            f"\nPreparing {fault.value} "
            f"({scenario.scenario_family})"
        )

        async with db.get_session() as session:
            await materialise_scenario(
                session,
                scenario,
                device_int_id=device_int_id,
            )

            gate_b_result = await evaluate_gate_b(
                session,
                device_int_id=device_int_id,
            )

        if gate_b_result.status != GateStatus.PASS:
            raise RuntimeError(
                f"COMPLETE_FRESH case {scenario.scenario_id} "
                "did not pass Gate B."
            )

        model_context = build_model_context(scenario)

        leakage_checks = {
            "scenario_id_absent": (
                scenario.scenario_id not in model_context
            ),
            "scenario_family_absent": (
                scenario.scenario_family not in model_context
            ),
            "private_fault_metadata_absent": (
                f"fault: {scenario.fault.value}" not in model_context
                and f"fault={scenario.fault.value}" not in model_context
            ),
        }

        if not all(leakage_checks.values()):
            raise RuntimeError(
                f"Leakage-control failure for {scenario.scenario_id}"
            )

        case_report = {
            "research_only": {
                "scenario_id": scenario.scenario_id,
                "scenario_family": scenario.scenario_family,
                "private_ground_truth": scenario.fault.value,
                "seed": scenario.seed,
            },
            "gate_b_status": gate_b_result.status.value,
            "leakage_checks": leakage_checks,
            "temperatures": {},
        }

        for temperature in TEMPERATURES:
            print(
                f"  temperature={temperature}: "
                f"{N_GENERATIONS} generations..."
            )

            adapter = ExperimentalOllamaAdapter(
                temperature=temperature,
            )

            if not adapter.check_health():
                raise RuntimeError(
                    "Ollama service is not reachable."
                )

            results = run_repeated_generations(
                adapter.generate,
                context=model_context,
                n_generations=N_GENERATIONS,
            )

            case_report["temperatures"][str(temperature)] = (
                _summarise_case(
                    results,
                    ground_truth=scenario.fault,
                )
            )

        report["cases"].append(case_report)

    temperature_summary = {}

    for temperature in TEMPERATURES:
        key = str(temperature)

        case_summaries = [
            case["temperatures"][key]
            for case in report["cases"]
        ]

        temperature_summary[key] = {
            "mean_modal_fraction": (
                sum(
                    case["modal_fraction"]
                    for case in case_summaries
                )
                / len(case_summaries)
            ),
            "mean_diagnostic_entropy": (
                sum(
                    case["diagnostic_entropy"]
                    for case in case_summaries
                )
                / len(case_summaries)
            ),
            "mean_self_confidence": (
                sum(
                    case["mean_self_confidence"]
                    for case in case_summaries
                )
                / len(case_summaries)
            ),
            "descriptive_generation_accuracy": (
                sum(
                    case["correct_generations"]
                    for case in case_summaries
                )
                / (
                    len(case_summaries)
                    * N_GENERATIONS
                )
            ),
            "cases_with_any_disagreement": sum(
                case["modal_fraction"] < 1.0
                for case in case_summaries
            ),
            "cases_with_complete_agreement": sum(
                case["modal_fraction"] == 1.0
                for case in case_summaries
            ),
        }

    report["temperature_summary"] = temperature_summary

    print("\n" + "=" * 72)
    print("TEMPERATURE PILOT RESULTS")
    print("=" * 72)

    print(json.dumps(report, indent=2, default=str))

    print("\nSUMMARY")
    print("-" * 72)

    for temperature in TEMPERATURES:
        summary = temperature_summary[str(temperature)]

        print(
            f"Temperature {temperature}: "
            f"mean modal fraction="
            f"{summary['mean_modal_fraction']:.3f}, "
            f"mean entropy="
            f"{summary['mean_diagnostic_entropy']:.3f}, "
            f"cases with disagreement="
            f"{summary['cases_with_any_disagreement']}/"
            f"{len(PILOT_FAULTS)}, "
            f"descriptive accuracy="
            f"{summary['descriptive_generation_accuracy']:.3f}"
        )

    print(
        "\nNo temperature is automatically selected by this script. "
        "Selection must be justified from stability and observable "
        "stochastic variation, not by maximising pilot accuracy."
    )

    print("\nA8.4a TEMPERATURE PILOT: COMPLETE")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))