"""Repeated-generation checkpoint for downstream diagnostic uncertainty.

This checkpoint runs one frozen controlled diagnostic case repeatedly through
the dissertation LLM path while keeping the following fixed:

- controlled operational scenario;
- evidence condition;
- model-visible context;
- diagnostic prompt;
- system prompt;
- model;
- generation settings.

Only the model's stochastic generation may vary.

The purpose is to establish whether repeated generations provide measurable
diagnostic-label disagreement before the larger pilot study is designed.

This is not the final uncertainty experiment and does not freeze the final
generation temperature or repetition count.
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
    gate_b_summary,
    materialise_scenario,
)


SITE_ID = "research-diagnostic-reliability"
DEVICE_ID = "EXP_DEVICE_01"

# Small checkpoint only.
#
# This is not yet the final repeated-generation count used in the dissertation.
N_GENERATIONS = 7

# Pilot candidate only.
#
# The pilot will later determine whether this temperature produces useful
# stochastic variation without artificially destabilising diagnosis.
PILOT_TEMPERATURE = 0.7


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


async def main() -> int:
    """Run repeated live generations for one frozen controlled scenario."""

    db = await get_database_service()

    base_scenario = generate_scenario(
        FaultLabel.CPU_SATURATION,
        instance=1,
        seed=20260912,
        end_time=datetime.now(timezone.utc),
        device_id=DEVICE_ID,
    )

    scenario = apply_evidence_condition(
        base_scenario,
        EvidenceCondition.COMPLETE_FRESH,
    )

    async with db.get_session() as session:
        device = await _ensure_experiment_device(session)
        device_int_id = device.id

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
        report = {
            "checkpoint": "A8.3f_repeated_generation",
            "scenario_id": scenario.scenario_id,
            "fault": scenario.fault.value,
            "evidence_condition": EvidenceCondition.COMPLETE_FRESH.value,
            "gate_b": gate_b_summary(gate_b_result),
            "result": "ABORTED",
            "reason": "COMPLETE_FRESH evidence did not pass Gate B",
        }

        print(json.dumps(report, indent=2, default=str))
        print("\nREPEATED-GENERATION CHECKPOINT: FAIL")
        return 1

    model_context = build_model_context(scenario)

    leakage_checks = {
        "scenario_id_absent": scenario.scenario_id not in model_context,
        "scenario_family_absent": (
            scenario.scenario_family not in model_context
        ),
        "private_fault_metadata_absent": (
            f"fault: {scenario.fault.value}" not in model_context
            and f"fault={scenario.fault.value}" not in model_context
        ),
    }

    if not all(leakage_checks.values()):
        report = {
            "checkpoint": "A8.3f_repeated_generation",
            "result": "ABORTED",
            "reason": "Leakage-control check failed",
            "leakage_checks": leakage_checks,
        }

        print(json.dumps(report, indent=2))
        print("\nREPEATED-GENERATION CHECKPOINT: FAIL")
        return 1

    adapter = ExperimentalOllamaAdapter(
        temperature=PILOT_TEMPERATURE,
    )

    if not adapter.check_health():
        report = {
            "checkpoint": "A8.3f_repeated_generation",
            "result": "ABORTED",
            "reason": "Ollama service is not reachable",
            "model": adapter.model,
            "base_url": adapter.base_url,
        }

        print(json.dumps(report, indent=2))
        print("\nREPEATED-GENERATION CHECKPOINT: FAIL")
        return 1

    results = run_repeated_generations(
        adapter.generate,
        context=model_context,
        n_generations=N_GENERATIONS,
    )

    labels = [result.parsed.diagnosis for result in results]
    confidences = [
        result.parsed.confidence
        for result in results
    ]

    distribution = diagnosis_distribution(labels)
    entropy = diagnostic_entropy(labels)
    agreement = modal_fraction(labels)

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

    correct_generations = sum(
        result.parsed.diagnosis.value == scenario.fault.value
        for result in results
    )

    generation_records = [
        {
            "generation_id": result.generation_id,
            "diagnosis": result.parsed.diagnosis.value,
            "confidence": result.parsed.confidence,
            "explanation": result.parsed.explanation,
            "correct": (
                result.parsed.diagnosis.value
                == scenario.fault.value
            ),
        }
        for result in results
    ]

    report = {
        "checkpoint": "A8.3f_repeated_generation",
        "research_only": {
            "scenario_id": scenario.scenario_id,
            "scenario_family": scenario.scenario_family,
            "private_ground_truth": scenario.fault.value,
            "seed": scenario.seed,
        },
        "evidence_condition": EvidenceCondition.COMPLETE_FRESH.value,
        "gate_b": {
            "status": gate_b_result.status.value,
            "score": gate_b_result.score,
        },
        "generation_configuration": {
            "model": adapter.model,
            "temperature": adapter.temperature,
            "n_generations": N_GENERATIONS,
        },
        "generations": generation_records,
        "uncertainty_summary": {
            "diagnosis_distribution": distribution,
            "diagnostic_entropy": entropy,
            "modal_fraction": agreement,
            "modal_diagnosis": modal_diagnosis,
            "mean_self_confidence": (
                sum(confidences) / len(confidences)
            ),
        },
        "evaluation": {
            "correct_generations": correct_generations,
            "incorrect_generations": (
                N_GENERATIONS - correct_generations
            ),
            "generation_accuracy": (
                correct_generations / N_GENERATIONS
            ),
            "leakage_checks": leakage_checks,
        },
    }

    print(json.dumps(report, indent=2, default=str))

    print("\nREPEATED-GENERATION CHECKPOINT: PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))