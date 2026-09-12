"""First live controlled LLM diagnostic checkpoint.

This checkpoint exercises one real dissertation diagnostic path:

    controlled CPU_SATURATION scenario
        -> COMPLETE_FRESH evidence
        -> HOMEPOT production Gate B
        -> leakage-controlled model context
        -> fixed diagnostic prompt
        -> controlled Ollama adapter
        -> strict structured-output parser
        -> diagnosis correctness check

The private experimental ground truth is used only after generation for
evaluation. It is never included in the model-visible context.

This checkpoint intentionally does not map Gate B's internal score onto the
Validation Envelope's trust_score. Those are distinct concepts. Full
model-visible upstream evidence-readiness fields are integrated separately
when the Validation Envelope is attached to the research path.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json

from sqlalchemy import select

from ai.gates.base import GateStatus
from homepot.database import get_database_service
from homepot.models import Device, HealthState, LifecycleState, Site

from .context import build_model_context
from .evidence_conditions import apply_evidence_condition
from .ollama_adapter import ExperimentalOllamaAdapter
from .parser import parse_diagnosis
from .prompt import DIAGNOSTIC_PROMPT, SYSTEM_PROMPT
from .scenarios import generate_scenario
from .taxonomy import EvidenceCondition, FaultLabel
from .validation_integration import (
    evaluate_gate_b,
    gate_b_summary,
    materialise_scenario,
)


SITE_ID = "research-diagnostic-reliability"
DEVICE_ID = "EXP_DEVICE_01"

# Pilot-only starting value.
#
# This is not yet the scientifically frozen experiment temperature. The pilot
# will later determine whether a non-zero setting produces enough repeated-
# generation variation without making diagnoses artificially unstable.
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
    """Run one controlled live diagnosis through the dissertation LLM path."""

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
            "checkpoint": "A8.3e_first_live_diagnosis",
            "scenario_id": scenario.scenario_id,
            "fault": scenario.fault.value,
            "evidence_condition": EvidenceCondition.COMPLETE_FRESH.value,
            "gate_b": gate_b_summary(gate_b_result),
            "result": "ABORTED",
            "reason": "COMPLETE_FRESH evidence did not pass Gate B",
        }

        print(json.dumps(report, indent=2, default=str))
        print("\nLIVE LLM DIAGNOSTIC CHECKPOINT: FAIL")
        return 1

    # IMPORTANT:
    # We deliberately do not manufacture Validation Envelope fields here.
    # Gate B PASS is verified as an upstream precondition, but Gate B's score
    # is not relabelled as trust_score.
    model_context = build_model_context(scenario)

    adapter = ExperimentalOllamaAdapter(
        temperature=PILOT_TEMPERATURE,
    )

    if not adapter.check_health():
        print(
            json.dumps(
                {
                    "checkpoint": "A8.3e_first_live_diagnosis",
                    "result": "ABORTED",
                    "reason": "Ollama service is not reachable",
                    "model": adapter.model,
                    "base_url": adapter.base_url,
                },
                indent=2,
            )
        )
        print("\nLIVE LLM DIAGNOSTIC CHECKPOINT: FAIL")
        return 1

    raw_response = adapter.generate(
        DIAGNOSTIC_PROMPT,
        model_context,
        SYSTEM_PROMPT,
    )

    parsed = parse_diagnosis(raw_response)

    correct = parsed.diagnosis.value == scenario.fault.value

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

    report = {
        "checkpoint": "A8.3e_first_live_diagnosis",
        "research_only": {
            "scenario_id": scenario.scenario_id,
            "scenario_family": scenario.scenario_family,
            "private_ground_truth": scenario.fault.value,
            "seed": scenario.seed,
        },
        "evidence_condition": EvidenceCondition.COMPLETE_FRESH.value,
        "gate_b": gate_b_summary(gate_b_result),
        "llm": {
            "model": adapter.model,
            "temperature": adapter.temperature,
            "diagnosis": parsed.diagnosis.value,
            "confidence": parsed.confidence,
            "explanation": parsed.explanation,
        },
        "evaluation": {
            "correct": correct,
            "leakage_checks": leakage_checks,
        },
    }

    print(json.dumps(report, indent=2, default=str))

    leakage_ok = all(leakage_checks.values())

    # A wrong diagnosis does NOT make the technical checkpoint fail.
    #
    # Diagnostic correctness is an experimental outcome. The purpose of this
    # checkpoint is to establish that the controlled path executes correctly,
    # preserves the leakage boundary, and returns a valid structured diagnosis.
    checkpoint_ok = leakage_ok

    print(
        "\nLIVE LLM DIAGNOSTIC CHECKPOINT:",
        "PASS" if checkpoint_ok else "FAIL",
    )

    return 0 if checkpoint_ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))