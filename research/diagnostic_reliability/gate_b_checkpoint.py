"""Gate B experimental-validity checkpoint for controlled evidence conditions.

This checkpoint proves that the dissertation's controlled evidence-quality
conditions drive the intended behaviour in HOMEPOT's real production
DataIntegrityGate.

Run from the repository root after normal backend/database setup:

    PYTHONPATH=.:backend/src \
    python -m research.diagnostic_reliability.gate_b_checkpoint

Expected contract:

    COMPLETE_FRESH -> Gate B PASS
    STALE          -> Gate B FAIL at B.freshness
    INCOMPLETE     -> Gate B FAIL at B.completeness
    DISCONTINUOUS  -> Gate B FAIL at B.continuity

The underlying operational fault must remain unchanged across all four
evidence-quality conditions.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json

from sqlalchemy import select

from ai.gates.base import GateStatus
from homepot.database import get_database_service
from homepot.models import Device, HealthState, LifecycleState, Site

from .evidence_conditions import apply_evidence_condition
from .scenarios import generate_scenario
from .taxonomy import EvidenceCondition, FaultLabel
from .validation_integration import (
    evaluate_gate_b,
    gate_b_summary,
    materialise_scenario,
)


SITE_ID = "research-diagnostic-reliability"
DEVICE_ID = "EXP_DEVICE_01"


async def _ensure_experiment_device(session):
    """Create or restore the dedicated research device."""

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


def _check_passed(result, check_id: str) -> bool:
    """Return whether one named Gate B check passed."""

    for check in result.checks:
        if check.check_id == check_id:
            return check.passed

    raise RuntimeError(
        f"Expected Gate B check {check_id!r} was not returned."
    )


async def main() -> int:
    """Run the complete evidence-condition to Gate B validity checkpoint."""

    db = await get_database_service()

    base = generate_scenario(
        FaultLabel.CPU_SATURATION,
        seed=20260909,
        end_time=datetime.now(timezone.utc),
        device_id=DEVICE_ID,
    )

    scenarios = {
        EvidenceCondition.COMPLETE_FRESH: base,
        EvidenceCondition.STALE: apply_evidence_condition(
            base,
            EvidenceCondition.STALE,
        ),
        EvidenceCondition.INCOMPLETE: apply_evidence_condition(
            base,
            EvidenceCondition.INCOMPLETE,
        ),
        EvidenceCondition.DISCONTINUOUS: apply_evidence_condition(
            base,
            EvidenceCondition.DISCONTINUOUS,
        ),
    }

    results = {}

    async with db.get_session() as session:
        device = await _ensure_experiment_device(session)
        device_int_id = device.id

    for condition, scenario in scenarios.items():
        async with db.get_session() as session:
            await materialise_scenario(
                session,
                scenario,
                device_int_id=device_int_id,
            )

            results[condition] = await evaluate_gate_b(
                session,
                device_int_id=device_int_id,
            )

    ground_truth_preserved = all(
        scenario.fault == base.fault
        for scenario in scenarios.values()
    )

    checks = {
        "ground_truth_preserved": ground_truth_preserved,
        "complete_fresh_passes": (
            results[EvidenceCondition.COMPLETE_FRESH].status
            == GateStatus.PASS
        ),
        "stale_fails_gate": (
            results[EvidenceCondition.STALE].status
            == GateStatus.FAIL
        ),
        "stale_fails_freshness": not _check_passed(
            results[EvidenceCondition.STALE],
            "B.freshness",
        ),
        "incomplete_fails_gate": (
            results[EvidenceCondition.INCOMPLETE].status
            == GateStatus.FAIL
        ),
        "incomplete_fails_completeness": not _check_passed(
            results[EvidenceCondition.INCOMPLETE],
            "B.completeness",
        ),
        "discontinuous_fails_gate": (
            results[EvidenceCondition.DISCONTINUOUS].status
            == GateStatus.FAIL
        ),
        "discontinuous_fails_continuity": not _check_passed(
            results[EvidenceCondition.DISCONTINUOUS],
            "B.continuity",
        ),
    }

    report = {
        "fault": base.fault.value,
        "conditions": {
            condition.value: gate_b_summary(result)
            for condition, result in results.items()
        },
        "validity_checks": checks,
    }

    print(json.dumps(report, indent=2, default=str))

    ok = all(checks.values())

    print(
        "\nGATE B EXPERIMENTAL-VALIDITY CHECKPOINT:",
        "PASS" if ok else "FAIL",
    )

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))