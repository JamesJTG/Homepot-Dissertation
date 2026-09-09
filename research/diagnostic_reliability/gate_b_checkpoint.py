"""Local M6.2 checkpoint: prove controlled Q changes drive real HOMEPOT Gate B.

Run from repository root after normal backend/database setup:
    PYTHONPATH=.:backend/src python -m research.diagnostic_reliability.gate_b_checkpoint

The script creates/reuses a dedicated research site/device, materialises the
same CPU-saturation scenario twice (fresh then stale), and runs production Gate
B after each materialisation. It exits non-zero if the expected PASS -> FAIL
manipulation is not observed.
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
from .validation_integration import evaluate_gate_b, gate_b_summary, materialise_scenario

SITE_ID = "research-diagnostic-reliability"
DEVICE_ID = "EXP_DEVICE_01"


async def _ensure_experiment_device(session):
    site = (
        await session.execute(select(Site).where(Site.site_id == SITE_ID))
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
        await session.execute(select(Device).where(Device.device_id == DEVICE_ID))
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
            config={"device_source": "simulation", "research_only": True},
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
    db = await get_database_service()
    base = generate_scenario(
        FaultLabel.CPU_SATURATION,
        seed=20260909,
        end_time=datetime.now(timezone.utc),
        device_id=DEVICE_ID,
    )
    stale = apply_evidence_condition(base, EvidenceCondition.STALE)

    async with db.get_session() as session:
        device = await _ensure_experiment_device(session)
        await materialise_scenario(session, base, device_int_id=device.id)
        fresh_result = await evaluate_gate_b(session, device_int_id=device.id)

    async with db.get_session() as session:
        await materialise_scenario(session, stale, device_int_id=device.id)
        stale_result = await evaluate_gate_b(session, device_int_id=device.id)

    report = {
        "ground_truth_preserved": base.fault.value == stale.fault.value,
        "fault": base.fault.value,
        "fresh": gate_b_summary(fresh_result),
        "stale": gate_b_summary(stale_result),
    }
    print(json.dumps(report, indent=2, default=str))

    ok = (
        report["ground_truth_preserved"]
        and fresh_result.status == GateStatus.PASS
        and stale_result.status == GateStatus.FAIL
        and any(
            check.check_id == "B.freshness" and not check.passed
            for check in stale_result.checks
        )
    )
    print("\nM6.2 CHECKPOINT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
