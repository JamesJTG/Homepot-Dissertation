"""Integration between controlled research scenarios and HOMEPOT Gate B.

This module deliberately depends on the real HOMEPOT ORM models and Gate B.
It does not create its own substitute validator.  A scenario is materialised
for a dedicated experimental device, then the production DataIntegrityGate is
run against exactly those rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timezone
from typing import Any

from sqlalchemy import delete

from ai.gates.base import GateContext, GateResult
from ai.gates.config import gate_b_kwargs
from ai.gates.gate_b import DataIntegrityGate
from homepot.app.models.AnalyticsModel import DeviceMetrics
from homepot.models import HealthCheck

from .scenarios import Scenario


@dataclass(frozen=True)
class MaterialisationResult:
    device_int_id: int
    metric_rows: int
    health_rows: int


def _db_timestamp(value):
    """Use naive UTC to match HOMEPOT's existing DeviceMetrics test fixtures."""
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


async def clear_experimental_rows(session: Any, *, device_int_id: int) -> None:
    """Remove prior telemetry/health rows for one dedicated experiment device."""
    await session.execute(delete(DeviceMetrics).where(DeviceMetrics.device_id == device_int_id))
    await session.execute(delete(HealthCheck).where(HealthCheck.device_id == device_int_id))
    await session.flush()


async def materialise_scenario(
    session: Any,
    scenario: Scenario,
    *,
    device_int_id: int,
    clear_existing: bool = True,
) -> MaterialisationResult:
    """Write a controlled scenario into HOMEPOT's real telemetry tables.

    Health checks mirror the scenario timestamps.  They are intentionally
    healthy because the primary M6 slice manipulates telemetry evidence quality,
    not health-check failure state.  Gate B can therefore test continuity while
    the operational fault remains encoded in DeviceMetrics.
    """
    if clear_existing:
        await clear_experimental_rows(session, device_int_id=device_int_id)

    for idx, point in enumerate(scenario.telemetry, start=1):
        ts = _db_timestamp(point.timestamp)
        session.add(
            DeviceMetrics(
                timestamp=ts,
                device_id=device_int_id,
                cpu_percent=point.cpu_percent,
                memory_percent=point.memory_percent,
                disk_percent=point.disk_percent,
                network_latency_ms=point.network_latency_ms,
                error_rate=point.error_rate,
                provenance="simulated",
            )
        )
        # Explicit IDs avoid relying on database-specific autoincrement
        # behaviour of the HealthCheck composite primary key.
        session.add(
            HealthCheck(
                id=(device_int_id * 100000) + idx,
                timestamp=ts,
                device_id=device_int_id,
                is_healthy=True,
                response_time_ms=20,
                status_code=200,
                endpoint="/research/diagnostic-reliability",
                response_data={"scenario_id": scenario.scenario_id},
            )
        )

    await session.commit()
    return MaterialisationResult(
        device_int_id=device_int_id,
        metric_rows=len(scenario.telemetry),
        health_rows=len(scenario.telemetry),
    )


async def evaluate_gate_b(
    session: Any,
    *,
    device_int_id: int,
    window_seconds: int = 3600,
) -> GateResult:
    """Evaluate the production HOMEPOT Gate B using configured thresholds."""
    gate = DataIntegrityGate(**gate_b_kwargs())
    return await gate.run(
        GateContext(
            session=session,
            device_int_id=device_int_id,
            window_seconds=window_seconds,
        )
    )


def gate_b_summary(result: GateResult) -> dict:
    """Return compact JSON-friendly values needed by the experiment dataset."""
    checks = {check.check_id: check for check in result.checks}
    return {
        "gate_id": result.gate_id,
        "status": result.status.value,
        "score": result.score,
        "checks": {
            check_id: {
                "passed": check.passed,
                "message": check.message,
                "evidence": [ref.to_dict() for ref in check.evidence],
            }
            for check_id, check in checks.items()
        },
    }
