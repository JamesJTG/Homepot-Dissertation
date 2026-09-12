"""Integration between controlled research scenarios and HOMEPOT validation.

This module deliberately depends on HOMEPOT's real ORM models and production
validation gates. It does not introduce substitute research validators.

Controlled scenarios are materialised for a dedicated experimental device.
The production Gate B can be evaluated independently for focused checks, while
the complete production Validation Envelope can be evaluated when the
experiment requires the upstream evidence-readiness signal E.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timezone
import hashlib
import uuid
from typing import Any

from sqlalchemy import delete, select

from ai.gates.base import GateContext, GateResult
from ai.gates.config import gate_b_kwargs
from ai.gates.envelope import EnvelopeResult, build_default_envelope
from ai.gates.gate_b import DataIntegrityGate
from homepot.app.models.AnalyticsModel import DeviceMetrics
from homepot.models import (
    Device,
    DeviceCredential,
    HealthCheck,
    HealthState,
    LifecycleState,
    Site,
)

from .scenarios import Scenario


RESEARCH_SITE_ID = "research-diagnostic-reliability"
RESEARCH_DEVICE_ID = "EXP_DEVICE_01"

RESEARCH_CAPABILITIES = {
    "diagnostics": True,
}

RESEARCH_PERMISSIONS = {
    "diagnostics": True,
}


@dataclass(frozen=True)
class MaterialisationResult:
    """Rows written for one controlled scenario."""

    device_int_id: int
    metric_rows: int
    health_rows: int


def _db_timestamp(value):
    """Use naive UTC to match HOMEPOT's existing DeviceMetrics fixtures."""

    if value.tzinfo is None:
        return value

    return value.astimezone(timezone.utc).replace(tzinfo=None)


async def provision_experimental_device(session: Any) -> Device:
    """Create or restore the dedicated dissertation research device.

    Infrastructure, permissions, capabilities and lifecycle state are treated
    as controlled constants in the diagnostic-reliability experiment.

    Evidence quality is manipulated separately through the controlled
    scenario evidence conditions.
    """

    site = (
        await session.execute(
            select(Site).where(
                Site.site_id == RESEARCH_SITE_ID
            )
        )
    ).scalar_one_or_none()

    if site is None:
        site = Site(
            site_id=RESEARCH_SITE_ID,
            name="Diagnostic Reliability Research",
            description="Dedicated dissertation experiment site",
            location="Controlled experimental environment",
            is_active=True,
        )
        session.add(site)
        await session.flush()

    device = (
        await session.execute(
            select(Device).where(
                Device.device_id == RESEARCH_DEVICE_ID
            )
        )
    ).scalar_one_or_none()

    if device is None:
        device = Device(
            device_id=RESEARCH_DEVICE_ID,
            name="Diagnostic Reliability Experiment Device",
            device_type="POS",
            site_id=site.id,
            lifecycle_state=LifecycleState.ACTIVE.value,
            health_state=HealthState.HEALTHY.value,
            is_active=True,
            is_monitored=True,
            is_simulated=True,
            capabilities=dict(RESEARCH_CAPABILITIES),
            device_permissions=dict(RESEARCH_PERMISSIONS),
            config={
                "device_source": "simulation",
                "research_only": True,
            },
        )

        session.add(device)
        await session.flush()

    else:
        device.site_id = site.id
        device.lifecycle_state = LifecycleState.ACTIVE.value
        device.health_state = HealthState.HEALTHY.value
        device.is_active = True
        device.is_monitored = True
        device.is_simulated = True
        device.capabilities = dict(RESEARCH_CAPABILITIES)
        device.device_permissions = dict(RESEARCH_PERMISSIONS)

        existing_config = dict(device.config or {})
        existing_config.update(
            {
                "device_source": "simulation",
                "research_only": True,
            }
        )
        device.config = existing_config

        await session.flush()

    await ensure_experimental_credential(
        session,
        device_int_id=device.id,
    )

    await session.commit()

    return device


async def ensure_experimental_credential(
    session: Any,
    *,
    device_int_id: int,
) -> DeviceCredential:
    """Ensure the research device has an active non-revoked credential.

    No plaintext API secret is required. Gate E verifies credential lifecycle
    state, so the experiment stores only a deterministic research-only hash.
    """

    credential = (
        await session.execute(
            select(DeviceCredential)
            .where(
                DeviceCredential.device_id == device_int_id,
                DeviceCredential.is_active.is_(True),
                DeviceCredential.revoked_at.is_(None),
            )
            .limit(1)
        )
    ).scalar_one_or_none()

    if credential is not None:
        return credential

    research_key_hash = hashlib.sha256(
        b"diagnostic-reliability-research-only-credential"
    ).hexdigest()

    credential = DeviceCredential(
        credential_id=str(uuid.uuid4()),
        device_id=device_int_id,
        key_hash=research_key_hash,
        is_active=True,
        revoked_at=None,
    )

    session.add(credential)
    await session.flush()

    return credential


async def clear_experimental_rows(
    session: Any,
    *,
    device_int_id: int,
) -> None:
    """Remove prior telemetry and health rows for the research device."""

    await session.execute(
        delete(DeviceMetrics).where(
            DeviceMetrics.device_id == device_int_id
        )
    )

    await session.execute(
        delete(HealthCheck).where(
            HealthCheck.device_id == device_int_id
        )
    )

    await session.flush()


async def materialise_scenario(
    session: Any,
    scenario: Scenario,
    *,
    device_int_id: int,
    clear_existing: bool = True,
) -> MaterialisationResult:
    """Write a controlled scenario into HOMEPOT's real telemetry tables.

    Health checks mirror the scenario timestamps. They are intentionally
    healthy because the diagnostic-reliability experiment manipulates
    telemetry evidence quality rather than device lifecycle or health-check
    failure state.

    Private scenario identity is retained only in research-side database
    metadata and is never included automatically in model-visible context.
    """

    if clear_existing:
        await clear_experimental_rows(
            session,
            device_int_id=device_int_id,
        )

    for idx, point in enumerate(
        scenario.telemetry,
        start=1,
    ):
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
                response_data={
                    "scenario_id": scenario.scenario_id,
                },
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
    """Evaluate production HOMEPOT Gate B using configured thresholds."""

    gate = DataIntegrityGate(
        **gate_b_kwargs()
    )

    return await gate.run(
        GateContext(
            session=session,
            device_int_id=device_int_id,
            window_seconds=window_seconds,
        )
    )


async def evaluate_validation_envelope(
    session: Any,
    *,
    device_id: str,
    device_int_id: int,
    assembled_context: str,
    window_seconds: int = 3600,
) -> EnvelopeResult:
    """Evaluate HOMEPOT's complete production A→E Validation Envelope.

    Gate B uses the same configured thresholds as the focused Gate B
    integration. Gate C receives the actual leakage-controlled model context.

    Gates A, C, D and E should remain controlled constants during the
    evidence-quality experiment. Q is manipulated through scenario evidence
    and is expected to affect Gate B.
    """

    envelope = build_default_envelope(
        **gate_b_kwargs()
    )

    return await envelope.run(
        GateContext(
            session=session,
            device_id=device_id,
            device_int_id=device_int_id,
            assembled_context=assembled_context,
            known_alert_ids=[],
            window_seconds=window_seconds,
        )
    )


def gate_b_summary(
    result: GateResult,
) -> dict:
    """Return compact JSON-friendly Gate B values for research records."""

    checks = {
        check.check_id: check
        for check in result.checks
    }

    return {
        "gate_id": result.gate_id,
        "status": result.status.value,
        "score": result.score,
        "checks": {
            check_id: {
                "passed": check.passed,
                "message": check.message,
                "evidence": [
                    ref.to_dict()
                    for ref in check.evidence
                ],
            }
            for check_id, check in checks.items()
        },
    }


def envelope_summary(
    result: EnvelopeResult,
) -> dict:
    """Return the upstream evidence-readiness values used by the study.

    The trust score is HOMEPOT's rule-derived Validation Envelope score.
    It must not be interpreted as an empirical probability that the final
    LLM diagnosis is correct.
    """

    gates = {}

    for gate_result in result.gate_results:
        gates[gate_result.gate_id] = {
            "status": gate_result.status.value,
            "score": gate_result.score,
            "checks": {
                check.check_id: {
                    "passed": check.passed,
                    "message": check.message,
                }
                for check in gate_result.checks
            },
        }

    return {
        "trust_mode": result.trust_mode.id,
        "trust_mode_label": result.trust_mode.label,
        "trust_score": result.trust_score,
        "actionable": result.is_actionable,
        "failed_gate_id": result.failed_gate_id,
        "passed_gate_ids": result.passed_gate_ids,
        "gates": gates,
    }


def model_visible_validation(
    result: EnvelopeResult,
) -> dict:
    """Return only approved envelope fields for LLM-visible context."""

    return {
        "trust_mode": result.trust_mode.id,
        "trust_score": round(
            result.trust_score,
            3,
        ),
        "actionable": result.is_actionable,
        "failed_gate_id": result.failed_gate_id,
    }