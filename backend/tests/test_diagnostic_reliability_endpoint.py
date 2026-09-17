"""Boundary tests for the diagnostic reliability API feature."""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from ai.post_inference_diagnostic_reliability.models import (
    DiagnosticDecisionState,
    DiagnosticReliabilityResult,
)
from research.diagnostic_reliability.taxonomy import DiagnosisLabel
from homepot.app.api.API_v1.Endpoints import AIEndpoint


class _ScalarResult:
    def __init__(self, *, one=None, many=None):
        self._one = one
        self._many = list(many or [])

    def scalar_one_or_none(self):
        return self._one

    def scalars(self):
        return self

    def all(self):
        return self._many


class _AsyncSession:
    def __init__(self, results):
        self._results = iter(results)

    async def execute(self, statement):
        return next(self._results)


class _SessionContext:
    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _DatabaseService:
    def __init__(self, session):
        self._session = session

    def get_session(self):
        return _SessionContext(self._session)


class _TrustMode:
    id = "grounded"


class _EnvelopeResult:
    def __init__(
        self,
        *,
        actionable=True,
        trust_score=1.0,
        failed_gate_id=None,
    ):
        self.is_actionable = actionable
        self.trust_score = trust_score
        self.failed_gate_id = failed_gate_id
        self.trust_mode = _TrustMode()


class _Envelope:
    def __init__(self, result):
        self.result = result
        self.received_context = None

    async def run(self, context):
        self.received_context = context
        return self.result


def _device(*, active=True):
    return SimpleNamespace(
        id=101,
        device_id="DEVICE_01",
        is_active=active,
    )


def _telemetry(count=24):
    start = datetime(2026, 9, 17, 12, 0)

    rows = []

    for index in range(count):
        rows.append(
            SimpleNamespace(
                timestamp=start + timedelta(minutes=index),
                cpu_percent=50.0 + index,
                memory_percent=60.0,
                disk_percent=40.0,
                network_latency_ms=10.0,
                error_rate=0.01,
                active_connections=5,
                queue_depth=2,
            )
        )

    return rows


def _sync_db():
    query = MagicMock()
    query.filter.return_value = query
    query.first.return_value = SimpleNamespace(
        id=1,
        email="test@example.com",
    )

    db = MagicMock()
    db.query.return_value = query

    return db


def _current_user():
    return {
        "email": "test@example.com",
        "user_id": 1,
    }


def _substantive_result(*, evidence_ready=True):
    return DiagnosticReliabilityResult(
        modal_diagnosis=DiagnosisLabel.CPU_SATURATION,
        evidence_ready=evidence_ready,
        mean_self_confidence=0.90,
        diagnostic_entropy=0.10,
        modal_fraction=6 / 7,
        predicted_reliability=0.82,
        decision_state=DiagnosticDecisionState.ISSUE,
        generation_count=7,
    )


def _abstain_result(*, evidence_ready=True):
    return DiagnosticReliabilityResult(
        modal_diagnosis=DiagnosisLabel.UNKNOWN,
        evidence_ready=evidence_ready,
        mean_self_confidence=0.72,
        diagnostic_entropy=0.20,
        modal_fraction=5 / 7,
        predicted_reliability=None,
        decision_state=DiagnosticDecisionState.ABSTAIN,
        generation_count=7,
    )


@pytest.mark.asyncio
async def test_diagnostic_endpoint_returns_substantive_reliability_result(
    monkeypatch,
):
    device = _device()
    telemetry = _telemetry()

    session = _AsyncSession(
        [
            _ScalarResult(one=device),
            _ScalarResult(many=list(reversed(telemetry))),
        ]
    )

    monkeypatch.setattr(
        AIEndpoint,
        "get_database_service",
        lambda: _async_value(_DatabaseService(session)),
    )

    monkeypatch.setattr(
        AIEndpoint,
        "verify_device_belongs_to_user",
        lambda *args, **kwargs: None,
    )

    trust = _EnvelopeResult(
        actionable=True,
        trust_score=1.0,
    )
    envelope = _Envelope(trust)

    monkeypatch.setattr(
        AIEndpoint,
        "build_envelope_from_config",
        lambda: envelope,
    )

    monkeypatch.setattr(
        AIEndpoint,
        "get_ai_services",
        lambda: (MagicMock(), MagicMock(), MagicMock()),
    )

    captured = {}

    class FakeReliabilityService:
        def __init__(self, llm):
            captured["llm"] = llm

        def assess(self, *, context, envelope):
            captured["context"] = context
            captured["envelope"] = envelope
            return _substantive_result(
                evidence_ready=envelope.is_actionable
            )

    monkeypatch.setattr(
        AIEndpoint,
        "PostInferenceDiagnosticReliabilityService",
        FakeReliabilityService,
    )

    response = (
        await AIEndpoint.assess_device_diagnostic_reliability(
            device_id="DEVICE_01",
            current_user=_current_user(),
            db=_sync_db(),
        )
    )

    assert response["device_id"] == "DEVICE_01"
    assert response["diagnosis"] == "CPU_SATURATION"
    assert response["decision_state"] == "ISSUE"
    assert response["predicted_reliability"] == 0.82
    assert response["uncertainty"]["generation_count"] == 7
    assert response["validation"]["evidence_ready"] is True

    assert captured["envelope"] is trust
    assert "[VALIDATION STATUS]" in captured["context"]
    assert "actionable: True" in captured["context"]


@pytest.mark.asyncio
async def test_diagnostic_endpoint_preserves_unknown_as_abstain(
    monkeypatch,
):
    device = _device()

    session = _AsyncSession(
        [
            _ScalarResult(one=device),
            _ScalarResult(many=_telemetry()),
        ]
    )

    monkeypatch.setattr(
        AIEndpoint,
        "get_database_service",
        lambda: _async_value(_DatabaseService(session)),
    )

    monkeypatch.setattr(
        AIEndpoint,
        "verify_device_belongs_to_user",
        lambda *args, **kwargs: None,
    )

    trust = _EnvelopeResult(
        actionable=True,
        trust_score=1.0,
    )

    monkeypatch.setattr(
        AIEndpoint,
        "build_envelope_from_config",
        lambda: _Envelope(trust),
    )

    monkeypatch.setattr(
        AIEndpoint,
        "get_ai_services",
        lambda: (MagicMock(), MagicMock(), MagicMock()),
    )

    class FakeReliabilityService:
        def __init__(self, llm):
            pass

        def assess(self, *, context, envelope):
            return _abstain_result(
                evidence_ready=envelope.is_actionable
            )

    monkeypatch.setattr(
        AIEndpoint,
        "PostInferenceDiagnosticReliabilityService",
        FakeReliabilityService,
    )

    response = (
        await AIEndpoint.assess_device_diagnostic_reliability(
            device_id="DEVICE_01",
            current_user=_current_user(),
            db=_sync_db(),
        )
    )

    assert response["diagnosis"] == "UNKNOWN"
    assert response["decision_state"] == "ABSTAIN"
    assert response["predicted_reliability"] is None


@pytest.mark.asyncio
async def test_diagnostic_endpoint_propagates_non_actionable_evidence(
    monkeypatch,
):
    device = _device()

    session = _AsyncSession(
        [
            _ScalarResult(one=device),
            _ScalarResult(many=_telemetry()),
        ]
    )

    monkeypatch.setattr(
        AIEndpoint,
        "get_database_service",
        lambda: _async_value(_DatabaseService(session)),
    )

    monkeypatch.setattr(
        AIEndpoint,
        "verify_device_belongs_to_user",
        lambda *args, **kwargs: None,
    )

    trust = _EnvelopeResult(
        actionable=False,
        trust_score=0.36,
        failed_gate_id="B",
    )

    monkeypatch.setattr(
        AIEndpoint,
        "build_envelope_from_config",
        lambda: _Envelope(trust),
    )

    monkeypatch.setattr(
        AIEndpoint,
        "get_ai_services",
        lambda: (MagicMock(), MagicMock(), MagicMock()),
    )

    captured = {}

    class FakeReliabilityService:
        def __init__(self, llm):
            pass

        def assess(self, *, context, envelope):
            captured["envelope"] = envelope

            return DiagnosticReliabilityResult(
                modal_diagnosis=DiagnosisLabel.CPU_SATURATION,
                evidence_ready=False,
                mean_self_confidence=0.95,
                diagnostic_entropy=0.0,
                modal_fraction=1.0,
                predicted_reliability=0.02,
                decision_state=DiagnosticDecisionState.WITHHOLD,
                generation_count=7,
            )

    monkeypatch.setattr(
        AIEndpoint,
        "PostInferenceDiagnosticReliabilityService",
        FakeReliabilityService,
    )

    response = (
        await AIEndpoint.assess_device_diagnostic_reliability(
            device_id="DEVICE_01",
            current_user=_current_user(),
            db=_sync_db(),
        )
    )

    assert captured["envelope"] is trust
    assert response["validation"]["evidence_ready"] is False
    assert response["validation"]["failed_gate_id"] == "B"
    assert response["decision_state"] == "WITHHOLD"


@pytest.mark.asyncio
async def test_diagnostic_endpoint_returns_404_for_missing_device(
    monkeypatch,
):
    session = _AsyncSession(
        [
            _ScalarResult(one=None),
        ]
    )

    monkeypatch.setattr(
        AIEndpoint,
        "get_database_service",
        lambda: _async_value(_DatabaseService(session)),
    )

    with pytest.raises(HTTPException) as exc_info:
        await AIEndpoint.assess_device_diagnostic_reliability(
            device_id="MISSING_DEVICE",
            current_user=_current_user(),
            db=_sync_db(),
        )

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_diagnostic_endpoint_returns_422_without_telemetry(
    monkeypatch,
):
    device = _device()

    session = _AsyncSession(
        [
            _ScalarResult(one=device),
            _ScalarResult(many=[]),
        ]
    )

    monkeypatch.setattr(
        AIEndpoint,
        "get_database_service",
        lambda: _async_value(_DatabaseService(session)),
    )

    monkeypatch.setattr(
        AIEndpoint,
        "verify_device_belongs_to_user",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(HTTPException) as exc_info:
        await AIEndpoint.assess_device_diagnostic_reliability(
            device_id="DEVICE_01",
            current_user=_current_user(),
            db=_sync_db(),
        )

    assert exc_info.value.status_code == 422


@pytest.mark.asyncio
async def test_diagnostic_endpoint_enforces_device_authorisation(
    monkeypatch,
):
    device = _device()

    session = _AsyncSession(
        [
            _ScalarResult(one=device),
        ]
    )

    monkeypatch.setattr(
        AIEndpoint,
        "get_database_service",
        lambda: _async_value(_DatabaseService(session)),
    )

    def reject_access(*args, **kwargs):
        raise HTTPException(
            status_code=403,
            detail="Forbidden",
        )

    monkeypatch.setattr(
        AIEndpoint,
        "verify_device_belongs_to_user",
        reject_access,
    )

    with pytest.raises(HTTPException) as exc_info:
        await AIEndpoint.assess_device_diagnostic_reliability(
            device_id="DEVICE_01",
            current_user=_current_user(),
            db=_sync_db(),
        )

    assert exc_info.value.status_code == 403


async def _async_value(value):
    return value