import math

import pytest

from ai.post_inference_diagnostic_reliability import (
    ISSUE_THRESHOLD,
    ReliabilityPolicyState,
    apply_reliability_policy,
    estimate_diagnostic_reliability,
)

from research.diagnostic_reliability.parser import ParsedDiagnosis
from research.diagnostic_reliability.taxonomy import DiagnosisLabel

from ai.post_inference_diagnostic_reliability import (
    DiagnosticDecisionState,
    assess_diagnostic_reliability,
)

from unittest.mock import MagicMock

from ai.post_inference_diagnostic_reliability import (
    DIAGNOSTIC_TEMPERATURE,
    GENERATION_COUNT,
    DiagnosticGenerationError,
    DiagnosticGenerator,
)

from ai.post_inference_diagnostic_reliability import (
    PostInferenceDiagnosticReliabilityService,
)

from types import SimpleNamespace

from ai.post_inference_diagnostic_reliability import (
    TELEMETRY_POINT_LIMIT,
    build_diagnostic_context,
    model_visible_validation,
)

def test_estimator_matches_frozen_equation():
    evidence_ready = True
    mean_self_confidence = 0.90
    diagnostic_entropy = 0.10

    expected_linear_predictor = (
        -19.046914056
        + 6.446372648
        + 14.868349383 * mean_self_confidence
        + 3.118253729 * diagnostic_entropy
    )

    expected = 1.0 / (
        1.0 + math.exp(-expected_linear_predictor)
    )

    actual = estimate_diagnostic_reliability(
        evidence_ready=evidence_ready,
        mean_self_confidence=mean_self_confidence,
        diagnostic_entropy=diagnostic_entropy,
    )

    assert actual == pytest.approx(expected, abs=1e-12)


def test_non_actionable_evidence_is_always_withheld():
    state = apply_reliability_policy(
        evidence_ready=False,
        predicted_reliability=0.99,
    )

    assert state is ReliabilityPolicyState.WITHHOLD


def test_actionable_probability_below_threshold_is_qualified():
    state = apply_reliability_policy(
        evidence_ready=True,
        predicted_reliability=ISSUE_THRESHOLD - 0.01,
    )

    assert state is ReliabilityPolicyState.QUALIFY


def test_actionable_probability_at_threshold_is_issued():
    state = apply_reliability_policy(
        evidence_ready=True,
        predicted_reliability=ISSUE_THRESHOLD,
    )

    assert state is ReliabilityPolicyState.ISSUE


def test_actionable_probability_above_threshold_is_issued():
    state = apply_reliability_policy(
        evidence_ready=True,
        predicted_reliability=0.90,
    )

    assert state is ReliabilityPolicyState.ISSUE


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_estimator_rejects_invalid_confidence(confidence):
    with pytest.raises(ValueError):
        estimate_diagnostic_reliability(
            evidence_ready=True,
            mean_self_confidence=confidence,
            diagnostic_entropy=0.0,
        )


@pytest.mark.parametrize("entropy", [-0.01, 1.01])
def test_estimator_rejects_invalid_entropy(entropy):
    with pytest.raises(ValueError):
        estimate_diagnostic_reliability(
            evidence_ready=True,
            mean_self_confidence=0.5,
            diagnostic_entropy=entropy,
        )


@pytest.mark.parametrize("probability", [-0.01, 1.01])
def test_policy_rejects_invalid_probability(probability):
    with pytest.raises(ValueError):
        apply_reliability_policy(
            evidence_ready=True,
            predicted_reliability=probability,
        )

def _generation(
    diagnosis: DiagnosisLabel,
    confidence: float,
) -> ParsedDiagnosis:
    return ParsedDiagnosis(
        diagnosis=diagnosis,
        confidence=confidence,
        explanation="Test explanation",
    )


def test_substantive_modal_diagnosis_is_constructed():
    generations = [
        _generation(DiagnosisLabel.CPU_SATURATION, 0.90),
        _generation(DiagnosisLabel.CPU_SATURATION, 0.80),
        _generation(DiagnosisLabel.CPU_SATURATION, 0.85),
        _generation(DiagnosisLabel.CPU_SATURATION, 0.95),
        _generation(DiagnosisLabel.CPU_SATURATION, 0.90),
        _generation(DiagnosisLabel.NORMAL, 0.70),
        _generation(DiagnosisLabel.NORMAL, 0.75),
    ]

    result = assess_diagnostic_reliability(
        generations=generations,
        evidence_ready=True,
    )

    assert (
        result.modal_diagnosis
        is DiagnosisLabel.CPU_SATURATION
    )

    assert result.generation_count == 7
    assert result.modal_fraction == pytest.approx(5 / 7)

    assert result.mean_self_confidence == pytest.approx(
        sum(
            generation.confidence
            for generation in generations
        )
        / 7
    )

    assert result.predicted_reliability is not None


def test_modal_unknown_abstains_without_reliability_estimate():
    generations = [
        _generation(DiagnosisLabel.UNKNOWN, 0.60),
        _generation(DiagnosisLabel.UNKNOWN, 0.70),
        _generation(DiagnosisLabel.UNKNOWN, 0.65),
        _generation(DiagnosisLabel.UNKNOWN, 0.55),
        _generation(DiagnosisLabel.UNKNOWN, 0.60),
        _generation(DiagnosisLabel.CPU_SATURATION, 0.80),
        _generation(DiagnosisLabel.CPU_SATURATION, 0.85),
    ]

    result = assess_diagnostic_reliability(
        generations=generations,
        evidence_ready=True,
    )

    assert result.modal_diagnosis is DiagnosisLabel.UNKNOWN
    assert result.decision_state is DiagnosticDecisionState.ABSTAIN
    assert result.predicted_reliability is None
    assert result.is_abstention is True
    assert result.has_reliability_estimate is False


def test_non_actionable_substantive_diagnosis_is_withheld():
    generations = [
        _generation(DiagnosisLabel.NETWORK_DEGRADATION, 1.0)
        for _ in range(7)
    ]

    result = assess_diagnostic_reliability(
        generations=generations,
        evidence_ready=False,
    )

    assert (
        result.modal_diagnosis
        is DiagnosisLabel.NETWORK_DEGRADATION
    )

    assert result.predicted_reliability is not None
    assert result.decision_state is DiagnosticDecisionState.WITHHOLD


def test_modal_tie_is_rejected():
    generations = [
        _generation(DiagnosisLabel.CPU_SATURATION, 0.8),
        _generation(DiagnosisLabel.NORMAL, 0.8),
    ]

    with pytest.raises(
        ValueError,
        match="unique modal diagnosis",
    ):
        assess_diagnostic_reliability(
            generations=generations,
            evidence_ready=True,
        )


def test_empty_generation_sequence_is_rejected():
    with pytest.raises(
        ValueError,
        match="At least one parsed generation",
    ):
        assess_diagnostic_reliability(
            generations=[],
            evidence_ready=True,
        )

def _structured_response(
    diagnosis: str = "CPU_SATURATION",
    confidence: float = 0.9,
) -> dict:
    return {
        "message": {
            "content": (
                "{"
                f'"diagnosis":"{diagnosis}",'
                f'"confidence":{confidence},'
                '"explanation":"Evidence supports the diagnosis."'
                "}"
            )
        }
    }


def test_generator_uses_frozen_measurement_settings():
    llm_service = MagicMock()
    llm_service.model = "gemma3:12b"
    llm_service.config = {
        "llm": {
            "context_window": 4096,
        }
    }
    llm_service.client.chat.return_value = _structured_response()

    generator = DiagnosticGenerator(
        llm_service=llm_service,
    )

    generations = generator.generate_repeated(
        context="Operational evidence",
    )

    assert len(generations) == GENERATION_COUNT
    assert GENERATION_COUNT == 7
    assert DIAGNOSTIC_TEMPERATURE == 0.4

    assert (
        llm_service.client.chat.call_count
        == GENERATION_COUNT
    )

    for call in llm_service.client.chat.call_args_list:
        kwargs = call.kwargs

        assert kwargs["model"] == "gemma3:12b"

        assert (
            kwargs["options"]["temperature"]
            == 0.4
        )

        assert (
            kwargs["options"]["num_ctx"]
            == 4096
        )

        assert "format" in kwargs


def test_generator_returns_strictly_parsed_diagnoses():
    llm_service = MagicMock()
    llm_service.model = "gemma3:12b"
    llm_service.config = {
        "llm": {
            "context_window": 4096,
        }
    }
    llm_service.client.chat.return_value = _structured_response(
        diagnosis="NETWORK_DEGRADATION",
        confidence=0.85,
    )

    generator = DiagnosticGenerator(
        llm_service=llm_service,
    )

    generations = generator.generate_repeated(
        context="Operational evidence",
    )

    assert all(
        generation.diagnosis
        is DiagnosisLabel.NETWORK_DEGRADATION
        for generation in generations
    )

    assert all(
        generation.confidence
        == pytest.approx(0.85)
        for generation in generations
    )


def test_generator_rejects_empty_context():
    llm_service = MagicMock()

    generator = DiagnosticGenerator(
        llm_service=llm_service,
    )

    with pytest.raises(
        ValueError,
        match="context must not be empty",
    ):
        generator.generate_repeated(
            context="   ",
        )


def test_generator_raises_on_technical_failure():
    llm_service = MagicMock()
    llm_service.model = "gemma3:12b"
    llm_service.config = {
        "llm": {
            "context_window": 4096,
        }
    }

    llm_service.client.chat.side_effect = RuntimeError(
        "Ollama unavailable"
    )

    generator = DiagnosticGenerator(
        llm_service=llm_service,
    )

    with pytest.raises(
        DiagnosticGenerationError,
        match="generation failed",
    ):
        generator.generate_repeated(
            context="Operational evidence",
        )


def test_generator_raises_on_invalid_structured_output():
    llm_service = MagicMock()
    llm_service.model = "gemma3:12b"
    llm_service.config = {
        "llm": {
            "context_window": 4096,
        }
    }

    llm_service.client.chat.return_value = {
        "message": {
            "content": '{"diagnosis":"NOT_A_REAL_LABEL"}'
        }
    }

    generator = DiagnosticGenerator(
        llm_service=llm_service,
    )

    with pytest.raises(
        DiagnosticGenerationError,
        match="strict diagnostic parsing",
    ):
        generator.generate_repeated(
            context="Operational evidence",
        )

def _mock_llm_service_for_diagnosis(
    diagnosis: str,
    confidence: float,
):
    llm_service = MagicMock()
    llm_service.model = "gemma3:12b"
    llm_service.config = {
        "llm": {
            "context_window": 4096,
        }
    }

    llm_service.client.chat.return_value = _structured_response(
        diagnosis=diagnosis,
        confidence=confidence,
    )

    return llm_service


def test_feature_derives_actionable_evidence_from_envelope():
    llm_service = _mock_llm_service_for_diagnosis(
        diagnosis="CPU_SATURATION",
        confidence=0.95,
    )

    envelope = MagicMock()
    envelope.is_actionable = True

    feature = PostInferenceDiagnosticReliabilityService(
        llm_service=llm_service,
    )

    result = feature.assess(
        context="Operational evidence",
        envelope=envelope,
    )

    assert result.evidence_ready is True

    assert (
        result.modal_diagnosis
        is DiagnosisLabel.CPU_SATURATION
    )

    assert result.predicted_reliability is not None

    assert (
        llm_service.client.chat.call_count
        == GENERATION_COUNT
    )


def test_feature_preserves_non_actionable_validation():
    llm_service = _mock_llm_service_for_diagnosis(
        diagnosis="CPU_SATURATION",
        confidence=1.0,
    )

    envelope = MagicMock()
    envelope.is_actionable = False

    feature = PostInferenceDiagnosticReliabilityService(
        llm_service=llm_service,
    )

    result = feature.assess(
        context="Operational evidence",
        envelope=envelope,
    )

    assert result.evidence_ready is False

    assert result.predicted_reliability is not None

    assert (
        result.decision_state
        is DiagnosticDecisionState.WITHHOLD
    )


def test_feature_preserves_unknown_as_abstention():
    llm_service = _mock_llm_service_for_diagnosis(
        diagnosis="UNKNOWN",
        confidence=0.70,
    )

    envelope = MagicMock()
    envelope.is_actionable = True

    feature = PostInferenceDiagnosticReliabilityService(
        llm_service=llm_service,
    )

    result = feature.assess(
        context="Operational evidence",
        envelope=envelope,
    )

    assert (
        result.modal_diagnosis
        is DiagnosisLabel.UNKNOWN
    )

    assert (
        result.decision_state
        is DiagnosticDecisionState.ABSTAIN
    )

    assert result.predicted_reliability is None

    assert (
        llm_service.client.chat.call_count
        == GENERATION_COUNT
    )


def test_feature_rejects_empty_context_before_generation():
    llm_service = MagicMock()

    envelope = MagicMock()
    envelope.is_actionable = True

    feature = PostInferenceDiagnosticReliabilityService(
        llm_service=llm_service,
    )

    with pytest.raises(
        ValueError,
        match="context must not be empty",
    ):
        feature.assess(
            context=" ",
            envelope=envelope,
        )

    llm_service.client.chat.assert_not_called()

def _telemetry_row(
    *,
    timestamp,
    cpu_percent=50.0,
    memory_percent=60.0,
    disk_percent=40.0,
    network_latency_ms=10.0,
    error_rate=0.01,
    active_connections=5,
    queue_depth=2,
):
    return SimpleNamespace(
        timestamp=timestamp,
        cpu_percent=cpu_percent,
        memory_percent=memory_percent,
        disk_percent=disk_percent,
        network_latency_ms=network_latency_ms,
        error_rate=error_rate,
        active_connections=active_connections,
        queue_depth=queue_depth,
    )


def test_production_context_uses_frozen_telemetry_point_limit():
    assert TELEMETRY_POINT_LIMIT == 24


def test_production_context_preserves_frozen_evidence_format():
    from datetime import datetime

    telemetry = [
        _telemetry_row(
            timestamp=datetime(2026, 9, 17, 12, 5),
            cpu_percent=80.0,
        ),
        _telemetry_row(
            timestamp=datetime(2026, 9, 17, 12, 0),
            cpu_percent=70.0,
        ),
    ]

    context = build_diagnostic_context(
        device_id="DEVICE_01",
        telemetry=telemetry,
    )

    assert "[CURRENT SYSTEM STATUS]" in context
    assert "Focus device: DEVICE_01" in context
    assert "[OPERATIONAL TELEMETRY]" in context

    assert (
        "timestamp,cpu_percent,memory_percent,disk_percent,"
        "network_latency_ms,error_rate,active_connections,queue_depth"
        in context
    )

    assert context.index("2026-09-17T12:00:00") < context.index(
        "2026-09-17T12:05:00"
    )


def test_production_context_represents_missing_values_as_na():
    from datetime import datetime

    telemetry = [
        _telemetry_row(
            timestamp=datetime(2026, 9, 17, 12, 0),
            memory_percent=None,
        )
    ]

    context = build_diagnostic_context(
        device_id="DEVICE_01",
        telemetry=telemetry,
    )

    assert ",NA," in context


def test_model_visible_validation_rejects_unapproved_fields():
    validation = model_visible_validation(
        {
            "trust_mode": "grounded",
            "trust_score": 1.0,
            "actionable": True,
            "failed_gate_id": None,
            "gates": {"B": "PRIVATE DETAIL"},
            "private_ground_truth": "CPU_SATURATION",
        }
    )

    assert validation == {
        "trust_mode": "grounded",
        "trust_score": 1.0,
        "actionable": True,
        "failed_gate_id": None,
    }


def test_production_context_appends_only_approved_validation_fields():
    from datetime import datetime

    telemetry = [
        _telemetry_row(
            timestamp=datetime(2026, 9, 17, 12, 0),
        )
    ]

    context = build_diagnostic_context(
        device_id="DEVICE_01",
        telemetry=telemetry,
        validation={
            "trust_mode": "grounded",
            "trust_score": 1.0,
            "actionable": True,
            "failed_gate_id": None,
            "private_ground_truth": "CPU_SATURATION",
        },
    )

    assert "[VALIDATION STATUS]" in context
    assert "trust_mode: grounded" in context
    assert "trust_score: 1.0" in context
    assert "actionable: True" in context
    assert "failed_gate_id: None" in context

    assert "private_ground_truth" not in context
    assert "CPU_SATURATION" not in context


def test_production_context_rejects_empty_telemetry():
    with pytest.raises(
        ValueError,
        match="At least one telemetry observation is required",
    ):
        build_diagnostic_context(
            device_id="DEVICE_01",
            telemetry=[],
        )