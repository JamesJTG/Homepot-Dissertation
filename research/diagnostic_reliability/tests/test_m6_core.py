"""M6 vertical-slice tests that do not require a live DB or Ollama service."""

from datetime import datetime, timezone
import json

import pytest

from research.diagnostic_reliability.context import build_model_context
from research.diagnostic_reliability.evidence_conditions import (
    apply_evidence_condition,
    expected_sufficiency,
)
from research.diagnostic_reliability.parser import parse_diagnosis
from research.diagnostic_reliability.runner import run_repeated_generations
from research.diagnostic_reliability.scenarios import generate_scenario
from research.diagnostic_reliability.scoring import ErrorCategory, score_diagnosis
from research.diagnostic_reliability.taxonomy import (
    DiagnosisLabel,
    EvidenceCondition,
    EvidenceSufficiency,
    FaultLabel,
)
from research.diagnostic_reliability.uncertainty import diagnostic_entropy, modal_fraction


def _fixed_scenario(fault=FaultLabel.CPU_SATURATION):
    return generate_scenario(
        fault,
        seed=42,
        end_time=datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc),
    )


def test_scenario_is_deterministic():
    a = _fixed_scenario()
    b = _fixed_scenario()
    assert a == b


def test_stale_transform_preserves_fault_pattern_but_moves_timestamps():
    fresh = _fixed_scenario()
    stale = apply_evidence_condition(fresh, EvidenceCondition.STALE)
    assert stale.fault == fresh.fault
    assert stale.telemetry[-1].cpu_percent == fresh.telemetry[-1].cpu_percent
    assert stale.telemetry[-1].timestamp < fresh.telemetry[-1].timestamp


def test_ground_truth_not_in_model_context():
    scenario = _fixed_scenario()
    context = build_model_context(
        scenario,
        validation={"trust_mode": "grounded", "trust_score": 1.0, "actionable": True},
    )
    assert scenario.fault.value not in context
    assert "[OPERATIONAL TELEMETRY]" in context


def test_expected_sufficiency():
    assert expected_sufficiency(EvidenceCondition.COMPLETE_FRESH) is EvidenceSufficiency.SUFFICIENT
    assert expected_sufficiency(EvidenceCondition.STALE) is EvidenceSufficiency.INSUFFICIENT


def test_parser_accepts_valid_response():
    parsed = parse_diagnosis(json.dumps({
        "diagnosis": "CPU_SATURATION",
        "confidence": 0.81,
        "explanation": "Sustained CPU utilisation is above the operational threshold."
    }))
    assert parsed.diagnosis is DiagnosisLabel.CPU_SATURATION
    assert parsed.confidence == 0.81


def test_parser_rejects_unknown_label():
    with pytest.raises(ValueError):
        parse_diagnosis(json.dumps({
            "diagnosis": "MAGIC_FAULT",
            "confidence": 0.5,
            "explanation": "x"
        }))


def test_scoring_separates_correctness_from_appropriate_abstention():
    correct = score_diagnosis(
        FaultLabel.CPU_SATURATION,
        DiagnosisLabel.CPU_SATURATION,
        EvidenceSufficiency.SUFFICIENT,
    )
    assert correct.diagnostic_correctness == 1
    assert correct.error_category is ErrorCategory.CORRECT

    abstain = score_diagnosis(
        FaultLabel.CPU_SATURATION,
        DiagnosisLabel.UNKNOWN,
        EvidenceSufficiency.INSUFFICIENT,
    )
    assert abstain.diagnostic_correctness is None
    assert abstain.decision_appropriateness == 1
    assert abstain.error_category is ErrorCategory.APPROPRIATE_ABSTENTION


def test_entropy_is_zero_for_complete_agreement():
    labels = [DiagnosisLabel.CPU_SATURATION] * 5
    assert diagnostic_entropy(labels) == 0.0
    assert modal_fraction(labels) == 1.0


def test_entropy_detects_disagreement():
    labels = [
        DiagnosisLabel.CPU_SATURATION,
        DiagnosisLabel.CPU_SATURATION,
        DiagnosisLabel.MEMORY_PRESSURE,
        DiagnosisLabel.NETWORK_DEGRADATION,
    ]
    assert diagnostic_entropy(labels) > 0.0
    assert modal_fraction(labels) == 0.5


def test_repeated_runner_reuses_same_context():
    calls = []
    def fake_generate(prompt, context, system_prompt):
        calls.append((prompt, context, system_prompt))
        return json.dumps({
            "diagnosis": "CPU_SATURATION",
            "confidence": 0.8,
            "explanation": "CPU evidence is persistently elevated."
        })

    context = "fixed-context"
    results = run_repeated_generations(fake_generate, context=context, n_generations=3)
    assert len(results) == 3
    assert [call[1] for call in calls] == [context, context, context]


def test_gate_b_summary_preserves_check_level_evidence():
    from ai.gates.base import CheckResult, EvidenceRef, GateResult, GateStatus
    from research.diagnostic_reliability.validation_integration import gate_b_summary

    result = GateResult(
        gate_id="B",
        name="Data Integrity",
        status=GateStatus.FAIL,
        checks=[
            CheckResult(
                check_id="B.freshness",
                name="Freshness",
                passed=False,
                message="stale",
                evidence=[EvidenceRef(table="device_metrics", threshold=300)],
            )
        ],
    )
    summary = gate_b_summary(result)
    assert summary["status"] == "fail"
    assert summary["checks"]["B.freshness"]["passed"] is False
    assert summary["checks"]["B.freshness"]["evidence"][0]["threshold"] == 300
