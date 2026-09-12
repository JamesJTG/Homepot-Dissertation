"""Live checkpoint for the complete HOMEPOT Validation Envelope.

This checkpoint verifies that the controlled diagnostic experiment uses the
real production A→E Validation Envelope rather than treating Gate B alone as
the upstream evidence-readiness signal.

Expected experimental behaviour:

COMPLETE_FRESH
    A PASS
    B PASS
    C PASS
    D PASS
    E PASS

STALE
    A PASS
    B FAIL because freshness is insufficient

INCOMPLETE
    A PASS
    B FAIL because completeness is insufficient

DISCONTINUOUS
    A PASS
    B FAIL because continuity is insufficient

Only evidence quality changes between the four cases.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json

from homepot.database import get_database_service

from .context import build_model_context
from .evidence_conditions import apply_evidence_condition
from .scenarios import generate_scenario
from .taxonomy import EvidenceCondition, FaultLabel
from .validation_integration import (
    RESEARCH_DEVICE_ID,
    envelope_summary,
    evaluate_validation_envelope,
    materialise_scenario,
    model_visible_validation,
    provision_experimental_device,
)


FAULT = FaultLabel.CPU_SATURATION
INSTANCE = 1
SEED = 20260912

EVIDENCE_CONDITIONS = (
    EvidenceCondition.COMPLETE_FRESH,
    EvidenceCondition.STALE,
    EvidenceCondition.INCOMPLETE,
    EvidenceCondition.DISCONTINUOUS,
)


def _assert_gate_status(
    summary: dict,
    gate_id: str,
    expected: str,
) -> None:
    """Assert one gate exists and has the expected status."""

    actual = (
        summary
        .get("gates", {})
        .get(gate_id, {})
        .get("status")
    )

    if actual != expected:
        raise AssertionError(
            f"Expected Gate {gate_id}={expected}, "
            f"got {actual!r}."
        )


def _assert_complete_fresh(
    summary: dict,
) -> None:
    """Validate the expected full-envelope success case."""

    for gate_id in ("A", "B", "C", "D", "E"):
        _assert_gate_status(
            summary,
            gate_id,
            "pass",
        )

    if not summary["actionable"]:
        raise AssertionError(
            "COMPLETE_FRESH should be actionable."
        )

    if summary["failed_gate_id"] is not None:
        raise AssertionError(
            "COMPLETE_FRESH should not have a failed gate."
        )

    expected_passed = [
        "A",
        "B",
        "C",
        "D",
        "E",
    ]

    if summary["passed_gate_ids"] != expected_passed:
        raise AssertionError(
            "COMPLETE_FRESH did not pass the full A→E sequence: "
            f"{summary['passed_gate_ids']}"
        )


def _assert_gate_b_failure(
    summary: dict,
    *,
    required_check_id: str,
) -> None:
    """Validate an evidence-quality failure at Gate B."""

    _assert_gate_status(
        summary,
        "A",
        "pass",
    )

    _assert_gate_status(
        summary,
        "B",
        "fail",
    )

    if summary["failed_gate_id"] != "B":
        raise AssertionError(
            "Expected Validation Envelope to stop at Gate B, "
            f"got failed_gate_id={summary['failed_gate_id']!r}."
        )

    if summary["actionable"]:
        raise AssertionError(
            "A Gate B failure must not be actionable."
        )

    # Because the envelope stops at the first failed gate,
    # downstream gates must not have been evaluated.
    downstream = {
        gate_id
        for gate_id in ("C", "D", "E")
        if gate_id in summary["gates"]
    }

    if downstream:
        raise AssertionError(
            "Validation Envelope evaluated downstream gates "
            f"after Gate B failure: {sorted(downstream)}"
        )

    checks = summary["gates"]["B"]["checks"]

    check = checks.get(required_check_id)

    if check is None:
        raise AssertionError(
            f"Expected Gate B check {required_check_id!r} "
            "was not present."
        )

    if check["passed"]:
        raise AssertionError(
            f"Expected {required_check_id} to fail."
        )


async def main() -> int:
    """Run the live full-envelope checkpoint."""

    db = await get_database_service()

    async with db.get_session() as session:
        device = await provision_experimental_device(
            session
        )

        device_int_id = device.id
        device_id = device.device_id

    if device_id != RESEARCH_DEVICE_ID:
        raise RuntimeError(
            "Unexpected research device identity."
        )

    base = generate_scenario(
        FAULT,
        instance=INSTANCE,
        seed=SEED,
        end_time=datetime.now(timezone.utc),
        device_id=device_id,
    )

    report = {
        "checkpoint": "full_validation_envelope",
        "fault_used_for_checkpoint": FAULT.value,
        "scenario_family": base.scenario_family,
        "seed": SEED,
        "conditions": {},
    }

    for condition in EVIDENCE_CONDITIONS:
        scenario = apply_evidence_condition(
            base,
            condition,
        )

        # Gate C validates the same structured context contract that will
        # ultimately be supplied to the LLM. Validation fields cannot be
        # included until after the envelope has produced them.
        pre_validation_context = build_model_context(
            scenario
        )

        async with db.get_session() as session:
            await materialise_scenario(
                session,
                scenario,
                device_int_id=device_int_id,
            )

            result = await evaluate_validation_envelope(
                session,
                device_id=device_id,
                device_int_id=device_int_id,
                assembled_context=pre_validation_context,
            )

        summary = envelope_summary(
            result
        )

        # Build the eventual model-visible context using only the approved
        # upstream readiness fields. This is not sent to the LLM here.
        visible_validation = model_visible_validation(
            result
        )

        model_context = build_model_context(
            scenario,
            validation=visible_validation,
        )

        leakage_checks = {
            "scenario_id_absent": (
                scenario.scenario_id
                not in model_context
            ),
            "scenario_family_absent": (
                scenario.scenario_family
                not in model_context
            ),
            "private_fault_absent": (
                f"fault: {scenario.fault.value}"
                not in model_context
                and
                f"fault={scenario.fault.value}"
                not in model_context
            ),
        }

        if not all(
            leakage_checks.values()
        ):
            raise AssertionError(
                "Model-context leakage check failed."
            )

        report["conditions"][
            condition.value
        ] = {
            "envelope": summary,
            "model_visible_validation": (
                visible_validation
            ),
            "leakage_checks": leakage_checks,
        }

    complete = report["conditions"][
        EvidenceCondition.COMPLETE_FRESH.value
    ]["envelope"]

    stale = report["conditions"][
        EvidenceCondition.STALE.value
    ]["envelope"]

    incomplete = report["conditions"][
        EvidenceCondition.INCOMPLETE.value
    ]["envelope"]

    discontinuous = report["conditions"][
        EvidenceCondition.DISCONTINUOUS.value
    ]["envelope"]

    _assert_complete_fresh(
        complete
    )

    _assert_gate_b_failure(
        stale,
        required_check_id="B.freshness",
    )

    _assert_gate_b_failure(
        incomplete,
        required_check_id="B.completeness",
    )

    _assert_gate_b_failure(
        discontinuous,
        required_check_id="B.continuity",
    )

    print(
        "=" * 72
    )
    print(
        "FULL VALIDATION ENVELOPE CHECKPOINT"
    )
    print(
        "=" * 72
    )

    print(
        json.dumps(
            report,
            indent=2,
            default=str,
        )
    )

    print(
        "\nSUMMARY"
    )
    print(
        "-" * 72
    )

    for condition in EVIDENCE_CONDITIONS:
        summary = (
            report["conditions"]
            [condition.value]
            ["envelope"]
        )

        evaluated = ", ".join(
            (
                f"{gate_id}:"
                f"{gate['status'].upper()}"
            )
            for gate_id, gate
            in summary["gates"].items()
        )

        print(
            f"{condition.value}: "
            f"{evaluated} | "
            f"trust_mode="
            f"{summary['trust_mode']} | "
            f"trust_score="
            f"{summary['trust_score']:.3f} | "
            f"actionable="
            f"{summary['actionable']} | "
            f"failed_gate="
            f"{summary['failed_gate_id']}"
        )

    print(
        "\nFULL VALIDATION ENVELOPE CHECKPOINT: PASS"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        asyncio.run(
            main()
        )
    )