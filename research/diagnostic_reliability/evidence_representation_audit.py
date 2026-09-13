"""Audit the evidence presented in the A8.4d confirmation pilot.

This script is deterministic and makes no LLM calls.

Its purpose is to investigate why several diagnostic classes were poorly
recognised during the confirmation pilot by answering two preliminary
questions:

1. Are the intended diagnostic distinctions actually present in the generated
   COMPLETE_FRESH telemetry?
2. Are the relevant telemetry variables represented in the model-visible
   context?

The script does not alter the scenario generators, Validation Envelope,
diagnostic prompt, or experimental configuration.
"""

from __future__ import annotations

from datetime import datetime, timezone
from statistics import mean

from .context import build_model_context
from .evidence_conditions import apply_evidence_condition
from .scenarios import generate_scenario
from .taxonomy import EvidenceCondition, FaultLabel


DEVICE_ID = "EXP_DEVICE_01"

# Match the exact seeds used by the A8.4d confirmation pilot.
PILOT_CASES = (
    (
        FaultLabel.NORMAL,
        20260921,
    ),
    (
        FaultLabel.CPU_SATURATION,
        20260922,
    ),
    (
        FaultLabel.MEMORY_PRESSURE,
        20260923,
    ),
    (
        FaultLabel.NETWORK_DEGRADATION,
        20260924,
    ),
    (
        FaultLabel.APPLICATION_FAILURE,
        20260925,
    ),
)

# A fixed timestamp makes the audit reproducible. The scenario values depend
# on the seed rather than the absolute timestamp.
FIXED_END_TIME = datetime(
    2026,
    9,
    12,
    23,
    30,
    0,
    tzinfo=timezone.utc,
)

METRICS = (
    (
        "cpu_percent",
        "CPU %",
    ),
    (
        "memory_percent",
        "Memory %",
    ),
    (
        "network_latency_ms",
        "Latency ms",
    ),
    (
        "error_rate",
        "Error rate",
    ),
    (
        "active_connections",
        "Connections",
    ),
    (
        "queue_depth",
        "Queue depth",
    ),
)

MODEL_CONTEXT_TERMS = {
    "cpu_percent": (
        "cpu",
    ),
    "memory_percent": (
        "memory",
    ),
    "network_latency_ms": (
        "latency",
    ),
    "error_rate": (
        "error",
    ),
    "active_connections": (
        "connection",
    ),
    "queue_depth": (
        "queue",
    ),
}


def _numeric_values(
    scenario,
    attribute: str,
) -> list[float]:
    """Return non-null numeric values for one telemetry attribute."""

    values = []

    for point in scenario.telemetry:
        value = getattr(
            point,
            attribute,
        )

        if value is not None:
            values.append(
                float(value)
            )

    if not values:
        raise ValueError(
            f"No values available for "
            f"{attribute} in "
            f"{scenario.scenario_id}"
        )

    return values


def _summarise_values(
    values: list[float],
) -> dict[str, float]:
    """Return simple temporal and distribution summaries."""

    midpoint = max(
        1,
        len(values) // 2,
    )

    first_half = values[
        :midpoint
    ]

    second_half = values[
        midpoint:
    ]

    if not second_half:
        second_half = first_half

    return {
        "start": values[0],
        "end": values[-1],
        "change": (
            values[-1]
            - values[0]
        ),
        "minimum": min(values),
        "maximum": max(values),
        "mean": mean(values),
        "first_half_mean": (
            mean(first_half)
        ),
        "second_half_mean": (
            mean(second_half)
        ),
        "half_change": (
            mean(second_half)
            - mean(first_half)
        ),
    }


def _context_presence(
    model_context: str,
) -> dict[str, bool]:
    """Check whether each diagnostic variable is represented in context."""

    lower_context = (
        model_context.lower()
    )

    return {
        attribute: any(
            term in lower_context
            for term in terms
        )
        for attribute, terms
        in MODEL_CONTEXT_TERMS.items()
    }


def _print_metric_summary(
    label: str,
    summary: dict[str, float],
) -> None:
    """Print one compact metric summary."""

    print(
        f"  {label:<13} "
        f"start={summary['start']:>9.3f}  "
        f"end={summary['end']:>9.3f}  "
        f"change={summary['change']:>9.3f}  "
        f"min={summary['minimum']:>9.3f}  "
        f"max={summary['maximum']:>9.3f}  "
        f"mean={summary['mean']:>9.3f}"
    )

    print(
        f"{'':17}"
        f"first-half mean="
        f"{summary['first_half_mean']:>9.3f}  "
        f"second-half mean="
        f"{summary['second_half_mean']:>9.3f}  "
        f"half-change="
        f"{summary['half_change']:>9.3f}"
    )


def main() -> int:
    """Run the evidence and representation audit."""

    print(
        "=" * 88
    )

    print(
        "A8.4 EVIDENCE REPRESENTATION AUDIT"
    )

    print(
        "=" * 88
    )

    print(
        "\nThis audit makes no LLM calls and does not modify "
        "the scenario generators."
    )

    all_context_checks = []

    for fault, seed in PILOT_CASES:
        base = generate_scenario(
            fault,
            instance=1,
            seed=seed,
            end_time=FIXED_END_TIME,
            device_id=DEVICE_ID,
        )

        scenario = (
            apply_evidence_condition(
                base,
                EvidenceCondition.COMPLETE_FRESH,
            )
        )

        model_context = (
            build_model_context(
                scenario
            )
        )

        print(
            "\n"
            + "-" * 88
        )

        print(
            f"Ground truth: "
            f"{scenario.fault.value}"
        )

        print(
            f"Scenario family: "
            f"{scenario.scenario_family}"
        )

        print(
            f"Scenario ID: "
            f"{scenario.scenario_id}"
        )

        print(
            f"Telemetry points: "
            f"{len(scenario.telemetry)}"
        )

        print(
            "\nTELEMETRY SUMMARY"
        )

        for attribute, label in METRICS:
            values = _numeric_values(
                scenario,
                attribute,
            )

            summary = (
                _summarise_values(
                    values
                )
            )

            _print_metric_summary(
                label,
                summary,
            )

        presence = (
            _context_presence(
                model_context
            )
        )

        all_context_checks.append(
            (
                scenario.fault.value,
                presence,
            )
        )

        print(
            "\nMODEL-CONTEXT VARIABLE PRESENCE"
        )

        for attribute, present in presence.items():
            print(
                f"  {attribute:<24} "
                f"{'PRESENT' if present else 'MISSING'}"
            )

        print(
            "\nMODEL CONTEXT SIZE"
        )

        print(
            f"  Characters: "
            f"{len(model_context)}"
        )

        print(
            f"  Lines: "
            f"{len(model_context.splitlines())}"
        )

    print(
        "\n"
        + "=" * 88
    )

    print(
        "REPRESENTATION-INTEGRITY SUMMARY"
    )

    print(
        "=" * 88
    )

    all_present = True

    for fault, presence in all_context_checks:
        missing = [
            attribute
            for attribute, present
            in presence.items()
            if not present
        ]

        if missing:
            all_present = False

            print(
                f"{fault}: MISSING "
                f"{', '.join(missing)}"
            )

        else:
            print(
                f"{fault}: all audited "
                "diagnostic variables represented"
            )

    print()

    if all_present:
        print(
            "Representation-integrity check: PASS"
        )
    else:
        print(
            "Representation-integrity check: REVIEW REQUIRED"
        )

    print(
        "\nInterpretation must be performed manually. "
        "This script does not declare the benchmark valid "
        "or invalid and does not modify any experimental "
        "configuration."
    )

    print(
        "\nA8.4 EVIDENCE REPRESENTATION AUDIT: COMPLETE"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )