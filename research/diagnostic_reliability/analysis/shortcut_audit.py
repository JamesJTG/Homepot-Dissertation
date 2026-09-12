"""Multi-seed single-variable shortcut audit for A8.2.

This script tests whether the controlled diagnostic benchmark contains
class-exclusive threshold shortcuts.

It deliberately does NOT train a machine-learning classifier. The question is
narrower:

    Can simple one-variable threshold rules identify fault classes across many
    generated scenario instances?

If a threshold event occurs in all or nearly all scenarios of one class and
never occurs in competing classes, the benchmark may be too easy for the
planned LLM diagnostic experiment.

The audit therefore evaluates threshold-event prevalence across multiple
scenario families and random seeds.

This is a benchmark-design diagnostic only. It is not part of the final
diagnostic reliability estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from research.diagnostic_reliability.scenarios import generate_scenario
from research.diagnostic_reliability.scenarios.application_failure import (
    AF1_FAMILY,
    AF2_FAMILY,
    AF3_FAMILY,
)
from research.diagnostic_reliability.scenarios.cpu_saturation import (
    CS1_FAMILY,
    CS2_FAMILY,
    CS3_FAMILY,
)
from research.diagnostic_reliability.scenarios.memory_pressure import (
    MP1_FAMILY,
    MP2_FAMILY,
    MP3_FAMILY,
)
from research.diagnostic_reliability.scenarios.network_degradation import (
    ND1_FAMILY,
    ND2_FAMILY,
    ND3_FAMILY,
)
from research.diagnostic_reliability.scenarios.normal import (
    N1_FAMILY,
    N2_FAMILY,
    N3_FAMILY,
)
from research.diagnostic_reliability.taxonomy import FaultLabel


N_SEEDS = 30

END_TIME = datetime(
    2026,
    9,
    11,
    12,
    0,
    tzinfo=timezone.utc,
)


@dataclass(frozen=True)
class FamilySpec:
    """One controlled scenario family."""

    short_name: str
    fault: FaultLabel
    family: str


@dataclass(frozen=True)
class ThresholdRule:
    """One simple threshold event used to probe for shortcuts."""

    name: str
    attribute: str
    threshold: float


FAMILIES = (
    FamilySpec("CS-1", FaultLabel.CPU_SATURATION, CS1_FAMILY),
    FamilySpec("CS-2", FaultLabel.CPU_SATURATION, CS2_FAMILY),
    FamilySpec("CS-3", FaultLabel.CPU_SATURATION, CS3_FAMILY),

    FamilySpec(
        "ND-1",
        FaultLabel.NETWORK_DEGRADATION,
        ND1_FAMILY,
    ),
    FamilySpec(
        "ND-2",
        FaultLabel.NETWORK_DEGRADATION,
        ND2_FAMILY,
    ),
    FamilySpec(
        "ND-3",
        FaultLabel.NETWORK_DEGRADATION,
        ND3_FAMILY,
    ),

    FamilySpec("MP-1", FaultLabel.MEMORY_PRESSURE, MP1_FAMILY),
    FamilySpec("MP-2", FaultLabel.MEMORY_PRESSURE, MP2_FAMILY),
    FamilySpec("MP-3", FaultLabel.MEMORY_PRESSURE, MP3_FAMILY),

    FamilySpec(
        "AF-1",
        FaultLabel.APPLICATION_FAILURE,
        AF1_FAMILY,
    ),
    FamilySpec(
        "AF-2",
        FaultLabel.APPLICATION_FAILURE,
        AF2_FAMILY,
    ),
    FamilySpec(
        "AF-3",
        FaultLabel.APPLICATION_FAILURE,
        AF3_FAMILY,
    ),

    FamilySpec("N-1", FaultLabel.NORMAL, N1_FAMILY),
    FamilySpec("N-2", FaultLabel.NORMAL, N2_FAMILY),
    FamilySpec("N-3", FaultLabel.NORMAL, N3_FAMILY),
)


RULES = (
    ThresholdRule(
        name="CPU > 80",
        attribute="cpu_percent",
        threshold=80.0,
    ),
    ThresholdRule(
        name="CPU > 90",
        attribute="cpu_percent",
        threshold=90.0,
    ),
    ThresholdRule(
        name="Memory > 80",
        attribute="memory_percent",
        threshold=80.0,
    ),
    ThresholdRule(
        name="Memory > 90",
        attribute="memory_percent",
        threshold=90.0,
    ),
    ThresholdRule(
        name="Latency > 250 ms",
        attribute="network_latency_ms",
        threshold=250.0,
    ),
    ThresholdRule(
        name="Latency > 500 ms",
        attribute="network_latency_ms",
        threshold=500.0,
    ),
    ThresholdRule(
        name="Error rate > 0.02",
        attribute="error_rate",
        threshold=0.02,
    ),
    ThresholdRule(
        name="Error rate > 0.05",
        attribute="error_rate",
        threshold=0.05,
    ),
)


def generate_instances(
    spec: FamilySpec,
) -> list:
    """Generate reproducible instances for one family."""

    return [
        generate_scenario(
            spec.fault,
            instance=seed,
            seed=seed,
            scenario_family=spec.family,
            end_time=END_TIME,
        )
        for seed in range(1, N_SEEDS + 1)
    ]


def scenario_triggers_rule(
    scenario,
    rule: ThresholdRule,
) -> bool:
    """Return whether any observation crosses the supplied threshold."""

    values = [
        getattr(point, rule.attribute)
        for point in scenario.telemetry
        if getattr(point, rule.attribute) is not None
    ]

    return any(
        float(value) > rule.threshold
        for value in values
    )


def prevalence(
    scenarios: list,
    rule: ThresholdRule,
) -> float:
    """Return proportion of scenarios triggering a threshold event."""

    triggered = sum(
        scenario_triggers_rule(scenario, rule)
        for scenario in scenarios
    )

    return triggered / len(scenarios)


def build_results() -> dict[str, dict[str, float]]:
    """Calculate threshold prevalence for every family."""

    results: dict[str, dict[str, float]] = {}

    for spec in FAMILIES:
        scenarios = generate_instances(spec)

        results[spec.short_name] = {
            rule.name: prevalence(scenarios, rule)
            for rule in RULES
        }

    return results


def print_family_prevalence(
    results: dict[str, dict[str, float]],
) -> None:
    """Print threshold-event prevalence by scenario family."""

    print()
    print("=" * 118)
    print(
        f"A8.2 MULTI-SEED SHORTCUT AUDIT "
        f"({N_SEEDS} instances per family)"
    )
    print("=" * 118)

    header = "family"

    for rule in RULES:
        header += f" | {rule.name:^14}"

    print(header)
    print("-" * 150)

    for spec in FAMILIES:
        row = f"{spec.short_name:<6}"

        for rule in RULES:
            value = results[spec.short_name][rule.name]
            row += f" | {value:>13.1%}"

        print(row)


def print_class_prevalence(
    results: dict[str, dict[str, float]],
) -> None:
    """Aggregate family prevalence by private ground-truth class."""

    print()
    print("=" * 118)
    print("GROUND-TRUTH CLASS PREVALENCE")
    print("=" * 118)

    classes = (
        FaultLabel.CPU_SATURATION,
        FaultLabel.NETWORK_DEGRADATION,
        FaultLabel.MEMORY_PRESSURE,
        FaultLabel.APPLICATION_FAILURE,
        FaultLabel.NORMAL,
    )

    header = "class"

    for rule in RULES:
        header += f" | {rule.name:^14}"

    print(header)
    print("-" * 165)

    for fault in classes:
        relevant = [
            spec
            for spec in FAMILIES
            if spec.fault == fault
        ]

        row = f"{fault.value:<20}"

        for rule in RULES:
            family_values = [
                results[spec.short_name][rule.name]
                for spec in relevant
            ]

            class_value = (
                sum(family_values)
                / len(family_values)
            )

            row += f" | {class_value:>13.1%}"

        print(row)


def print_exclusivity_warnings(
    results: dict[str, dict[str, float]],
) -> None:
    """Flag thresholds that behave like class-exclusive shortcuts."""

    print()
    print("=" * 90)
    print("POTENTIAL SHORTCUT WARNINGS")
    print("=" * 90)

    classes = (
        FaultLabel.CPU_SATURATION,
        FaultLabel.NETWORK_DEGRADATION,
        FaultLabel.MEMORY_PRESSURE,
        FaultLabel.APPLICATION_FAILURE,
        FaultLabel.NORMAL,
    )

    warning_count = 0

    for rule in RULES:
        class_rates: dict[FaultLabel, float] = {}

        for fault in classes:
            relevant = [
                spec
                for spec in FAMILIES
                if spec.fault == fault
            ]

            class_rates[fault] = (
                sum(
                    results[spec.short_name][rule.name]
                    for spec in relevant
                )
                / len(relevant)
            )

        sorted_rates = sorted(
            class_rates.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        highest_class, highest_rate = sorted_rates[0]
        second_rate = sorted_rates[1][1]

        if (
            highest_rate >= 0.90
            and second_rate <= 0.10
        ):
            warning_count += 1

            print(
                f"WARNING: {rule.name:<20} "
                f"appears highly exclusive to "
                f"{highest_class.value}: "
                f"{highest_rate:.1%} vs next-highest "
                f"{second_rate:.1%}"
            )

    if warning_count == 0:
        print(
            "No threshold met the current high-exclusivity "
            "warning criterion."
        )


def print_interpretation_note() -> None:
    """Print the intended interpretation of this audit."""

    print()
    print("=" * 90)
    print("INTERPRETATION")
    print("=" * 90)
    print(
        "A warning does not automatically mean a scenario family is invalid."
    )
    print(
        "It indicates that a simple single-variable threshold may provide "
        "a class shortcut."
    )
    print(
        "Any refinement should preserve causal and temporal plausibility "
        "rather than adding arbitrary noise solely to defeat the threshold."
    )


def main() -> None:
    """Run the complete multi-seed shortcut audit."""

    results = build_results()

    print_family_prevalence(results)
    print_class_prevalence(results)
    print_exclusivity_warnings(results)
    print_interpretation_note()


if __name__ == "__main__":
    main()