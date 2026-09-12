"""Cross-taxonomy audit for controlled diagnostic scenarios.

This script performs the final A8.2 review across all 15 controlled scenario
families.

The purpose is not to build a classifier or perform unnecessary exploratory
analysis. It checks whether the scenario taxonomy is suitable for the planned
diagnostic-reliability experiment by examining:

1. cross-class signal ranges,
2. obvious single-threshold separability,
3. deliberately overlapping scenario families,
4. challenging NORMAL cases,
5. temporal persistence versus transient behaviour.

The audit uses a fixed seed and timestamp so that results are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import mean

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


SEED = 42

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
    """One scenario family and its private ground-truth class."""

    short_name: str
    fault: FaultLabel
    family: str


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


def _generate(spec: FamilySpec):
    """Generate one reproducible scenario for the supplied family."""

    return generate_scenario(
        spec.fault,
        scenario_family=spec.family,
        seed=SEED,
        end_time=END_TIME,
    )


def _values(scenario, attribute: str) -> list[float]:
    """Extract non-null numeric values for one telemetry attribute."""

    return [
        float(getattr(point, attribute))
        for point in scenario.telemetry
        if getattr(point, attribute) is not None
    ]


def _average(values: list[float]) -> float:
    """Return the arithmetic mean."""

    return mean(values)


def _count_above(values: list[float], threshold: float) -> int:
    """Count observations strictly above a threshold."""

    return sum(value > threshold for value in values)


def _first_mean(values: list[float], n: int = 6) -> float:
    """Mean of the first observations."""

    return mean(values[:n])


def _last_mean(values: list[float], n: int = 6) -> float:
    """Mean of the final observations."""

    return mean(values[-n:])


def print_family_summary() -> None:
    """Print a compact signal summary for every scenario family."""

    print()
    print("=" * 123)
    print("A8.2 CROSS-TAXONOMY FAMILY SUMMARY")
    print("=" * 123)

    print(
        "family | class                | "
        "cpu max | mem max | latency max | error max | "
        "queue max | CPU>80 | MEM>80 | LAT>500 | ERR>.05"
    )
    print("-" * 123)

    for spec in FAMILIES:
        scenario = _generate(spec)

        cpu = _values(scenario, "cpu_percent")
        memory = _values(scenario, "memory_percent")
        latency = _values(scenario, "network_latency_ms")
        errors = _values(scenario, "error_rate")
        queue = _values(scenario, "queue_depth")

        print(
            f"{spec.short_name:<6} | "
            f"{spec.fault.value:<20} | "
            f"{max(cpu):>7.1f} | "
            f"{max(memory):>7.1f} | "
            f"{max(latency):>11.1f} | "
            f"{max(errors):>9.4f} | "
            f"{max(queue):>9.0f} | "
            f"{_count_above(cpu, 80):>6} | "
            f"{_count_above(memory, 80):>6} | "
            f"{_count_above(latency, 500):>7} | "
            f"{_count_above(errors, 0.05):>7}"
        )


def print_temporal_summary() -> None:
    """Print early-versus-late behaviour for all families."""

    print()
    print("=" * 111)
    print("TEMPORAL SHAPE SUMMARY")
    print("=" * 111)

    print(
        "family | "
        "CPU early→late | "
        "MEM early→late | "
        "LAT early→late | "
        "ERR early→late | "
        "QUEUE early→late"
    )
    print("-" * 111)

    for spec in FAMILIES:
        scenario = _generate(spec)

        cpu = _values(scenario, "cpu_percent")
        memory = _values(scenario, "memory_percent")
        latency = _values(scenario, "network_latency_ms")
        errors = _values(scenario, "error_rate")
        queue = _values(scenario, "queue_depth")

        print(
            f"{spec.short_name:<6} | "
            f"{_first_mean(cpu):>5.1f}→{_last_mean(cpu):<5.1f} | "
            f"{_first_mean(memory):>5.1f}→{_last_mean(memory):<5.1f} | "
            f"{_first_mean(latency):>6.1f}→{_last_mean(latency):<6.1f} | "
            f"{_first_mean(errors):>6.3f}→"
            f"{_last_mean(errors):<6.3f} | "
            f"{_first_mean(queue):>4.1f}→"
            f"{_last_mean(queue):<4.1f}"
        )


def print_targeted_pair(
    left_spec: FamilySpec,
    right_spec: FamilySpec,
    reason: str,
) -> None:
    """Compare one deliberately important cross-taxonomy pair."""

    left = _generate(left_spec)
    right = _generate(right_spec)

    print()
    print("-" * 88)
    print(
        f"{left_spec.short_name} vs {right_spec.short_name}: "
        f"{reason}"
    )
    print("-" * 88)

    attributes = (
        ("cpu_percent", "CPU"),
        ("memory_percent", "Memory"),
        ("network_latency_ms", "Latency"),
        ("error_rate", "Error rate"),
        ("active_connections", "Connections"),
        ("queue_depth", "Queue"),
    )

    for attribute, label in attributes:
        left_values = _values(left, attribute)
        right_values = _values(right, attribute)

        print(
            f"{label:<12} "
            f"{left_spec.short_name}: "
            f"mean={_average(left_values):>7.2f}, "
            f"max={max(left_values):>7.2f}, "
            f"late={_last_mean(left_values):>7.2f} | "
            f"{right_spec.short_name}: "
            f"mean={_average(right_values):>7.2f}, "
            f"max={max(right_values):>7.2f}, "
            f"late={_last_mean(right_values):>7.2f}"
        )


def print_targeted_comparisons() -> None:
    """Inspect the cross-class pairs identified during scenario design."""

    lookup = {
        spec.short_name: spec
        for spec in FAMILIES
    }

    print()
    print("=" * 88)
    print("TARGETED CROSS-TAXONOMY COMPARISONS")
    print("=" * 88)

    print_targeted_pair(
        lookup["CS-3"],
        lookup["N-2"],
        "high concurrency/CPU pressure versus legitimate high workload",
    )

    print_targeted_pair(
        lookup["ND-2"],
        lookup["N-3"],
        "intermittent network degradation versus benign transient",
    )

    print_targeted_pair(
        lookup["ND-3"],
        lookup["AF-2"],
        "network degradation with errors versus dependency failure",
    )

    print_targeted_pair(
        lookup["MP-3"],
        lookup["CS-3"],
        "memory/queue pressure versus CPU/concurrency pressure",
    )

    print_targeted_pair(
        lookup["AF-3"],
        lookup["N-3"],
        "application error bursts versus benign transient",
    )


def print_normal_challenge_check() -> None:
    """Summarise why N-2 and N-3 are non-trivial normal cases."""

    lookup = {
        spec.short_name: spec
        for spec in FAMILIES
    }

    print()
    print("=" * 88)
    print("NORMAL CHALLENGE CHECK")
    print("=" * 88)

    for name in ("N-1", "N-2", "N-3"):
        spec = lookup[name]
        scenario = _generate(spec)

        cpu = _values(scenario, "cpu_percent")
        memory = _values(scenario, "memory_percent")
        latency = _values(scenario, "network_latency_ms")
        errors = _values(scenario, "error_rate")
        queue = _values(scenario, "queue_depth")

        print()
        print(name)
        print(
            f"  CPU:     mean={mean(cpu):.1f}, max={max(cpu):.1f}, "
            f">80 count={_count_above(cpu, 80)}"
        )
        print(
            f"  Memory:  mean={mean(memory):.1f}, max={max(memory):.1f}, "
            f">80 count={_count_above(memory, 80)}"
        )
        print(
            f"  Latency: mean={mean(latency):.1f}, "
            f"max={max(latency):.1f}, "
            f">250 count={_count_above(latency, 250)}"
        )
        print(
            f"  Errors:  mean={mean(errors):.4f}, "
            f"max={max(errors):.4f}, "
            f">.02 count={_count_above(errors, 0.02)}"
        )
        print(
            f"  Queue:   mean={mean(queue):.1f}, "
            f"max={max(queue):.0f}"
        )

        if name == "N-3":
            print(
                "  Recovery:"
                f" late CPU mean={_last_mean(cpu):.1f},"
                f" late latency mean={_last_mean(latency):.1f},"
                f" late error mean={_last_mean(errors):.4f}"
            )


def main() -> None:
    """Run the complete final A8.2 cross-taxonomy audit."""

    print_family_summary()
    print_temporal_summary()
    print_targeted_comparisons()
    print_normal_challenge_check()


if __name__ == "__main__":
    main()