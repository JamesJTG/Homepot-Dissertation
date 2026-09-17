"""Check empirical compatibility of the frozen A8.7 scenario partition.

This script does not create or modify the partition and does not fit any
reliability model. It joins the frozen scenario assignment to the frozen
A8.6 case-level analytical dataset and reports the resulting modelling
sample structure.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from pathlib import Path


A8_6_DATASET = Path(
    "research/diagnostic_reliability/analysis/outputs/"
    "a8_6_outcome_uncertainty_construction/"
    "a8_6_evidence_case_reliability_dataset.csv"
)

PARTITION_FILE = Path(
    "research/diagnostic_reliability/analysis/outputs/"
    "a8_7_reliability_modelling/"
    "a8_7_scenario_development_holdout_assignment.csv"
)


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def find_column(rows: list[dict], candidates: list[str]) -> str:
    if not rows:
        raise RuntimeError("Dataset is empty.")

    columns = set(rows[0].keys())

    for candidate in candidates:
        if candidate in columns:
            return candidate

    raise RuntimeError(
        "Could not locate required column. "
        f"Tried {candidates}. Available columns: {sorted(columns)}"
    )


def normalise_yd(value: str) -> int | None:
    value = str(value).strip().lower()

    if value in {"1", "1.0", "true"}:
        return 1

    if value in {"0", "0.0", "false"}:
        return 0

    if value in {"", "none", "null", "nan"}:
        return None

    raise ValueError(f"Unexpected Y_D value: {value!r}")


def print_counter(title: str, counter: Counter) -> None:
    print(f"\n{title}")

    for key, value in sorted(
        counter.items(),
        key=lambda item: str(item[0]),
    ):
        print(f"  {key}: {value}")


def main() -> None:
    cases = load_csv(A8_6_DATASET)
    assignments = load_csv(PARTITION_FILE)

    if len(cases) != 600:
        raise RuntimeError(
            f"Expected 600 A8.6 cases, found {len(cases)}"
        )

    if len(assignments) != 150:
        raise RuntimeError(
            f"Expected 150 scenario assignments, "
            f"found {len(assignments)}"
        )

    scenario_col = find_column(
        cases,
        ["scenario_id"],
    )

    yd_col = find_column(
        cases,
        [
            "diagnostic_correctness",
            "Y_D",
            "y_d",
        ],
    )

    q_col = find_column(
        cases,
        [
            "evidence_condition",
            "q_condition",
            "condition",
            "Q",
        ],
    )

    fault_col = find_column(
        cases,
        ["ground_truth"],
    )

    partition_by_scenario = {
        row["scenario_id"]: row["partition"]
        for row in assignments
    }

    if len(partition_by_scenario) != 150:
        raise RuntimeError(
            "Partition file does not contain 150 unique scenario IDs."
        )

    case_scenarios = {
        row[scenario_col]
        for row in cases
    }

    missing_assignments = (
        case_scenarios - set(partition_by_scenario)
    )

    if missing_assignments:
        raise RuntimeError(
            "A8.6 cases contain scenarios missing from the frozen "
            f"partition: {sorted(missing_assignments)}"
        )

    stats = {}

    for partition in ["DEVELOPMENT", "HOLDOUT"]:
        partition_cases = [
            row
            for row in cases
            if partition_by_scenario[row[scenario_col]]
            == partition
        ]

        eligible = [
            row
            for row in partition_cases
            if normalise_yd(row[yd_col]) is not None
        ]

        unknown = [
            row
            for row in partition_cases
            if normalise_yd(row[yd_col]) is None
        ]

        represented_scenarios = {
            row[scenario_col]
            for row in eligible
        }

        yd_counts = Counter(
            normalise_yd(row[yd_col])
            for row in eligible
        )

        q_all = Counter(
            row[q_col]
            for row in partition_cases
        )

        q_eligible = Counter(
            row[q_col]
            for row in eligible
        )

        fault_all = Counter(
            row[fault_col]
            for row in partition_cases
        )

        fault_eligible = Counter(
            row[fault_col]
            for row in eligible
        )

        eligible_per_scenario = Counter(
            Counter(
                row[scenario_col]
                for row in eligible
            ).values()
        )

        stats[partition] = {
            "all_cases": len(partition_cases),
            "eligible_cases": len(eligible),
            "unknown_cases": len(unknown),
            "represented_scenarios": len(
                represented_scenarios
            ),
            "yd_counts": yd_counts,
            "q_all": q_all,
            "q_eligible": q_eligible,
            "fault_all": fault_all,
            "fault_eligible": fault_eligible,
            "eligible_per_scenario": eligible_per_scenario,
        }

    print("=" * 72)
    print("A8.7.2 FROZEN PARTITION COMPATIBILITY CHECK")
    print("=" * 72)

    for partition in ["DEVELOPMENT", "HOLDOUT"]:
        s = stats[partition]

        print(f"\n{'-' * 72}")
        print(partition)
        print("-" * 72)

        print(
            f"All evidence cases:          "
            f"{s['all_cases']}"
        )
        print(
            f"Y_D-eligible cases:          "
            f"{s['eligible_cases']}"
        )
        print(
            f"UNKNOWN / Y_D-ineligible:    "
            f"{s['unknown_cases']}"
        )
        print(
            f"Y_D-eligible scenarios:      "
            f"{s['represented_scenarios']}"
        )
        print(
            f"Y_D = 0:                     "
            f"{s['yd_counts'][0]}"
        )
        print(
            f"Y_D = 1:                     "
            f"{s['yd_counts'][1]}"
        )

        if s["eligible_cases"]:
            rate = (
                s["yd_counts"][1]
                / s["eligible_cases"]
            )
            print(
                f"Observed correctness rate:   "
                f"{rate:.4f}"
            )

        print_counter(
            "Q distribution, all cases:",
            s["q_all"],
        )

        print_counter(
            "Q distribution, Y_D eligible:",
            s["q_eligible"],
        )

        print_counter(
            "Fault distribution, all cases:",
            s["fault_all"],
        )

        print_counter(
            "Fault distribution, Y_D eligible:",
            s["fault_eligible"],
        )

        print_counter(
            "Eligible cases per represented scenario:",
            s["eligible_per_scenario"],
        )

    total_all = sum(
        s["all_cases"]
        for s in stats.values()
    )

    total_eligible = sum(
        s["eligible_cases"]
        for s in stats.values()
    )

    total_unknown = sum(
        s["unknown_cases"]
        for s in stats.values()
    )

    total_y0 = sum(
        s["yd_counts"][0]
        for s in stats.values()
    )

    total_y1 = sum(
        s["yd_counts"][1]
        for s in stats.values()
    )

    print(f"\n{'=' * 72}")
    print("GLOBAL RECONCILIATION")
    print("=" * 72)

    print(f"All cases:       {total_all} / 600")
    print(f"Eligible cases:  {total_eligible} / 292")
    print(f"UNKNOWN cases:   {total_unknown} / 308")
    print(f"Y_D = 0:         {total_y0} / 236")
    print(f"Y_D = 1:         {total_y1} / 56")

    assert total_all == 600
    assert total_eligible == 292
    assert total_unknown == 308
    assert total_y0 == 236
    assert total_y1 == 56

    dev = stats["DEVELOPMENT"]
    hold = stats["HOLDOUT"]

    print(f"\n{'=' * 72}")
    print("MODELLING READINESS")
    print("=" * 72)

    dev_two_classes = (
        dev["yd_counts"][0] > 0
        and dev["yd_counts"][1] > 0
    )

    hold_two_classes = (
        hold["yd_counts"][0] > 0
        and hold["yd_counts"][1] > 0
    )

    print(
        "Development contains both Y_D classes: "
        f"{'PASS' if dev_two_classes else 'FAIL'}"
    )

    print(
        "Holdout contains both Y_D classes:     "
        f"{'PASS' if hold_two_classes else 'FAIL'}"
    )

    print(
        "\nNOTE: PASS here establishes only basic empirical "
        "compatibility. It does not establish that ordinary "
        "maximum-likelihood logistic regression is suitable. "
        "Separation, sparse predictors, grouped cross-validation "
        "and E representation remain to be checked."
    )


if __name__ == "__main__":
    main()