"""A8.7.2.4 predictor structure and logistic-separation diagnostic.

Uses only the frozen DEVELOPMENT partition.

Purpose:
1. Confirm the primary modelling sample.
2. Encode upstream evidence readiness E from actionability.
3. Characterise E, mean self-confidence, and diagnostic entropy.
4. Cross-tabulate E against Y_D.
5. Examine downstream predictors by outcome and E.
6. Diagnose whether E perfectly separates Y_D.
7. Check whether downstream predictors themselves show obvious
   deterministic separation.

No reliability model is fitted by this script.
"""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
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


def parse_yd(value: str) -> int | None:
    value = str(value).strip().lower()

    if value in {"1", "1.0", "true"}:
        return 1

    if value in {"0", "0.0", "false"}:
        return 0

    if value in {"", "none", "null", "nan"}:
        return None

    raise ValueError(f"Unexpected Y_D value: {value!r}")


def parse_bool(value: str) -> bool:
    value = str(value).strip().lower()

    if value in {"true", "1", "1.0", "yes"}:
        return True

    if value in {"false", "0", "0.0", "no"}:
        return False

    raise ValueError(f"Unexpected boolean value: {value!r}")


def summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "n": 0,
            "min": math.nan,
            "mean": math.nan,
            "max": math.nan,
            "unique": 0,
        }

    return {
        "n": len(values),
        "min": min(values),
        "mean": sum(values) / len(values),
        "max": max(values),
        "unique": len(set(values)),
    }


def print_summary(label: str, values: list[float]) -> None:
    s = summary(values)

    print(
        f"{label:<30}"
        f"n={s['n']:<4} "
        f"min={s['min']:.6f} "
        f"mean={s['mean']:.6f} "
        f"max={s['max']:.6f} "
        f"unique={s['unique']}"
    )


def ranges_overlap(
    values_0: list[float],
    values_1: list[float],
) -> bool:
    if not values_0 or not values_1:
        return False

    return not (
        max(values_0) < min(values_1)
        or max(values_1) < min(values_0)
    )


def main() -> None:
    cases = load_csv(A8_6_DATASET)
    assignments = load_csv(PARTITION_FILE)

    partition_by_scenario = {
        row["scenario_id"]: row["partition"]
        for row in assignments
    }

    development_cases = [
        row
        for row in cases
        if partition_by_scenario[row["scenario_id"]]
        == "DEVELOPMENT"
    ]

    eligible = [
        row
        for row in development_cases
        if parse_yd(row["diagnostic_correctness"])
        is not None
    ]

    if len(development_cases) != 480:
        raise RuntimeError(
            "Expected 480 DEVELOPMENT evidence cases, "
            f"found {len(development_cases)}."
        )

    if len(eligible) != 238:
        raise RuntimeError(
            "Expected 238 Y_D-eligible DEVELOPMENT cases, "
            f"found {len(eligible)}."
        )

    rows = []

    for row in eligible:
        yd = parse_yd(
            row["diagnostic_correctness"]
        )

        actionable = parse_bool(
            row["actionable"]
        )

        rows.append(
            {
                "scenario_id": row["scenario_id"],
                "case_id": row["case_id"],
                "Y_D": yd,
                "E": 1 if actionable else 0,
                "confidence": float(
                    row["mean_self_confidence"]
                ),
                "entropy": float(
                    row["diagnostic_entropy"]
                ),
                "trust_score": float(
                    row["trust_score"]
                ),
                "trust_mode": row["trust_mode"],
                "q": row["evidence_condition"],
            }
        )

    print("=" * 76)
    print(
        "A8.7.2.4 PREDICTOR STRUCTURE AND "
        "LOGISTIC-SEPARATION CHECK"
    )
    print("=" * 76)

    print("\nPRIMARY DEVELOPMENT SAMPLE")
    print("-" * 76)

    print(
        f"Development evidence cases:   "
        f"{len(development_cases)}"
    )
    print(
        f"Y_D-eligible cases:           "
        f"{len(rows)}"
    )

    yd_counts = Counter(
        row["Y_D"]
        for row in rows
    )

    print(
        f"Y_D = 0:                      "
        f"{yd_counts[0]}"
    )
    print(
        f"Y_D = 1:                      "
        f"{yd_counts[1]}"
    )

    print(
        f"Eligible scenarios:           "
        f"{len(set(row['scenario_id'] for row in rows))}"
    )

    print("\nE REPRESENTATION CHECK")
    print("-" * 76)

    e_counts = Counter(
        row["E"]
        for row in rows
    )

    print(f"E = 0: {e_counts[0]}")
    print(f"E = 1: {e_counts[1]}")

    e_mapping = defaultdict(
        lambda: {
            "trust_scores": set(),
            "trust_modes": set(),
            "q": set(),
        }
    )

    for row in rows:
        e_mapping[row["E"]]["trust_scores"].add(
            row["trust_score"]
        )
        e_mapping[row["E"]]["trust_modes"].add(
            row["trust_mode"]
        )
        e_mapping[row["E"]]["q"].add(
            row["q"]
        )

    for e in sorted(e_mapping):
        print(f"\nE = {e}")
        print(
            "  trust_score: "
            f"{sorted(e_mapping[e]['trust_scores'])}"
        )
        print(
            "  trust_mode:  "
            f"{sorted(e_mapping[e]['trust_modes'])}"
        )
        print(
            "  Q:           "
            f"{sorted(e_mapping[e]['q'])}"
        )

    print("\nE x Y_D CROSS-TABULATION")
    print("-" * 76)

    table = Counter(
        (row["E"], row["Y_D"])
        for row in rows
    )

    print("              Y_D=0    Y_D=1    Total")

    for e in [0, 1]:
        y0 = table[(e, 0)]
        y1 = table[(e, 1)]

        print(
            f"E={e:<8} "
            f"{y0:>7} "
            f"{y1:>9} "
            f"{y0 + y1:>8}"
        )

    print("\nCORRECTNESS RATE BY E")
    print("-" * 76)

    for e in [0, 1]:
        y0 = table[(e, 0)]
        y1 = table[(e, 1)]
        total = y0 + y1

        rate = y1 / total if total else math.nan

        print(
            f"E={e}: n={total}, "
            f"correct={y1}, "
            f"rate={rate:.6f}"
        )

    print("\nDOWNSTREAM PREDICTOR STRUCTURE")
    print("-" * 76)

    confidence_all = [
        row["confidence"]
        for row in rows
    ]

    entropy_all = [
        row["entropy"]
        for row in rows
    ]

    print_summary(
        "Mean self-confidence:",
        confidence_all,
    )

    print_summary(
        "Diagnostic entropy:",
        entropy_all,
    )

    print("\nDOWNSTREAM SIGNALS BY Y_D")
    print("-" * 76)

    for yd in [0, 1]:
        subset = [
            row
            for row in rows
            if row["Y_D"] == yd
        ]

        print(f"\nY_D = {yd}")

        print_summary(
            "  Mean self-confidence:",
            [
                row["confidence"]
                for row in subset
            ],
        )

        print_summary(
            "  Diagnostic entropy:",
            [
                row["entropy"]
                for row in subset
            ],
        )

    print("\nDOWNSTREAM SIGNALS BY E")
    print("-" * 76)

    for e in [0, 1]:
        subset = [
            row
            for row in rows
            if row["E"] == e
        ]

        print(f"\nE = {e}")

        print_summary(
            "  Mean self-confidence:",
            [
                row["confidence"]
                for row in subset
            ],
        )

        print_summary(
            "  Diagnostic entropy:",
            [
                row["entropy"]
                for row in subset
            ],
        )

    print("\nSIMPLE RANGE-SEPARATION DIAGNOSTICS")
    print("-" * 76)

    incorrect = [
        row
        for row in rows
        if row["Y_D"] == 0
    ]

    correct = [
        row
        for row in rows
        if row["Y_D"] == 1
    ]

    c0 = [
        row["confidence"]
        for row in incorrect
    ]

    c1 = [
        row["confidence"]
        for row in correct
    ]

    h0 = [
        row["entropy"]
        for row in incorrect
    ]

    h1 = [
        row["entropy"]
        for row in correct
    ]

    print(
        "Confidence ranges overlap: "
        f"{'YES' if ranges_overlap(c0, c1) else 'NO'}"
    )

    print(
        "Entropy ranges overlap:    "
        f"{'YES' if ranges_overlap(h0, h1) else 'NO'}"
    )

    print("\nSEPARATION ASSESSMENT")
    print("-" * 76)

    zero_cells = [
        (e, yd)
        for e in [0, 1]
        for yd in [0, 1]
        if table[(e, yd)] == 0
    ]

    if zero_cells:
        print(
            "WARNING: E x Y_D contains one or more "
            "zero cells:"
        )

        for e, yd in zero_cells:
            print(
                f"  E={e}, Y_D={yd}: 0 cases"
            )

        if (
            table[(0, 1)] == 0
            and table[(0, 0)] > 0
        ):
            print(
                "\nRESULT: E exhibits complete one-sided "
                "outcome separation in the development "
                "sample: no correct substantive diagnoses "
                "occur when E=0."
            )

            print(
                "Ordinary maximum-likelihood logistic "
                "regression containing E may therefore "
                "produce an unbounded or highly unstable "
                "coefficient."
            )
    else:
        print(
            "No zero cells observed in E x Y_D."
        )

    print(
        "\nThis diagnostic does not choose a remedy. "
        "If separation is confirmed, the estimator "
        "strategy must be resolved explicitly before "
        "fitting M_E or M_EU."
    )


if __name__ == "__main__":
    main()