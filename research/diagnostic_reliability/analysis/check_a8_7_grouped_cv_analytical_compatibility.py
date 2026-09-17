import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


BASE = Path("research/diagnostic_reliability/analysis/outputs")

DATASET_FILE = (
    BASE
    / "a8_6_outcome_uncertainty_construction"
    / "a8_6_evidence_case_reliability_dataset.csv"
)

PARTITION_FILE = (
    BASE
    / "a8_7_reliability_modelling"
    / "a8_7_scenario_development_holdout_assignment.csv"
)

FOLD_FILE = (
    BASE
    / "a8_7_reliability_modelling"
    / "a8_7_development_scenario_grouped_cv_assignment.csv"
)


def load_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def first_existing(row, candidates):
    for name in candidates:
        if name in row:
            return name
    raise KeyError(
        f"Could not find any of {candidates}. "
        f"Available columns: {list(row.keys())}"
    )


dataset = load_csv(DATASET_FILE)
partition = load_csv(PARTITION_FILE)
folds = load_csv(FOLD_FILE)

print("=" * 76)
print("A8.7.3.2 GROUPED CV ANALYTICAL COMPATIBILITY CHECK")
print("=" * 76)


# ------------------------------------------------------------
# Resolve schema
# ------------------------------------------------------------

scenario_col = first_existing(
    dataset[0],
    ["scenario_id", "scenario"],
)

yd_col = first_existing(
    dataset[0],
    [
        "diagnostic_correctness",
        "Y_D",
        "y_d",
    ],
)

actionable_col = first_existing(
    dataset[0],
    ["actionable"],
)

confidence_col = first_existing(
    dataset[0],
      [
        "mean_self_confidence",
        "mean_self_reported_confidence",
        "mean_confidence",
        "self_confidence_mean",
    ],
)

entropy_col = first_existing(
    dataset[0],
    [
        "diagnostic_entropy",
        "diagnosis_entropy",
        "entropy",
    ],
)


# ------------------------------------------------------------
# Frozen development IDs
# ------------------------------------------------------------

partition_scenario_col = first_existing(
    partition[0],
    ["scenario_id", "scenario"],
)

partition_col = first_existing(
    partition[0],
    [
        "partition",
        "split",
        "assignment",
        "dataset_partition",
    ],
)

development_ids = {
    row[partition_scenario_col]
    for row in partition
    if row[partition_col].strip().upper() == "DEVELOPMENT"
}

assert len(development_ids) == 120


# ------------------------------------------------------------
# Frozen fold map
# ------------------------------------------------------------

fold_scenario_col = first_existing(
    folds[0],
    ["scenario_id", "scenario"],
)

fold_col = first_existing(
    folds[0],
    ["cv_fold", "fold"],
)

fold_map = {
    row[fold_scenario_col]: int(row[fold_col])
    for row in folds
}

assert len(fold_map) == 120
assert set(fold_map) == development_ids


# ------------------------------------------------------------
# Select DEVELOPMENT cases only
# ------------------------------------------------------------

development_cases = [
    row
    for row in dataset
    if row[scenario_col] in development_ids
]

if len(development_cases) != 480:
    raise RuntimeError(
        f"Expected 480 DEVELOPMENT evidence cases, "
        f"found {len(development_cases)}."
    )


# ------------------------------------------------------------
# Parse Y_D
# ------------------------------------------------------------

def parse_binary_or_none(value):
    text = str(value).strip().lower()

    if text in {"", "none", "null", "nan"}:
        return None

    if text in {"1", "1.0", "true"}:
        return 1

    if text in {"0", "0.0", "false"}:
        return 0

    raise ValueError(f"Unexpected binary value: {value!r}")


eligible = []

for row in development_cases:
    yd = parse_binary_or_none(row[yd_col])

    if yd is None:
        continue

    item = dict(row)
    item["_Y_D"] = yd
    item["_fold"] = fold_map[row[scenario_col]]

    eligible.append(item)


print("\nDevelopment evidence cases:", len(development_cases))
print("Primary Y_D eligible:", len(eligible))
print("Excluded because Y_D undefined:", len(development_cases) - len(eligible))

overall_y = Counter(row["_Y_D"] for row in eligible)

print("\nOverall eligible Y_D:")
print("Y_D=0:", overall_y[0])
print("Y_D=1:", overall_y[1])


if len(eligible) != 238:
    raise RuntimeError(
        f"Expected 238 eligible DEVELOPMENT cases, found {len(eligible)}."
    )

if overall_y[0] != 193 or overall_y[1] != 45:
    raise RuntimeError(
        f"Unexpected DEVELOPMENT outcome counts: {dict(overall_y)}"
    )


# ------------------------------------------------------------
# Helper parsers
# ------------------------------------------------------------

def parse_bool(value):
    text = str(value).strip().lower()

    if text in {"true", "1", "1.0"}:
        return 1

    if text in {"false", "0", "0.0"}:
        return 0

    raise ValueError(f"Unexpected boolean value: {value!r}")


def numeric_values(rows, column):
    return np.array(
        [float(row[column]) for row in rows],
        dtype=float,
    )


# ------------------------------------------------------------
# Fold-by-fold diagnostics
# ------------------------------------------------------------

all_pass = True

print("\n" + "=" * 76)
print("VALIDATION-FOLD DIAGNOSTICS")
print("=" * 76)

for fold in range(1, 6):

    validation = [
        row for row in eligible
        if row["_fold"] == fold
    ]

    training = [
        row for row in eligible
        if row["_fold"] != fold
    ]

    val_y = Counter(row["_Y_D"] for row in validation)
    train_y = Counter(row["_Y_D"] for row in training)

    val_scenarios = {
        row[scenario_col] for row in validation
    }

    train_scenarios = {
        row[scenario_col] for row in training
    }

    val_e = Counter(
        parse_bool(row[actionable_col])
        for row in validation
    )

    train_e = Counter(
        parse_bool(row[actionable_col])
        for row in training
    )

    val_c = numeric_values(
        validation,
        confidence_col,
    )

    train_c = numeric_values(
        training,
        confidence_col,
    )

    val_h = numeric_values(
        validation,
        entropy_col,
    )

    train_h = numeric_values(
        training,
        entropy_col,
    )

    print(f"\nFold {fold}")
    print("-" * 36)

    print(
        f"Validation: "
        f"{len(validation)} eligible cases, "
        f"{len(val_scenarios)} represented scenarios"
    )

    print(
        f"  Y_D=0: {val_y[0]}"
    )

    print(
        f"  Y_D=1: {val_y[1]}"
    )

    print(
        f"  E=0: {val_e[0]}"
    )

    print(
        f"  E=1: {val_e[1]}"
    )

    print(
        "  Confidence range:",
        f"{val_c.min():.6f} to {val_c.max():.6f}",
        f"({len(np.unique(val_c))} unique)",
    )

    print(
        "  Entropy range:",
        f"{val_h.min():.6f} to {val_h.max():.6f}",
        f"({len(np.unique(val_h))} unique)",
    )

    print(
        f"Training: "
        f"{len(training)} eligible cases, "
        f"{len(train_scenarios)} represented scenarios"
    )

    print(
        f"  Y_D=0: {train_y[0]}"
    )

    print(
        f"  Y_D=1: {train_y[1]}"
    )

    print(
        f"  E=0: {train_e[0]}"
    )

    print(
        f"  E=1: {train_e[1]}"
    )

    print(
        "  Confidence range:",
        f"{train_c.min():.6f} to {train_c.max():.6f}",
        f"({len(np.unique(train_c))} unique)",
    )

    print(
        "  Entropy range:",
        f"{train_h.min():.6f} to {train_h.max():.6f}",
        f"({len(np.unique(train_h))} unique)",
    )

    checks = {
        "validation_has_both_Y_classes":
            val_y[0] > 0 and val_y[1] > 0,

        "training_has_both_Y_classes":
            train_y[0] > 0 and train_y[1] > 0,

        "validation_has_both_E_states":
            val_e[0] > 0 and val_e[1] > 0,

        "training_has_both_E_states":
            train_e[0] > 0 and train_e[1] > 0,

        "training_confidence_varies":
            len(np.unique(train_c)) > 1,

        "training_entropy_varies":
            len(np.unique(train_h)) > 1,

        "scenario_sets_disjoint":
            val_scenarios.isdisjoint(train_scenarios),
    }

    print("  Checks:")

    for name, passed in checks.items():
        print(
            f"    {name}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

        if not passed:
            all_pass = False


# ------------------------------------------------------------
# Global reconciliation
# ------------------------------------------------------------

fold_case_counts = Counter(
    row["_fold"] for row in eligible
)

print("\n" + "=" * 76)
print("GLOBAL RECONCILIATION")
print("=" * 76)

for fold in range(1, 6):
    print(
        f"Fold {fold} eligible cases:",
        fold_case_counts[fold],
    )

print(
    "\nSum across folds:",
    sum(fold_case_counts.values()),
)

assert sum(fold_case_counts.values()) == 238


print("\nFinal status:")

if all_pass:
    print("A8.7.3.2 GROUPED CV ANALYTICAL COMPATIBILITY: PASS")
else:
    print("A8.7.3.2 GROUPED CV ANALYTICAL COMPATIBILITY: REVIEW REQUIRED")