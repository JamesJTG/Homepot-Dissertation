import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


SEED = 20260917
N_FOLDS = 5

BASE = Path("research/diagnostic_reliability/analysis/outputs")
A8_7_DIR = BASE / "a8_7_reliability_modelling"

PARTITION_FILE = (
    A8_7_DIR /
    "a8_7_scenario_development_holdout_assignment.csv"
)

OUTPUT_FILE = (
    A8_7_DIR /
    "a8_7_development_scenario_grouped_cv_assignment.csv"
)

MANIFEST_FILE = (
    A8_7_DIR /
    "a8_7_development_scenario_grouped_cv_manifest.json"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def first_existing(row, candidates):
    for name in candidates:
        if name in row and row[name] not in ("", None):
            return row[name]
    raise KeyError(
        f"Could not find any of {candidates}. "
        f"Available columns: {list(row.keys())}"
    )


# ------------------------------------------------------------
# Load frozen 120/30 scenario partition
# ------------------------------------------------------------

with PARTITION_FILE.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

if len(rows) != 150:
    raise RuntimeError(
        f"Expected 150 scenario assignments, found {len(rows)}."
    )


def get_partition(row):
    return first_existing(
        row,
        [
            "partition",
            "split",
            "assignment",
            "dataset_partition",
        ],
    ).strip().upper()


development = [
    row for row in rows
    if get_partition(row) == "DEVELOPMENT"
]

if len(development) != 120:
    raise RuntimeError(
        f"Expected 120 DEVELOPMENT scenarios, found {len(development)}."
    )


# ------------------------------------------------------------
# Identify experimental design fields
# ------------------------------------------------------------

def scenario_id(row):
    return first_existing(
        row,
        ["scenario_id", "scenario", "id"],
    )


def fault_class(row):
    return first_existing(
        row,
        [
            "ground_truth",
            "ground_truth_fault",
            "fault_class",
            "fault",
        ],
    )


def scenario_family(row):
    return first_existing(
        row,
        [
            "scenario_family",
            "family",
            "family_id",
            "fault_family",
        ],
    )


# ------------------------------------------------------------
# Group DEVELOPMENT scenarios by fault × family
#
# IMPORTANT:
# No Y_D, UNKNOWN, confidence, entropy, agreement or other
# outcome/model-derived variable is used here.
# ------------------------------------------------------------

strata = defaultdict(list)

for row in development:
    key = (
        fault_class(row),
        scenario_family(row),
    )
    strata[key].append(row)

print("=" * 72)
print("A8.7 DEVELOPMENT SCENARIO-GROUPED 5-FOLD CV ASSIGNMENT")
print("=" * 72)

print("\nDevelopment scenarios:", len(development))
print("Fault × family strata:", len(strata))

stratum_sizes = Counter(
    len(items) for items in strata.values()
)

print("Stratum-size distribution:", dict(sorted(stratum_sizes.items())))

if len(strata) != 15:
    raise RuntimeError(
        f"Expected 15 fault × family strata, found {len(strata)}."
    )

if any(len(items) != 8 for items in strata.values()):
    bad = {
        str(key): len(items)
        for key, items in strata.items()
        if len(items) != 8
    }
    raise RuntimeError(
        f"Expected exactly 8 DEVELOPMENT scenarios per stratum. "
        f"Unexpected strata: {bad}"
    )


# ------------------------------------------------------------
# Reproducible outcome-blind fold assignment
#
# Each stratum contains 8 development scenarios.
# We shuffle scenarios within each stratum using the fixed seed.
#
# We also rotate the starting fold across strata so the extra
# three scenarios in each 8-scenario stratum do not repeatedly
# accumulate in the same folds.
# ------------------------------------------------------------

rng = random.Random(SEED)

assignments = []

sorted_strata = sorted(
    strata.items(),
    key=lambda item: (str(item[0][0]), str(item[0][1])),
)

for stratum_index, ((fault, family), items) in enumerate(sorted_strata):

    items = sorted(
        items,
        key=lambda row: scenario_id(row),
    )

    rng.shuffle(items)

    start_fold = stratum_index % N_FOLDS

    for position, row in enumerate(items):
        fold = ((start_fold + position) % N_FOLDS) + 1

        assignments.append(
            {
                "scenario_id": scenario_id(row),
                "ground_truth": fault,
                "scenario_family": family,
                "cv_fold": fold,
                "partition": "DEVELOPMENT",
            }
        )


# ------------------------------------------------------------
# Validate assignment
# ------------------------------------------------------------

if len(assignments) != 120:
    raise RuntimeError(
        f"Expected 120 fold assignments, found {len(assignments)}."
    )

ids = [row["scenario_id"] for row in assignments]

if len(set(ids)) != 120:
    raise RuntimeError("Duplicate scenario IDs detected.")

development_ids = {
    scenario_id(row) for row in development
}

if set(ids) != development_ids:
    raise RuntimeError(
        "CV assignment scenario IDs do not exactly match "
        "the frozen DEVELOPMENT partition."
    )

fold_counts = Counter(
    row["cv_fold"] for row in assignments
)

print("\nFold sizes:")
for fold in range(1, N_FOLDS + 1):
    print(f"Fold {fold}: {fold_counts[fold]} scenarios")

if max(fold_counts.values()) - min(fold_counts.values()) > 1:
    raise RuntimeError(
        f"Fold sizes are unexpectedly imbalanced: {dict(fold_counts)}"
    )


# ------------------------------------------------------------
# Design balance diagnostics
# ------------------------------------------------------------

print("\nFault-class distribution by fold:")

fault_by_fold = defaultdict(Counter)

for row in assignments:
    fault_by_fold[row["cv_fold"]][row["ground_truth"]] += 1

for fold in range(1, N_FOLDS + 1):
    print(f"\nFold {fold}")
    for fault, count in sorted(fault_by_fold[fold].items()):
        print(f"  {fault}: {count}")


print("\nScenario-family distribution by fold:")

family_by_fold = defaultdict(Counter)

for row in assignments:
    family_by_fold[row["cv_fold"]][row["scenario_family"]] += 1

for fold in range(1, N_FOLDS + 1):
    print(f"\nFold {fold}")
    for family, count in sorted(family_by_fold[fold].items()):
        print(f"  {family}: {count}")


# ------------------------------------------------------------
# Write frozen assignment
# ------------------------------------------------------------

assignments = sorted(
    assignments,
    key=lambda row: row["scenario_id"],
)

with OUTPUT_FILE.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "scenario_id",
            "ground_truth",
            "scenario_family",
            "cv_fold",
            "partition",
        ],
    )

    writer.writeheader()
    writer.writerows(assignments)


manifest = {
    "stage": "A8.7",
    "substage": "A8.7.3.1",
    "name": "Development Scenario-Grouped Cross-Validation Assignment",
    "status": "FROZEN_AFTER_VALIDATION",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "seed": SEED,
    "n_folds": N_FOLDS,
    "source_partition": str(PARTITION_FILE),
    "source_partition_sha256": sha256(PARTITION_FILE),
    "development_scenarios": 120,
    "assignment_policy": {
        "experimental_unit": "scenario_id",
        "grouping_rule": (
            "All evidence-condition variants belonging to the same "
            "scenario remain in the same cross-validation fold."
        ),
        "stratification_basis": [
            "ground_truth",
            "scenario_family",
        ],
        "outcome_blind": True,
        "prohibited_assignment_variables": [
            "Y_D",
            "Y_A",
            "UNKNOWN status",
            "mean self-reported confidence",
            "diagnostic entropy",
            "modal agreement",
            "model predictions",
            "model performance",
        ],
        "notes": (
            "Fold assignment instantiates the pre-specified "
            "5-fold scenario-grouped development protocol. "
            "The assignment uses experimental-design variables only."
        ),
    },
    "fold_sizes": {
        str(fold): fold_counts[fold]
        for fold in range(1, N_FOLDS + 1)
    },
    "output_assignment": str(OUTPUT_FILE),
}

with MANIFEST_FILE.open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(manifest, f, indent=2)


print("\nOutput artefacts:")
print(OUTPUT_FILE)
print(MANIFEST_FILE)

print("\nAssignment SHA-256:")
print(sha256(OUTPUT_FILE))

print("\nValidation:")
print("120 DEVELOPMENT scenarios only: PASS")
print("Unique scenario assignment: PASS")
print("Frozen DEVELOPMENT IDs preserved: PASS")
print("Outcome-blind assignment: PASS")
print("Scenario grouping rule: PASS")
print("5-fold size balance: PASS")

print("\nA8.7.3.1 GROUPED CV ASSIGNMENT: PASS")