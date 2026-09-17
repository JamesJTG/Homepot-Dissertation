import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


# ============================================================
# Paths
# ============================================================

BASE = Path(
    "research/diagnostic_reliability/analysis/outputs/"
    "a8_7_reliability_modelling"
)

PREDICTIONS_FILE = (
    BASE
    / "a8_7_grouped_cv_out_of_fold_reliability_predictions.csv"
)

COEFFICIENTS_FILE = (
    BASE
    / "a8_7_grouped_cv_flic_model_coefficients.csv"
)

FOLD_FILE = (
    BASE
    / "a8_7_development_scenario_grouped_cv_assignment.csv"
)

MODELLING_MANIFEST_FILE = (
    BASE
    / "a8_7_grouped_cv_reliability_modelling_manifest.json"
)


MODEL_SPECS = {
    "M_E": [
        "E",
    ],
    "M_C": [
        "mean_self_confidence",
    ],
    "M_H": [
        "diagnostic_entropy",
    ],
    "M_U": [
        "mean_self_confidence",
        "diagnostic_entropy",
    ],
    "M_EU": [
        "E",
        "mean_self_confidence",
        "diagnostic_entropy",
    ],
}


# ============================================================
# Utilities
# ============================================================

def load_csv(path):
    with path.open(
        newline="",
        encoding="utf-8",
    ) as f:
        return list(csv.DictReader(f))


def finite(value):
    return math.isfinite(float(value))


predictions = load_csv(PREDICTIONS_FILE)
coefficients = load_csv(COEFFICIENTS_FILE)
folds = load_csv(FOLD_FILE)

with MODELLING_MANIFEST_FILE.open(
    encoding="utf-8",
) as f:
    manifest = json.load(f)


print("=" * 80)
print("A8.7.3.4 MODEL-FITTING INTEGRITY AND COEFFICIENT-STABILITY CHECK")
print("=" * 80)


# ============================================================
# 1. Global prediction integrity
# ============================================================

print("\n1. GLOBAL PREDICTION INTEGRITY")
print("-" * 80)

assert len(predictions) == 1190

prediction_keys = [
    (
        row["case_id"],
        row["model"],
    )
    for row in predictions
]

assert len(set(prediction_keys)) == 1190

model_counts = Counter(
    row["model"]
    for row in predictions
)

for model in MODEL_SPECS:
    assert model_counts[model] == 238

probabilities = np.array(
    [
        float(row["predicted_reliability"])
        for row in predictions
    ],
    dtype=float,
)

assert np.isfinite(probabilities).all()
assert np.all(probabilities > 0.0)
assert np.all(probabilities < 1.0)

print("1,190 prediction records: PASS")
print("Unique case/model prediction keys: PASS")
print("238 predictions per model: PASS")
print("All probabilities finite: PASS")
print("All probabilities strictly within (0,1): PASS")


# ============================================================
# 2. Outcome and case consistency across models
# ============================================================

print("\n2. CASE CONSISTENCY ACROSS FIVE MODELS")
print("-" * 80)

by_case = defaultdict(list)

for row in predictions:
    by_case[row["case_id"]].append(row)

assert len(by_case) == 238

for case_id, rows in by_case.items():

    assert len(rows) == 5

    assert {
        row["model"]
        for row in rows
    } == set(MODEL_SPECS)

    scenario_ids = {
        row["scenario_id"]
        for row in rows
    }

    folds_seen = {
        int(row["cv_fold"])
        for row in rows
    }

    outcomes = {
        int(row["Y_D"])
        for row in rows
    }

    e_values = {
        float(row["E"])
        for row in rows
    }

    confidence_values = {
        float(row["mean_self_confidence"])
        for row in rows
    }

    entropy_values = {
        float(row["diagnostic_entropy"])
        for row in rows
    }

    assert len(scenario_ids) == 1
    assert len(folds_seen) == 1
    assert len(outcomes) == 1
    assert len(e_values) == 1
    assert len(confidence_values) == 1
    assert len(entropy_values) == 1


print("238 unique analytical cases: PASS")
print("Exactly five predictions per case: PASS")
print("Case metadata identical across model predictions: PASS")


# ============================================================
# 3. Frozen fold traceability
# ============================================================

print("\n3. FROZEN FOLD TRACEABILITY")
print("-" * 80)

fold_map = {
    row["scenario_id"]: int(row["cv_fold"])
    for row in folds
}

assert len(fold_map) == 120

for row in predictions:

    sid = row["scenario_id"]
    observed_fold = int(row["cv_fold"])

    assert sid in fold_map
    assert fold_map[sid] == observed_fold


print("120 frozen DEVELOPMENT scenario assignments loaded: PASS")
print("Every OOF prediction matches frozen scenario fold: PASS")


# ============================================================
# 4. Coefficient structure
# ============================================================

print("\n4. COEFFICIENT STRUCTURE")
print("-" * 80)

coeff_lookup = defaultdict(dict)

for row in coefficients:

    key = (
        int(row["cv_fold"]),
        row["model"],
    )

    parameter = row["parameter"]
    value = float(row["value"])

    assert finite(value)

    if parameter in coeff_lookup[key]:
        raise RuntimeError(
            f"Duplicate parameter {parameter} "
            f"for {key}."
        )

    coeff_lookup[key][parameter] = value


assert len(coeff_lookup) == 25


for fold in range(1, 6):

    for model, predictors in MODEL_SPECS.items():

        key = (
            fold,
            model,
        )

        assert key in coeff_lookup

        expected_parameters = {
            "intercept_firth",
            "intercept_flic",
            *predictors,
        }

        actual_parameters = set(
            coeff_lookup[key]
        )

        assert (
            actual_parameters
            == expected_parameters
        ), (
            f"{key}: expected "
            f"{expected_parameters}, "
            f"found {actual_parameters}"
        )


print("25 fold/model fits represented: PASS")
print("All coefficients finite: PASS")
print("Expected parameter structure for every fit: PASS")


# ============================================================
# 5. Firth versus FLIC intercept behaviour
# ============================================================

print("\n5. FIRTH / FLIC INTERCEPT BEHAVIOUR")
print("-" * 80)

intercept_differences = []

for key, params in coeff_lookup.items():

    firth_intercept = params[
        "intercept_firth"
    ]

    flic_intercept = params[
        "intercept_flic"
    ]

    difference = (
        flic_intercept
        - firth_intercept
    )

    intercept_differences.append(
        difference
    )


intercept_differences = np.array(
    intercept_differences,
    dtype=float,
)

assert np.isfinite(
    intercept_differences
).all()


print(
    "FLIC - Firth intercept difference range:",
    f"{intercept_differences.min():.6f}",
    "to",
    f"{intercept_differences.max():.6f}",
)

print(
    "Mean absolute intercept correction:",
    f"{np.abs(intercept_differences).mean():.6f}",
)

print("All intercept corrections finite: PASS")


# ============================================================
# 6. Predictor coefficient stability
# ============================================================

print("\n6. PREDICTOR COEFFICIENT STABILITY ACROSS FOLDS")
print("-" * 80)

stability_records = []

for model, predictors in MODEL_SPECS.items():

    print(f"\n{model}")

    for predictor in predictors:

        values = np.array(
            [
                coeff_lookup[
                    (fold, model)
                ][predictor]
                for fold in range(1, 6)
            ],
            dtype=float,
        )

        signs = np.sign(values)

        nonzero_signs = {
            int(sign)
            for sign in signs
            if sign != 0
        }

        sign_consistent = (
            len(nonzero_signs) <= 1
        )

        mean_value = float(
            values.mean()
        )

        sd_value = float(
            values.std(ddof=1)
        )

        min_value = float(
            values.min()
        )

        max_value = float(
            values.max()
        )

        print(
            f"  {predictor}: "
            f"mean={mean_value:.6f}, "
            f"sd={sd_value:.6f}, "
            f"range={min_value:.6f} "
            f"to {max_value:.6f}, "
            f"sign-consistent="
            f"{'YES' if sign_consistent else 'NO'}"
        )

        stability_records.append(
            {
                "model": model,
                "predictor": predictor,
                "values": values,
                "sign_consistent":
                    sign_consistent,
            }
        )


# ============================================================
# 7. Model-specific OOF probability ranges
# ============================================================

print("\n7. OUT-OF-FOLD PROBABILITY STRUCTURE")
print("-" * 80)

for model in MODEL_SPECS:

    values = np.array(
        [
            float(
                row[
                    "predicted_reliability"
                ]
            )
            for row in predictions
            if row["model"] == model
        ],
        dtype=float,
    )

    print(
        f"{model}: "
        f"min={values.min():.6f}, "
        f"mean={values.mean():.6f}, "
        f"median={np.median(values):.6f}, "
        f"max={values.max():.6f}, "
        f"sd={values.std(ddof=1):.6f}"
    )


# ============================================================
# 8. Fold coverage by model
# ============================================================

print("\n8. FOLD COVERAGE BY MODEL")
print("-" * 80)

for model in MODEL_SPECS:

    counts = Counter(
        int(row["cv_fold"])
        for row in predictions
        if row["model"] == model
    )

    expected = {
        1: 46,
        2: 49,
        3: 49,
        4: 49,
        5: 45,
    }

    assert dict(counts) == expected

    print(
        f"{model}: "
        f"{dict(sorted(counts.items()))} PASS"
    )


# ============================================================
# 9. Manifest policy checks
# ============================================================

print("\n9. MODELLING MANIFEST")
print("-" * 80)

assert (
    manifest[
        "development_scenarios"
    ]
    == 120
)

assert (
    manifest[
        "development_eligible_cases"
    ]
    == 238
)

assert (
    manifest[
        "n_folds"
    ]
    == 5
)

assert (
    manifest[
        "estimator"
    ]
    ==
    "Firth logistic regression with "
    "FLIC intercept correction"
)

print("Development scenario count: PASS")
print("Eligible case count: PASS")
print("Five-fold policy: PASS")
print("Firth + FLIC estimator recorded: PASS")


# ============================================================
# Final status
# ============================================================

print("\n" + "=" * 80)
print("FINAL STATUS")
print("=" * 80)

print(
    "\nA8.7.3.4 MODEL-FITTING "
    "INTEGRITY CHECK: PASS"
)

print(
    "\nNOTE: Coefficient sign consistency is "
    "reported diagnostically and is not used "
    "to alter the frozen models or folds."
)