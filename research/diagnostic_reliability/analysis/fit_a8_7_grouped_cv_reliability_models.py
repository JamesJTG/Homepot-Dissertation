import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
from firthmodels import FirthLogisticRegression


# ============================================================
# Configuration
# ============================================================

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

OUTPUT_DIR = (
    BASE
    / "a8_7_reliability_modelling"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "a8_7_grouped_cv_out_of_fold_reliability_predictions.csv"
)

COEFFICIENTS_FILE = (
    OUTPUT_DIR
    / "a8_7_grouped_cv_flic_model_coefficients.csv"
)

MANIFEST_FILE = (
    OUTPUT_DIR
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

def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def load_csv(path: Path):
    with path.open(
        newline="",
        encoding="utf-8",
    ) as f:
        return list(csv.DictReader(f))


def first_existing(row, candidates):
    for name in candidates:
        if name in row:
            return name

    raise KeyError(
        f"Could not find any of {candidates}. "
        f"Available columns: {list(row.keys())}"
    )


def parse_binary_or_none(value):
    text = str(value).strip().lower()

    if text in {
        "",
        "none",
        "null",
        "nan",
    }:
        return None

    if text in {
        "1",
        "1.0",
        "true",
    }:
        return 1

    if text in {
        "0",
        "0.0",
        "false",
    }:
        return 0

    raise ValueError(
        f"Unexpected binary value: {value!r}"
    )


def parse_bool(value):
    text = str(value).strip().lower()

    if text in {
        "true",
        "1",
        "1.0",
    }:
        return 1.0

    if text in {
        "false",
        "0",
        "0.0",
    }:
        return 0.0

    raise ValueError(
        f"Unexpected boolean value: {value!r}"
    )


# ============================================================
# FLIC implementation
# ============================================================

def fit_firth_with_flic(X, y):
    """
    Fit Firth logistic regression, then perform FLIC intercept
    correction.

    The Firth slope coefficients are retained.

    The intercept is corrected so that the mean fitted
    probability in the training sample equals the observed
    training event proportion.
    """

    model = FirthLogisticRegression()
    model.fit(X, y)

    firth_intercept = float(
        np.asarray(model.intercept_).ravel()[0]
    )

    slopes = np.asarray(
        model.coef_,
        dtype=float,
    ).ravel()

    linear_without_intercept = X @ slopes

    observed_rate = float(np.mean(y))

    def calibration_equation(intercept):
        probabilities = expit(
            intercept + linear_without_intercept
        )

        return float(
            probabilities.mean() - observed_rate
        )

    corrected_intercept = brentq(
        calibration_equation,
        -100.0,
        100.0,
    )

    training_probabilities = expit(
        corrected_intercept
        + linear_without_intercept
    )

    if not np.isfinite(
        corrected_intercept
    ):
        raise RuntimeError(
            "Non-finite FLIC intercept."
        )

    if not np.isfinite(
        slopes
    ).all():
        raise RuntimeError(
            "Non-finite Firth slope."
        )

    if not np.isfinite(
        training_probabilities
    ).all():
        raise RuntimeError(
            "Non-finite training probabilities."
        )

    if not np.isclose(
        training_probabilities.mean(),
        observed_rate,
        atol=1e-10,
    ):
        raise RuntimeError(
            "FLIC mean training probability does not "
            "equal the observed training event rate."
        )

    return {
        "firth_intercept": firth_intercept,
        "flic_intercept": corrected_intercept,
        "slopes": slopes,
        "training_event_rate": observed_rate,
        "training_mean_probability": float(
            training_probabilities.mean()
        ),
    }


def predict_flic(fit, X):
    probabilities = expit(
        fit["flic_intercept"]
        + X @ fit["slopes"]
    )

    if not np.isfinite(
        probabilities
    ).all():
        raise RuntimeError(
            "Non-finite validation probability."
        )

    if np.any(
        (probabilities <= 0.0)
        | (probabilities >= 1.0)
    ):
        raise RuntimeError(
            "Probability outside strict (0,1) range."
        )

    return probabilities


# ============================================================
# Load frozen artefacts
# ============================================================

dataset = load_csv(DATASET_FILE)
partition = load_csv(PARTITION_FILE)
folds = load_csv(FOLD_FILE)

print("=" * 78)
print("A8.7.3.3 FIVE-MODEL GROUPED CROSS-VALIDATED FLIC FITTING")
print("=" * 78)


# ============================================================
# Resolve schema
# ============================================================

scenario_col = first_existing(
    dataset[0],
    [
        "scenario_id",
        "scenario",
    ],
)

case_col = first_existing(
    dataset[0],
    [
        "case_id",
        "evidence_case_id",
    ],
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
    [
        "actionable",
    ],
)

confidence_col = first_existing(
    dataset[0],
    [
        "mean_self_confidence",
    ],
)

entropy_col = first_existing(
    dataset[0],
    [
        "diagnostic_entropy",
    ],
)

evidence_condition_col = first_existing(
    dataset[0],
    [
        "evidence_condition",
    ],
)


# ============================================================
# Frozen DEVELOPMENT IDs
# ============================================================

partition_scenario_col = first_existing(
    partition[0],
    [
        "scenario_id",
        "scenario",
    ],
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
    if row[partition_col].strip().upper()
    == "DEVELOPMENT"
}

if len(development_ids) != 120:
    raise RuntimeError(
        f"Expected 120 DEVELOPMENT scenarios, "
        f"found {len(development_ids)}."
    )


# ============================================================
# Frozen fold assignment
# ============================================================

fold_scenario_col = first_existing(
    folds[0],
    [
        "scenario_id",
        "scenario",
    ],
)

fold_col = first_existing(
    folds[0],
    [
        "cv_fold",
        "fold",
    ],
)

fold_map = {
    row[fold_scenario_col]:
        int(row[fold_col])
    for row in folds
}

if set(fold_map) != development_ids:
    raise RuntimeError(
        "Fold IDs do not exactly match frozen "
        "DEVELOPMENT IDs."
    )


# ============================================================
# Construct primary modelling dataset
# ============================================================

modelling_rows = []

for row in dataset:

    sid = row[scenario_col]

    if sid not in development_ids:
        continue

    yd = parse_binary_or_none(
        row[yd_col]
    )

    # Primary reliability modelling uses only substantive
    # diagnoses with defined Y_D.
    if yd is None:
        continue

    modelling_rows.append(
        {
            "case_id": row[case_col],
            "scenario_id": sid,
            "cv_fold": fold_map[sid],
            "Y_D": int(yd),
            "E": parse_bool(
                row[actionable_col]
            ),
            "mean_self_confidence": float(
                row[confidence_col]
            ),
            "diagnostic_entropy": float(
                row[entropy_col]
            ),
            "evidence_condition":
                row[evidence_condition_col],
        }
    )


if len(modelling_rows) != 238:
    raise RuntimeError(
        f"Expected 238 DEVELOPMENT Y_D-eligible "
        f"cases, found {len(modelling_rows)}."
    )

y_all = np.array(
    [
        row["Y_D"]
        for row in modelling_rows
    ],
    dtype=int,
)

if int((y_all == 0).sum()) != 193:
    raise RuntimeError(
        "Expected 193 Y_D=0 cases."
    )

if int((y_all == 1).sum()) != 45:
    raise RuntimeError(
        "Expected 45 Y_D=1 cases."
    )


print("\nPrimary modelling dataset:")
print("Eligible cases:", len(modelling_rows))
print("Y_D=0:", int((y_all == 0).sum()))
print("Y_D=1:", int((y_all == 1).sum()))


# ============================================================
# Grouped CV fitting
# ============================================================

prediction_records = []
coefficient_records = []

for fold in range(1, 6):

    training = [
        row
        for row in modelling_rows
        if row["cv_fold"] != fold
    ]

    validation = [
        row
        for row in modelling_rows
        if row["cv_fold"] == fold
    ]

    training_scenarios = {
        row["scenario_id"]
        for row in training
    }

    validation_scenarios = {
        row["scenario_id"]
        for row in validation
    }

    if not training_scenarios.isdisjoint(
        validation_scenarios
    ):
        raise RuntimeError(
            f"Scenario leakage detected in fold {fold}."
        )

    print("\n" + "-" * 78)
    print(
        f"Fold {fold}: "
        f"{len(training)} train / "
        f"{len(validation)} validation"
    )
    print("-" * 78)

    y_train = np.array(
        [
            row["Y_D"]
            for row in training
        ],
        dtype=int,
    )

    for model_name, predictors in MODEL_SPECS.items():

        X_train = np.array(
            [
                [
                    row[predictor]
                    for predictor in predictors
                ]
                for row in training
            ],
            dtype=float,
        )

        X_validation = np.array(
            [
                [
                    row[predictor]
                    for predictor in predictors
                ]
                for row in validation
            ],
            dtype=float,
        )

        fit = fit_firth_with_flic(
            X_train,
            y_train,
        )

        validation_probabilities = predict_flic(
            fit,
            X_validation,
        )

        print(
            f"{model_name}: "
            f"train rate="
            f"{fit['training_event_rate']:.6f}, "
            f"FLIC mean train p="
            f"{fit['training_mean_probability']:.6f}, "
            f"validation p range="
            f"{validation_probabilities.min():.6f}"
            f" to "
            f"{validation_probabilities.max():.6f}"
        )

        # Save fold-level model parameters.
        coefficient_records.append(
            {
                "cv_fold": fold,
                "model": model_name,
                "parameter": "intercept_firth",
                "value": fit["firth_intercept"],
            }
        )

        coefficient_records.append(
            {
                "cv_fold": fold,
                "model": model_name,
                "parameter": "intercept_flic",
                "value": fit["flic_intercept"],
            }
        )

        for predictor, coefficient in zip(
            predictors,
            fit["slopes"],
        ):
            coefficient_records.append(
                {
                    "cv_fold": fold,
                    "model": model_name,
                    "parameter": predictor,
                    "value": float(coefficient),
                }
            )

        # Save out-of-fold predictions.
        for row, probability in zip(
            validation,
            validation_probabilities,
        ):
            prediction_records.append(
                {
                    "case_id": row["case_id"],
                    "scenario_id":
                        row["scenario_id"],
                    "cv_fold": fold,
                    "Y_D": row["Y_D"],
                    "E": row["E"],
                    "mean_self_confidence":
                        row["mean_self_confidence"],
                    "diagnostic_entropy":
                        row["diagnostic_entropy"],
                    "evidence_condition":
                        row["evidence_condition"],
                    "model": model_name,
                    "predicted_reliability":
                        float(probability),
                }
            )


# ============================================================
# Validate prediction coverage
# ============================================================

expected_predictions = (
    len(modelling_rows)
    * len(MODEL_SPECS)
)

if len(prediction_records) != expected_predictions:
    raise RuntimeError(
        f"Expected {expected_predictions} predictions, "
        f"found {len(prediction_records)}."
    )


prediction_keys = [
    (
        row["case_id"],
        row["model"],
    )
    for row in prediction_records
]

if len(set(prediction_keys)) != expected_predictions:
    raise RuntimeError(
        "Duplicate or missing case/model "
        "prediction keys detected."
    )


for model_name in MODEL_SPECS:

    model_predictions = [
        row
        for row in prediction_records
        if row["model"] == model_name
    ]

    if len(model_predictions) != 238:
        raise RuntimeError(
            f"{model_name}: expected 238 OOF "
            f"predictions, found "
            f"{len(model_predictions)}."
        )

    values = np.array(
        [
            row["predicted_reliability"]
            for row in model_predictions
        ],
        dtype=float,
    )

    if not np.isfinite(values).all():
        raise RuntimeError(
            f"{model_name}: non-finite "
            f"OOF probabilities."
        )

    if np.any(
        (values <= 0.0)
        | (values >= 1.0)
    ):
        raise RuntimeError(
            f"{model_name}: OOF probability "
            f"outside strict (0,1)."
        )


# ============================================================
# Write durable artefacts
# ============================================================

with PREDICTIONS_FILE.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    fieldnames = [
        "case_id",
        "scenario_id",
        "cv_fold",
        "Y_D",
        "E",
        "mean_self_confidence",
        "diagnostic_entropy",
        "evidence_condition",
        "model",
        "predicted_reliability",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(prediction_records)


with COEFFICIENTS_FILE.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    fieldnames = [
        "cv_fold",
        "model",
        "parameter",
        "value",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(coefficient_records)


manifest = {
    "stage": "A8.7",
    "substage": "A8.7.3.3",
    "name": (
        "Five-Model Scenario-Grouped "
        "Cross-Validated Reliability Modelling"
    ),
    "status": "COMPLETED_AFTER_VALIDATION",
    "created_at_utc":
        datetime.now(timezone.utc).isoformat(),
    "estimator": (
        "Firth logistic regression with "
        "FLIC intercept correction"
    ),
    "target": "diagnostic_correctness (Y_D)",
    "target_policy": (
        "Primary modelling includes only "
        "substantive modal diagnoses for which "
        "Y_D is defined."
    ),
    "development_scenarios": 120,
    "development_eligible_cases": 238,
    "development_y_d_0": 193,
    "development_y_d_1": 45,
    "n_folds": 5,
    "cv_policy": (
        "Scenario-grouped 5-fold cross-validation "
        "using the frozen outcome-blind assignment."
    ),
    "models": MODEL_SPECS,
    "source_files": {
        "a8_6_dataset": {
            "path": str(DATASET_FILE),
            "sha256": sha256(DATASET_FILE),
        },
        "scenario_partition": {
            "path": str(PARTITION_FILE),
            "sha256": sha256(PARTITION_FILE),
        },
        "cv_assignment": {
            "path": str(FOLD_FILE),
            "sha256": sha256(FOLD_FILE),
        },
    },
    "outputs": {
        "oof_predictions": str(PREDICTIONS_FILE),
        "fold_coefficients": str(COEFFICIENTS_FILE),
    },
    "holdout_policy": (
        "The 30-scenario final holdout is not "
        "used for model fitting or OOF prediction."
    ),
}


with MANIFEST_FILE.open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        manifest,
        f,
        indent=2,
    )


print("\n" + "=" * 78)
print("OUTPUT VALIDATION")
print("=" * 78)

print(
    "\nTotal OOF predictions:",
    len(prediction_records),
)

print(
    "Expected OOF predictions:",
    expected_predictions,
)

for model_name in MODEL_SPECS:

    count = sum(
        row["model"] == model_name
        for row in prediction_records
    )

    print(
        f"{model_name}: {count} predictions"
    )


print("\nOutput artefacts:")
print(PREDICTIONS_FILE)
print(COEFFICIENTS_FILE)
print(MANIFEST_FILE)

print("\nOOF prediction SHA-256:")
print(sha256(PREDICTIONS_FILE))

print("\nValidation:")
print("120 DEVELOPMENT scenarios only: PASS")
print("30-scenario holdout excluded: PASS")
print("Scenario-grouped CV preserved: PASS")
print("Five frozen model specifications fitted: PASS")
print("Firth finite estimation: PASS")
print("FLIC training mean correction: PASS")
print("238 OOF predictions per model: PASS")
print("1,190 total OOF predictions: PASS")
print("Unique case/model prediction keys: PASS")

print(
    "\nA8.7.3.3 FIVE-MODEL GROUPED CV "
    "RELIABILITY FITTING: PASS"
)