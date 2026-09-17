import csv
import json
import math
from pathlib import Path

import numpy as np
from firthmodels import FirthLogisticRegression


# ============================================================
# Paths
# ============================================================

BASE = Path(
    "research/diagnostic_reliability/analysis/outputs"
)

A8_7_DIR = BASE / "a8_7_reliability_modelling"

A8_8_DIR = BASE / "a8_8_evaluation_and_calibration"

PREDICTIONS_FILE = (
    A8_7_DIR
    / "a8_7_grouped_cv_out_of_fold_reliability_predictions.csv"
)

OUTPUT_FILE = (
    A8_8_DIR
    / "a8_8_detailed_calibration_assessment.csv"
)

SUMMARY_FILE = (
    A8_8_DIR
    / "a8_8_detailed_calibration_assessment.json"
)


MODEL_ORDER = [
    "M_E",
    "M_C",
    "M_H",
    "M_U",
    "M_EU",
]


# ============================================================
# Utilities
# ============================================================

def load_csv(path):
    with path.open(
        newline="",
        encoding="utf-8",
    ) as f:
        return list(csv.DictReader(f))


def logit(p):
    return np.log(
        p / (1.0 - p)
    )


def sigmoid(x):
    return 1.0 / (
        1.0 + np.exp(-x)
    )


def fit_flic_calibration(y, p):
    """
    Fit:
        logit P(Y=1) = intercept + slope * logit(p)

    using Firth logistic regression followed by
    FLIC intercept correction.

    The slope is retained from the Firth fit.
    Only the intercept is shifted so that the
    mean fitted probability equals the observed
    event rate.
    """

    x = logit(p).reshape(-1, 1)

    model = FirthLogisticRegression()
    model.fit(x, y)

    firth_intercept = float(
        np.asarray(
            model.intercept_
        ).reshape(-1)[0]
    )

    slope = float(
        np.asarray(
            model.coef_
        ).reshape(-1)[0]
    )

    linear_without_intercept = (
        slope * x[:, 0]
    )

    target_rate = float(
        np.mean(y)
    )

    # --------------------------------------------------------
    # Solve for FLIC intercept by bisection:
    #
    # mean(sigmoid(a + slope*x)) = observed event rate
    # --------------------------------------------------------

    lower = -50.0
    upper = 50.0

    for _ in range(200):

        midpoint = (
            lower + upper
        ) / 2.0

        mean_probability = float(
            np.mean(
                sigmoid(
                    midpoint
                    + linear_without_intercept
                )
            )
        )

        if mean_probability < target_rate:
            lower = midpoint
        else:
            upper = midpoint

    flic_intercept = (
        lower + upper
    ) / 2.0

    fitted_probability = sigmoid(
        flic_intercept
        + linear_without_intercept
    )

    return {
        "firth_intercept":
            firth_intercept,

        "flic_intercept":
            float(flic_intercept),

        "calibration_slope":
            slope,

        "mean_recalibrated_probability":
            float(
                np.mean(
                    fitted_probability
                )
            ),

        "observed_rate":
            target_rate,
    }


# ============================================================
# Load OOF predictions
# ============================================================

rows = load_csv(
    PREDICTIONS_FILE
)

if len(rows) != 1190:
    raise RuntimeError(
        f"Expected 1,190 OOF rows, "
        f"found {len(rows)}."
    )


print("=" * 80)
print("A8.8.4 DETAILED CALIBRATION ASSESSMENT")
print("=" * 80)

print(
    "\nAssessment data: "
    "development out-of-fold predictions"
)

print(
    "Holdout used: NO"
)


# ============================================================
# Assess models
# ============================================================

records = []


for model_name in MODEL_ORDER:

    model_rows = [
        row
        for row in rows
        if row["model"] == model_name
    ]

    if len(model_rows) != 238:
        raise RuntimeError(
            f"{model_name}: expected 238 rows, "
            f"found {len(model_rows)}."
        )

    y = np.array(
        [
            int(row["Y_D"])
            for row in model_rows
        ],
        dtype=int,
    )

    p = np.array(
        [
            float(
                row[
                    "predicted_reliability"
                ]
            )
            for row in model_rows
        ],
        dtype=float,
    )

    if not np.isfinite(p).all():
        raise RuntimeError(
            f"{model_name}: non-finite probabilities."
        )

    if np.any(
        (p <= 0.0)
        | (p >= 1.0)
    ):
        raise RuntimeError(
            f"{model_name}: probabilities "
            "must lie strictly inside (0,1)."
        )

    observed_rate = float(
        np.mean(y)
    )

    mean_predicted = float(
        np.mean(p)
    )

    calibration_in_large_difference = (
        mean_predicted
        - observed_rate
    )

    calibration = fit_flic_calibration(
        y,
        p,
    )

    if not all(
        math.isfinite(
            float(value)
        )
        for value in calibration.values()
    ):
        raise RuntimeError(
            f"{model_name}: non-finite "
            "calibration result."
        )

    flic_mean_error = abs(
        calibration[
            "mean_recalibrated_probability"
        ]
        - observed_rate
    )

    if flic_mean_error > 1e-10:
        raise RuntimeError(
            f"{model_name}: FLIC calibration "
            "mean constraint failed."
        )

    record = {
        "model": model_name,
        "n": len(y),
        "positive_cases":
            int(np.sum(y == 1)),
        "negative_cases":
            int(np.sum(y == 0)),
        "observed_correctness_rate":
            observed_rate,
        "mean_predicted_reliability":
            mean_predicted,
        "mean_prediction_minus_observed":
            calibration_in_large_difference,
        "firth_calibration_intercept":
            calibration[
                "firth_intercept"
            ],
        "flic_calibration_intercept":
            calibration[
                "flic_intercept"
            ],
        "calibration_slope":
            calibration[
                "calibration_slope"
            ],
        "mean_flic_recalibrated_probability":
            calibration[
                "mean_recalibrated_probability"
            ],
    }

    records.append(record)

    print(
        f"\n{model_name}"
    )
    print("-" * 50)

    print(
        "Observed correctness:",
        f"{observed_rate:.6f}"
    )

    print(
        "Mean predicted reliability:",
        f"{mean_predicted:.6f}"
    )

    print(
        "Mean prediction - observed:",
        f"{calibration_in_large_difference:.6f}"
    )

    print(
        "Firth calibration intercept:",
        f"{calibration['firth_intercept']:.6f}"
    )

    print(
        "FLIC calibration intercept:",
        f"{calibration['flic_intercept']:.6f}"
    )

    print(
        "Calibration slope:",
        f"{calibration['calibration_slope']:.6f}"
    )

    print(
        "Mean FLIC recalibrated probability:",
        f"{calibration['mean_recalibrated_probability']:.6f}"
    )


# ============================================================
# Write CSV
# ============================================================

with OUTPUT_FILE.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    fieldnames = [
        "model",
        "n",
        "positive_cases",
        "negative_cases",
        "observed_correctness_rate",
        "mean_predicted_reliability",
        "mean_prediction_minus_observed",
        "firth_calibration_intercept",
        "flic_calibration_intercept",
        "calibration_slope",
        "mean_flic_recalibrated_probability",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(records)


# ============================================================
# Durable summary
# ============================================================

summary = {
    "stage": "A8.8",
    "substage": "A8.8.4",
    "name": "Detailed Calibration Assessment",
    "status": "COMPLETED_AFTER_VALIDATION",

    "data": {
        "type": (
            "development out-of-fold predictions"
        ),
        "cases_per_model": 238,
        "positive_cases": 45,
        "negative_cases": 193,
        "holdout_used": False,
    },

    "method": {
        "calibration_model": (
            "Firth logistic calibration model "
            "with FLIC intercept correction"
        ),
        "equation": (
            "logit P(Y_D=1) = alpha + "
            "beta * logit(predicted_reliability)"
        ),
        "ideal_intercept": 0.0,
        "ideal_slope": 1.0,
        "interpretation_warning": (
            "Calibration intercept and slope are "
            "descriptive development-stage diagnostics. "
            "They are not used here to select a model "
            "or fit a post-hoc calibrator."
        ),
    },

    "results": records,

    "model_selection_performed": False,
    "posthoc_recalibration_performed": False,
    "threshold_selection_performed": False,
}


with SUMMARY_FILE.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


# ============================================================
# Final validation
# ============================================================

print("\n" + "=" * 80)
print("OUTPUT ARTEFACTS")
print("=" * 80)

print(OUTPUT_FILE)
print(SUMMARY_FILE)

print("\nValidation:")
print("Five models assessed: PASS")
print("238 OOF cases per model: PASS")
print("Calibration-in-the-large calculated: PASS")
print("Firth calibration models fitted: PASS")
print("FLIC intercept corrections applied: PASS")
print("Calibration slopes calculated: PASS")
print("FLIC mean constraints verified: PASS")
print("Holdout untouched: PASS")
print("No model selection performed: PASS")
print("No post-hoc calibrator fitted: PASS")

print(
    "\nA8.8.4 DETAILED CALIBRATION "
    "ASSESSMENT: PASS"
)