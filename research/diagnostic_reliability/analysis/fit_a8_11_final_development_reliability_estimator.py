import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from firthmodels import FirthLogisticRegression


# ============================================================
# Frozen configuration
# ============================================================

MODEL = "M_EU"

PREDICTORS = [
    "E",
    "mean_self_confidence",
    "diagnostic_entropy",
]

OUTCOME = "diagnostic_correctness"


# ============================================================
# Paths
# ============================================================

ANALYSIS = Path(
    "research/diagnostic_reliability/analysis"
)

OUTPUTS = ANALYSIS / "outputs"

A8_6_FILE = (
    OUTPUTS
    / "a8_6_outcome_uncertainty_construction"
    / "a8_6_evidence_case_reliability_dataset.csv"
)

A8_7_DIR = (
    OUTPUTS
    / "a8_7_reliability_modelling"
)

PARTITION_FILE = (
    A8_7_DIR
    / "a8_7_scenario_development_holdout_assignment.csv"
)

A8_9_FILE = (
    OUTPUTS
    / "a8_9_reliability_estimator_selection"
    / "a8_9_reliability_estimator_selection.json"
)

A8_10_FILE = (
    OUTPUTS
    / "a8_10_post_inference_reliability_policy"
    / "a8_10_frozen_post_inference_reliability_policy.json"
)

A8_11_DIR = (
    OUTPUTS
    / "a8_11_homepot_integration"
)

A8_11_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PARAMETERS_FILE = (
    A8_11_DIR
    / "a8_11_final_development_meu_parameters.json"
)

PREDICTIONS_FILE = (
    A8_11_DIR
    / "a8_11_final_development_meu_fitted_predictions.csv"
)

MANIFEST_FILE = (
    A8_11_DIR
    / "a8_11_final_development_estimator_manifest.json"
)


# ============================================================
# Utilities
# ============================================================

def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def load_csv(path):
    with path.open(
        newline="",
        encoding="utf-8",
    ) as f:
        return list(
            csv.DictReader(f)
        )


def logistic(z):
    z = np.asarray(
        z,
        dtype=float,
    )

    return 1.0 / (
        1.0 + np.exp(-z)
    )


def solve_flic_intercept(
    X,
    y,
    slopes,
):
    """
    FLIC correction used in frozen A8.7 protocol.

    Retain the Firth slopes and solve only the
    intercept so that mean fitted probability on
    the training data equals the observed event rate.
    """

    linear_without_intercept = (
        X @ slopes
    )

    target = float(
        np.mean(y)
    )

    def objective(intercept):
        probabilities = logistic(
            intercept
            + linear_without_intercept
        )

        return (
            float(
                np.mean(
                    probabilities
                )
            )
            - target
        )

    return float(
        brentq(
            objective,
            -100.0,
            100.0,
        )
    )


# ============================================================
# Validate frozen research decisions
# ============================================================

required_files = [
    A8_6_FILE,
    PARTITION_FILE,
    A8_9_FILE,
    A8_10_FILE,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(
            path
        )


with A8_9_FILE.open(
    encoding="utf-8",
) as f:
    selection = json.load(f)


if (
    selection[
        "selected_candidate"
    ][
        "model"
    ]
    != MODEL
):
    raise RuntimeError(
        "Frozen A8.9 selected "
        "estimator is not M_EU."
    )


with A8_10_FILE.open(
    encoding="utf-8",
) as f:
    policy = json.load(f)


if (
    policy[
        "selected_reliability_estimator"
    ]
    != MODEL
):
    raise RuntimeError(
        "Frozen A8.10 policy does "
        "not use M_EU."
    )


if abs(
    float(
        policy[
            "issue_threshold"
        ][
            "value"
        ]
    )
    - 0.64
) > 1e-12:
    raise RuntimeError(
        "Unexpected frozen "
        "A8.10 Issue threshold."
    )


# ============================================================
# Load frozen scenario partition
# ============================================================

partition_rows = load_csv(
    PARTITION_FILE
)


development_scenarios = set()
holdout_scenarios = set()


for row in partition_rows:

    scenario_id = (
        row["scenario_id"]
    )

    partition = (
        row["partition"]
        .strip()
        .lower()
    )

    if partition == "development":
        development_scenarios.add(
            scenario_id
        )

    elif partition == "holdout":
        holdout_scenarios.add(
            scenario_id
        )

    else:
        raise RuntimeError(
            f"Unknown partition: "
            f"{partition}"
        )


if len(
    development_scenarios
) != 120:
    raise RuntimeError(
        "Expected 120 development "
        "scenarios."
    )


if len(
    holdout_scenarios
) != 30:
    raise RuntimeError(
        "Expected 30 holdout "
        "scenarios."
    )


if (
    development_scenarios
    & holdout_scenarios
):
    raise RuntimeError(
        "Development/holdout "
        "scenario overlap detected."
    )


# ============================================================
# Load A8.6 analytical dataset
# ============================================================

rows = load_csv(
    A8_6_FILE
)


development_rows = []


for row in rows:

    scenario_id = (
        row["scenario_id"]
    )

    # Critical:
    # holdout rows are not used below.
    if (
        scenario_id
        not in development_scenarios
    ):
        continue

    y_raw = (
        row[
            OUTCOME
        ]
        .strip()
    )

    # Primary M_EU estimator is defined only
    # for Y_D-eligible substantive diagnoses.
    if y_raw == "":
        continue

    y = int(
        float(
            y_raw
        )
    )

    if y not in {0, 1}:
        raise RuntimeError(
            f"Unexpected Y_D value: {y}"
        )

    actionable_raw = (
        row["actionable"]
        .strip()
        .lower()
    )

    if actionable_raw in {
        "true",
        "1",
        "1.0",
    }:
        E = 1.0

    elif actionable_raw in {
        "false",
        "0",
        "0.0",
    }:
        E = 0.0

    else:
        raise RuntimeError(
            "Unexpected actionable "
            f"value: {row['actionable']}"
        )

    development_rows.append({
        "case_id":
            row["case_id"],

        "scenario_id":
            scenario_id,

        "evidence_condition":
            row[
                "evidence_condition"
            ],

        "E":
            E,

        "mean_self_confidence":
            float(
                row[
                    "mean_self_confidence"
                ]
            ),

        "diagnostic_entropy":
            float(
                row[
                    "diagnostic_entropy"
                ]
            ),

        "Y_D":
            y,
    })


# ============================================================
# Development-only integrity checks
# ============================================================

if len(
    development_rows
) != 238:
    raise RuntimeError(
        f"Expected 238 eligible "
        f"development cases, found "
        f"{len(development_rows)}."
    )


represented_scenarios = set(
    row["scenario_id"]
    for row in development_rows
)


if len(
    represented_scenarios
) != 68:
    raise RuntimeError(
        "Expected 68 represented "
        "Y_D-eligible development scenarios."
    )


if (
    represented_scenarios
    & holdout_scenarios
):
    raise RuntimeError(
        "Holdout contamination "
        "detected."
    )


# ============================================================
# Construct frozen M_EU design matrix
# ============================================================

X = np.asarray(
    [
        [
            row["E"],
            row[
                "mean_self_confidence"
            ],
            row[
                "diagnostic_entropy"
            ],
        ]
        for row in development_rows
    ],
    dtype=float,
)

y = np.asarray(
    [
        row["Y_D"]
        for row in development_rows
    ],
    dtype=int,
)


if X.shape != (
    238,
    3,
):
    raise RuntimeError(
        f"Unexpected X shape: "
        f"{X.shape}"
    )


n_positive = int(
    np.sum(y == 1)
)

n_negative = int(
    np.sum(y == 0)
)


if n_positive != 45:
    raise RuntimeError(
        f"Expected 45 positive "
        f"development outcomes, "
        f"found {n_positive}."
    )


if n_negative != 193:
    raise RuntimeError(
        f"Expected 193 negative "
        f"development outcomes, "
        f"found {n_negative}."
    )


# ============================================================
# Fit frozen estimator family
# ============================================================

firth = FirthLogisticRegression()

firth.fit(
    X,
    y,
)


firth_slopes = np.asarray(
    firth.coef_,
    dtype=float,
).reshape(-1)


if len(
    firth_slopes
) != 3:
    raise RuntimeError(
        "Unexpected number of "
        "Firth slopes."
    )


raw_intercept = float(
    np.asarray(
        firth.intercept_,
        dtype=float,
    ).reshape(-1)[0]
)


# ============================================================
# Apply frozen FLIC correction
# ============================================================

flic_intercept = (
    solve_flic_intercept(
        X=X,
        y=y,
        slopes=firth_slopes,
    )
)


linear_predictor = (
    flic_intercept
    + X @ firth_slopes
)


predicted_reliability = (
    logistic(
        linear_predictor
    )
)


observed_event_rate = float(
    np.mean(y)
)

mean_predicted_reliability = float(
    np.mean(
        predicted_reliability
    )
)


if abs(
    mean_predicted_reliability
    - observed_event_rate
) > 1e-10:
    raise RuntimeError(
        "FLIC calibration identity "
        "failed."
    )


if not np.all(
    np.isfinite(
        predicted_reliability
    )
):
    raise RuntimeError(
        "Non-finite fitted "
        "probabilities detected."
    )


# ============================================================
# Export fitted development predictions
# ============================================================

with PREDICTIONS_FILE.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    fieldnames = [
        "case_id",
        "scenario_id",
        "evidence_condition",
        "E",
        "mean_self_confidence",
        "diagnostic_entropy",
        "Y_D",
        "predicted_reliability",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    for row, probability in zip(
        development_rows,
        predicted_reliability,
    ):

        writer.writerow({
            **row,
            "predicted_reliability":
                float(probability),
        })


# ============================================================
# Export deployment parameters
# ============================================================

parameters = {
    "stage":
        "A8.11",

    "substage":
        "A8.11.2",

    "model":
        MODEL,

    "status":
        "FROZEN_FOR_PROTOTYPE_INTEGRATION",

    "target":
        "diagnostic_correctness",

    "estimator":
        "Firth logistic regression + FLIC",

    "predictors": [
        {
            "name":
                "E",

            "source":
                "actionable",

            "encoding":
                "1=actionable, 0=non-actionable",

            "coefficient":
                float(
                    firth_slopes[0]
                ),
        },
        {
            "name":
                "mean_self_confidence",

            "coefficient":
                float(
                    firth_slopes[1]
                ),
        },
        {
            "name":
                "diagnostic_entropy",

            "coefficient":
                float(
                    firth_slopes[2]
                ),
        },
    ],

    "firth_intercept_before_flic":
        raw_intercept,

    "flic_intercept":
        flic_intercept,

    "prediction_equation": (
        "sigmoid("
        "flic_intercept"
        " + beta_E * E"
        " + beta_C * mean_self_confidence"
        " + beta_H * diagnostic_entropy"
        ")"
    ),

    "development_fit": {
        "eligible_cases":
            len(
                development_rows
            ),

        "represented_scenarios":
            len(
                represented_scenarios
            ),

        "positive_outcomes":
            n_positive,

        "negative_outcomes":
            n_negative,

        "observed_correctness_rate":
            observed_event_rate,

        "mean_fitted_probability":
            mean_predicted_reliability,
    },

    "policy": {
        "issue_threshold":
            0.64,

        "E_equals_0":
            "WITHHOLD",

        "E_equals_1_below_threshold":
            "QUALIFY",

        "E_equals_1_at_or_above_threshold":
            "ISSUE",
    },

    "unknown_semantics": {
        "UNKNOWN_is_substantive_fault":
            False,

        "apply_primary_reliability_estimator":
            False,

        "handling":
            "ABSTAIN",
    },

    "holdout": {
        "scenario_count":
            30,

        "used_for_fit":
            False,

        "used_for_tuning":
            False,

        "used_for_evaluation":
            False,

        "status":
            "UNTOUCHED",
    },
}


with PARAMETERS_FILE.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        parameters,
        f,
        indent=2,
    )


# ============================================================
# Durable manifest
# ============================================================

manifest = {
    "stage":
        "A8.11.2",

    "status":
        "COMPLETE_AND_FROZEN",

    "purpose": (
        "Fit the already-selected "
        "M_EU specification once on "
        "all eligible development data "
        "to obtain deployable prototype "
        "parameters."
    ),

    "model_selection_performed":
        False,

    "hyperparameter_tuning_performed":
        False,

    "threshold_selection_performed":
        False,

    "predictor_selection_performed":
        False,

    "holdout_used":
        False,

    "inputs": {
        str(A8_6_FILE):
            sha256(A8_6_FILE),

        str(PARTITION_FILE):
            sha256(PARTITION_FILE),

        str(A8_9_FILE):
            sha256(A8_9_FILE),

        str(A8_10_FILE):
            sha256(A8_10_FILE),
    },

    "outputs": {
        str(PARAMETERS_FILE):
            sha256(PARAMETERS_FILE),

        str(PREDICTIONS_FILE):
            sha256(PREDICTIONS_FILE),
    },
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


# ============================================================
# Report
# ============================================================

print("=" * 80)
print(
    "A8.11.2 FINAL DEVELOPMENT "
    "RELIABILITY ESTIMATOR FIT"
)
print("=" * 80)

print("\nFrozen model:")
print("  M_EU")

print("\nPredictors:")
print(
    "  E, mean_self_confidence, "
    "diagnostic_entropy"
)

print("\nEstimator:")
print(
    "  Firth logistic regression "
    "+ FLIC"
)

print("\nDevelopment data:")
print(
    "  Eligible cases:",
    len(development_rows),
)
print(
    "  Represented scenarios:",
    len(represented_scenarios),
)
print(
    "  Correct:",
    n_positive,
)
print(
    "  Incorrect:",
    n_negative,
)

print("\nFirth coefficients:")
print(
    "  Raw Firth intercept:",
    f"{raw_intercept:.9f}",
)
print(
    "  beta_E:",
    f"{firth_slopes[0]:.9f}",
)
print(
    "  beta_C:",
    f"{firth_slopes[1]:.9f}",
)
print(
    "  beta_H:",
    f"{firth_slopes[2]:.9f}",
)

print("\nFLIC:")
print(
    "  FLIC intercept:",
    f"{flic_intercept:.9f}",
)
print(
    "  Observed correctness:",
    f"{observed_event_rate:.9f}",
)
print(
    "  Mean fitted probability:",
    f"{mean_predicted_reliability:.9f}",
)

print("\nFrozen policy:")
print("  E=0 -> WITHHOLD")
print(
    "  E=1 and p<0.64 -> QUALIFY"
)
print(
    "  E=1 and p>=0.64 -> ISSUE"
)

print("\nUNKNOWN:")
print(
    "  Primary reliability estimator "
    "not applied -> ABSTAIN"
)

print("\nHoldout:")
print("  30 scenarios")
print("  Used: NO")
print("  Status: UNTOUCHED")

print("\nOutput artefacts:")
print(PARAMETERS_FILE)
print(PREDICTIONS_FILE)
print(MANIFEST_FILE)

print("\nValidation:")
print("Frozen M_EU specification: PASS")
print("Raw predictor encoding retained: PASS")
print("238 development cases: PASS")
print("45 correct / 193 incorrect: PASS")
print("68 represented scenarios: PASS")
print("Firth fit finite: PASS")
print("FLIC calibration identity: PASS")
print("A8.10 policy unchanged: PASS")
print("No model selection: PASS")
print("No threshold selection: PASS")
print("Holdout untouched: PASS")

print(
    "\nA8.11.2 FINAL DEVELOPMENT "
    "ESTIMATOR: COMPLETE + FROZEN"
)