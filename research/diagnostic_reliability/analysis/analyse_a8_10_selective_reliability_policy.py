import csv
import json
from pathlib import Path

import numpy as np


# ============================================================
# Configuration
# ============================================================

MODELS = [
    "M_E",
    "M_EU",
]

THRESHOLDS = np.round(
    np.arange(
        0.00,
        1.0001,
        0.01,
    ),
    2,
)


# ============================================================
# Paths
# ============================================================

BASE = Path(
    "research/diagnostic_reliability/analysis/outputs"
)

A8_7_DIR = (
    BASE
    / "a8_7_reliability_modelling"
)

A8_9_DIR = (
    BASE
    / "a8_9_reliability_estimator_selection"
)

A8_10_DIR = (
    BASE
    / "a8_10_post_inference_reliability_policy"
)

A8_10_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PREDICTIONS_FILE = (
    A8_7_DIR
    / "a8_7_grouped_cv_out_of_fold_reliability_predictions.csv"
)

SELECTION_FILE = (
    A8_9_DIR
    / "a8_9_reliability_estimator_selection.json"
)

PROFILE_FILE = (
    A8_10_DIR
    / "a8_10_development_selective_risk_coverage_profile.csv"
)

SUMMARY_FILE = (
    A8_10_DIR
    / "a8_10_development_selective_risk_coverage_summary.json"
)


# ============================================================
# Utilities
# ============================================================

def load_csv(path):
    with path.open(
        newline="",
        encoding="utf-8",
    ) as f:
        return list(
            csv.DictReader(f)
        )


# ============================================================
# Load frozen inputs
# ============================================================

rows = load_csv(
    PREDICTIONS_FILE
)

with SELECTION_FILE.open(
    encoding="utf-8",
) as f:
    selection = json.load(f)


if (
    selection[
        "selected_candidate"
    ][
        "model"
    ]
    != "M_EU"
):
    raise RuntimeError(
        "Frozen A8.9 selected candidate "
        "is not M_EU."
    )

if (
    selection[
        "primary_benchmark"
    ][
        "model"
    ]
    != "M_E"
):
    raise RuntimeError(
        "Frozen A8.9 primary benchmark "
        "is not M_E."
    )


print("=" * 80)
print("A8.10.1 DEVELOPMENT SELECTIVE-RISK / COVERAGE PROFILE")
print("=" * 80)

print("\nSelected estimator: M_EU")
print("Primary benchmark: M_E")
print("Holdout used: NO")


# ============================================================
# Construct profiles
# ============================================================

profile_records = []
summary = {}


for model in MODELS:

    model_rows = [
        row
        for row in rows
        if row["model"] == model
    ]

    if len(model_rows) != 238:
        raise RuntimeError(
            f"{model}: expected 238 "
            f"development OOF cases, "
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
            f"{model}: non-finite probabilities."
        )

    n_total = len(y)

    model_records = []

    for threshold in THRESHOLDS:

        accepted = (
            p >= threshold
        )

        n_accepted = int(
            np.sum(accepted)
        )

        n_withheld = (
            n_total
            - n_accepted
        )

        coverage = (
            n_accepted
            / n_total
        )

        if n_accepted > 0:

            correct_accepted = int(
                np.sum(
                    y[accepted] == 1
                )
            )

            incorrect_accepted = int(
                np.sum(
                    y[accepted] == 0
                )
            )

            accepted_accuracy = (
                correct_accepted
                / n_accepted
            )

            selective_risk = (
                incorrect_accepted
                / n_accepted
            )

        else:

            correct_accepted = 0
            incorrect_accepted = 0
            accepted_accuracy = None
            selective_risk = None

        record = {
            "model": model,
            "threshold":
                float(threshold),
            "n_total":
                n_total,
            "n_accepted":
                n_accepted,
            "n_withheld":
                n_withheld,
            "coverage":
                coverage,
            "correct_accepted":
                correct_accepted,
            "incorrect_accepted":
                incorrect_accepted,
            "accepted_accuracy":
                accepted_accuracy,
            "selective_risk":
                selective_risk,
        }

        profile_records.append(
            record
        )

        model_records.append(
            record
        )

    nonempty = [
        r
        for r in model_records
        if r[
            "n_accepted"
        ] > 0
    ]

    highest_threshold_nonempty = max(
        nonempty,
        key=lambda r: r["threshold"],
    )

    highest_accuracy_record = max(
        nonempty,
        key=lambda r: (
            r["accepted_accuracy"],
            r["coverage"],
        ),
    )

    summary[model] = {
        "n_cases":
            n_total,

        "observed_correctness_rate":
            float(np.mean(y)),

        "prediction_min":
            float(np.min(p)),

        "prediction_max":
            float(np.max(p)),

        "highest_nonempty_threshold":
            highest_threshold_nonempty[
                "threshold"
            ],

        "highest_accuracy_observed":
            highest_accuracy_record[
                "accepted_accuracy"
            ],

        "coverage_at_highest_accuracy":
            highest_accuracy_record[
                "coverage"
            ],

        "threshold_at_highest_accuracy":
            highest_accuracy_record[
                "threshold"
            ],
    }


# ============================================================
# Write profile
# ============================================================

with PROFILE_FILE.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    fieldnames = [
        "model",
        "threshold",
        "n_total",
        "n_accepted",
        "n_withheld",
        "coverage",
        "correct_accepted",
        "incorrect_accepted",
        "accepted_accuracy",
        "selective_risk",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(
        profile_records
    )


# ============================================================
# Print informative operating points
# ============================================================

print("\n" + "=" * 80)
print("SELECTIVE-RISK PROFILES")
print("=" * 80)


for model in MODELS:

    print(f"\n{model}")
    print("-" * 60)

    print(
        "Prediction range:",
        f"{summary[model]['prediction_min']:.6f}",
        "to",
        f"{summary[model]['prediction_max']:.6f}",
    )

    print(
        "Observed correctness rate:",
        f"{summary[model]['observed_correctness_rate']:.6f}",
    )

    print("\nSelected descriptive thresholds:")

    model_records = [
        r
        for r in profile_records
        if r["model"] == model
    ]

    for target in [
        0.10,
        0.25,
        0.50,
        0.60,
        0.70,
        0.75,
        0.80,
        0.85,
    ]:

        record = next(
            r
            for r in model_records
            if abs(
                r["threshold"]
                - target
            ) < 1e-12
        )

        if (
            record[
                "accepted_accuracy"
            ]
            is None
        ):
            accuracy_text = "NA"
            risk_text = "NA"

        else:
            accuracy_text = (
                f"{record['accepted_accuracy']:.6f}"
            )

            risk_text = (
                f"{record['selective_risk']:.6f}"
            )

        print(
            f"  t={target:.2f}: "
            f"accepted={record['n_accepted']:3d}, "
            f"coverage={record['coverage']:.6f}, "
            f"accuracy={accuracy_text}, "
            f"risk={risk_text}"
        )


# ============================================================
# Durable summary
# ============================================================

durable_summary = {
    "stage": "A8.10",
    "substage": "A8.10.1",
    "name": (
        "Development Selective-Risk "
        "and Coverage Profile"
    ),
    "status": "COMPLETED_AFTER_VALIDATION",

    "selected_model": "M_EU",
    "benchmark_model": "M_E",

    "data": {
        "type": (
            "development out-of-fold "
            "reliability predictions"
        ),
        "cases_per_model": 238,
        "holdout_used": False,
    },

    "threshold_grid": {
        "minimum": 0.0,
        "maximum": 1.0,
        "step": 0.01,
    },

    "policy_semantics": {
        "issue": (
            "Diagnosis may be presented normally "
            "when the final higher reliability "
            "criterion is satisfied."
        ),

        "qualify": (
            "Diagnosis may be presented with "
            "explicit reliability qualification "
            "when it falls between the final "
            "lower and higher criteria."
        ),

        "withhold": (
            "Generated fault diagnosis is not "
            "presented as sufficiently reliable "
            "for operational decision support "
            "when the lower criterion is not met."
        ),
    },

    "definitions": {
        "coverage": (
            "Proportion of eligible diagnoses "
            "with predicted reliability at or "
            "above a candidate threshold."
        ),

        "selective_risk": (
            "Proportion of accepted diagnoses "
            "that are diagnostically incorrect."
        ),
    },

    "model_summaries":
        summary,

    "thresholds_selected":
        False,

    "model_retrained":
        False,

    "holdout_used":
        False,
}


with SUMMARY_FILE.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        durable_summary,
        f,
        indent=2,
    )


print("\n" + "=" * 80)
print("OUTPUT ARTEFACTS")
print("=" * 80)

print(PROFILE_FILE)
print(SUMMARY_FILE)

print("\nValidation:")
print("Selected M_EU analysed: PASS")
print("Benchmark M_E analysed: PASS")
print("238 OOF cases per model: PASS")
print("Threshold grid 0.00 to 1.00: PASS")
print("Coverage calculated: PASS")
print("Selective risk calculated: PASS")
print("Holdout untouched: PASS")
print("No thresholds selected: PASS")
print("No model retraining: PASS")

print(
    "\nA8.10.1 DEVELOPMENT SELECTIVE-RISK "
    "PROFILE: PASS"
)