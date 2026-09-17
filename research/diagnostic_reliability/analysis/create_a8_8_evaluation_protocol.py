import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


SEED = 20260917
BOOTSTRAP_REPLICATES = 2000
CALIBRATION_BINS = 5

BASE = Path(
    "research/diagnostic_reliability/analysis/outputs"
)

A8_7_DIR = (
    BASE
    / "a8_7_reliability_modelling"
)

A8_8_DIR = (
    BASE
    / "a8_8_evaluation_and_calibration"
)

A8_8_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PREDICTIONS_FILE = (
    A8_7_DIR
    / "a8_7_grouped_cv_out_of_fold_reliability_predictions.csv"
)

OUTPUT_FILE = (
    A8_8_DIR
    / "a8_8_evaluation_protocol.json"
)


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


if not PREDICTIONS_FILE.exists():
    raise FileNotFoundError(
        PREDICTIONS_FILE
    )


protocol = {
    "stage": "A8.8",
    "substage": "A8.8.1",
    "name": "Reliability Evaluation Protocol",
    "status": "FROZEN_BEFORE_PERFORMANCE_EVALUATION",
    "created_at_utc": datetime.now(
        timezone.utc
    ).isoformat(),

    "evaluation_data": {
        "type": (
            "scenario-grouped out-of-fold "
            "development predictions"
        ),
        "eligible_cases": 238,
        "positive_cases": 45,
        "negative_cases": 193,
        "target": "Y_D",
        "prediction": (
            "predicted_reliability = "
            "estimated P(Y_D=1)"
        ),
        "source_file": str(
            PREDICTIONS_FILE
        ),
        "source_sha256": sha256(
            PREDICTIONS_FILE
        ),
    },

    "candidate_models": [
        "M_E",
        "M_C",
        "M_H",
        "M_U",
        "M_EU",
    ],

    "metrics": {
        "brier_score": {
            "role": "primary",
            "direction": "lower_is_better",
            "purpose": (
                "Overall probabilistic "
                "prediction accuracy."
            ),
        },

        "auroc": {
            "role": "supporting",
            "direction": "higher_is_better",
            "purpose": (
                "Discrimination between "
                "correct and incorrect diagnoses."
            ),
        },

        "auprc": {
            "role": "supporting",
            "direction": "higher_is_better",
            "purpose": (
                "Discrimination with explicit "
                "attention to class imbalance."
            ),
        },

        "ece": {
            "role": "supporting",
            "direction": "lower_is_better",
            "purpose": (
                "Summary of calibration error."
            ),
            "warning": (
                "ECE depends on binning and "
                "must not be interpreted alone."
            ),
        },

        "reliability_analysis": {
            "role": "primary_descriptive",
            "purpose": (
                "Compare mean predicted "
                "reliability with observed "
                "correctness."
            ),
        },
    },

    "calibration": {
        "primary_bins": CALIBRATION_BINS,
        "binning_strategy": "quantile",
        "reason": (
            "Candidate models have substantially "
            "different probability distributions; "
            "quantile bins reduce sparse or empty "
            "bin problems in descriptive "
            "reliability analysis."
        ),
    },

    "uncertainty_estimation": {
        "method": "scenario_level_bootstrap",
        "experimental_unit": "scenario_id",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": SEED,
        "confidence_level": 0.95,
        "reason": (
            "Evidence cases are nested within "
            "underlying scenarios and should not "
            "be treated as fully independent "
            "rows during resampling."
        ),
    },

    "comparison_policy": {
        "primary_numerical_metric":
            "brier_score",

        "model_selection_stage":
            "A8.9",

        "no_model_selected_in_a8_8":
            True,

        "no_threshold_selection_in_a8_8":
            True,

        "no_holdout_use":
            True,

        "interpretation": (
            "Models are evaluated across "
            "probabilistic accuracy, calibration "
            "and discrimination. No single "
            "supporting metric determines the "
            "final estimator."
        ),
    },

    "holdout_policy": {
        "status": "UNTOUCHED_FOR_MODEL_SELECTION",
        "scenario_count": 30,
        "use_in_a8_8": False,
    },
}


with OUTPUT_FILE.open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        protocol,
        f,
        indent=2,
    )


print("=" * 72)
print("A8.8.1 RELIABILITY EVALUATION PROTOCOL")
print("=" * 72)

print("\nEvaluation source:")
print(PREDICTIONS_FILE)

print("\nSource SHA-256:")
print(
    protocol[
        "evaluation_data"
    ][
        "source_sha256"
    ]
)

print("\nCandidate models:")
for model in protocol[
    "candidate_models"
]:
    print(" ", model)

print("\nEvaluation framework:")
print(" Primary numerical metric: Brier score")
print(" Discrimination: AUROC + AUPRC")
print(" Calibration: reliability analysis + ECE")
print(" Calibration bins: 5 quantile bins")
print(" Bootstrap: 2,000 scenario-level replicates")
print(" Bootstrap seed:", SEED)

print("\nPolicies:")
print(" Holdout excluded: PASS")
print(" No model selection in A8.8: PASS")
print(" No policy threshold selection in A8.8: PASS")
print(" Scenario-level resampling specified: PASS")

print("\nOutput:")
print(OUTPUT_FILE)

print(
    "\nA8.8.1 EVALUATION PROTOCOL: "
    "FROZEN"
)