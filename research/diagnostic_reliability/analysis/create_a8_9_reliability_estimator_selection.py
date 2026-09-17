import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


BASE = Path(
    "research/diagnostic_reliability/analysis/outputs"
)

A8_8_DIR = (
    BASE
    / "a8_8_evaluation_and_calibration"
)

A8_9_DIR = (
    BASE
    / "a8_9_reliability_estimator_selection"
)

A8_9_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

METRICS_FILE = (
    A8_8_DIR
    / "a8_8_oof_model_performance_metrics.csv"
)

BOOTSTRAP_FILE = (
    A8_8_DIR
    / "a8_8_scenario_bootstrap_paired_model_differences.csv"
)

CALIBRATION_FILE = (
    A8_8_DIR
    / "a8_8_detailed_calibration_assessment.csv"
)

OUTPUT_FILE = (
    A8_9_DIR
    / "a8_9_reliability_estimator_selection.json"
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


for path in [
    METRICS_FILE,
    BOOTSTRAP_FILE,
    CALIBRATION_FILE,
]:
    if not path.exists():
        raise FileNotFoundError(path)


selection = {
    "stage": "A8.9",
    "substage": "A8.9.1",
    "name": "Reliability Estimator Selection",
    "status": "FROZEN_BEFORE_FINAL_HOLDOUT_EVALUATION",

    "created_at_utc": datetime.now(
        timezone.utc
    ).isoformat(),

    "selected_candidate": {
        "model": "M_EU",

        "predictors": [
            "E",
            "mean_self_confidence",
            "diagnostic_entropy",
        ],

        "estimator": (
            "Firth logistic regression "
            "with FLIC intercept correction"
        ),

        "target": "Y_D",

        "output": (
            "Estimated probability of "
            "diagnostic correctness"
        ),
    },

    "primary_benchmark": {
        "model": "M_E",

        "predictors": [
            "E",
        ],

        "purpose": (
            "Determine whether downstream "
            "uncertainty contributes useful "
            "reliability information beyond "
            "upstream evidence readiness."
        ),
    },

    "selection_criteria": [
        "probabilistic_accuracy",
        "calibration",
        "discrimination",
        "scenario_level_stability",
        "parsimony",
        "research_question_alignment",
    ],

    "development_evidence": {
        "M_EU": {
            "brier_score": 0.048415,
            "auroc": 0.967991,
            "auprc": 0.781659,
            "ece": 0.007563,
            "calibration_slope": 0.981167,
        },

        "M_E": {
            "brier_score": 0.050774,
            "auroc": 0.948129,
            "auprc": 0.680979,
            "ece": 0.017183,
            "calibration_slope": 1.001434,
        },

        "paired_M_EU_minus_M_E": {
            "brier_difference": -0.002359,
            "brier_ci_95": [
                -0.011065,
                0.006915,
            ],

            "auroc_difference": 0.019862,
            "auroc_ci_95": [
                0.000119,
                0.041197,
            ],

            "auprc_difference": 0.100680,
            "auprc_ci_95": [
                -0.016649,
                0.224392,
            ],
        },
    },

    "selection_rationale": (
        "M_EU is carried forward because it "
        "preserves the strong upstream "
        "evidence-readiness signal, incorporates "
        "the downstream uncertainty signals "
        "required by the research question, "
        "and provides the strongest overall "
        "development-stage point estimates across "
        "probabilistic accuracy, calibration and "
        "discrimination. The incremental advantage "
        "over M_E remains uncertain for Brier score "
        "and AUPRC under scenario-level bootstrap, "
        "so superiority is not treated as established."
    ),

    "interpretation_constraints": [
        (
            "Selection is based only on frozen "
            "development out-of-fold evidence."
        ),
        (
            "The final holdout has not been used "
            "to make this selection."
        ),
        (
            "M_EU is not claimed to be conclusively "
            "superior to M_E."
        ),
        (
            "M_E remains the primary benchmark "
            "for evaluating incremental downstream "
            "uncertainty value."
        ),
        (
            "No reliability policy threshold is "
            "selected at this stage."
        ),
    ],

    "evidence_files": {
        "oof_metrics": {
            "path": str(METRICS_FILE),
            "sha256": sha256(METRICS_FILE),
        },

        "scenario_bootstrap": {
            "path": str(BOOTSTRAP_FILE),
            "sha256": sha256(BOOTSTRAP_FILE),
        },

        "detailed_calibration": {
            "path": str(CALIBRATION_FILE),
            "sha256": sha256(CALIBRATION_FILE),
        },
    },

    "holdout_policy": {
        "status": "UNTOUCHED",
        "used_for_selection": False,
    },

    "next_stage": "A8.10 Post-inference Reliability Policy",
}


with OUTPUT_FILE.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        selection,
        f,
        indent=2,
    )


print("=" * 78)
print("A8.9 RELIABILITY ESTIMATOR SELECTION")
print("=" * 78)

print("\nSelected candidate:")
print("  M_EU")

print("\nPredictors:")
print("  E")
print("  mean_self_confidence")
print("  diagnostic_entropy")

print("\nEstimator:")
print(
    "  Firth logistic regression "
    "with FLIC intercept correction"
)

print("\nPrimary benchmark:")
print("  M_E")

print("\nSelection interpretation:")
print(
    "  M_EU carried forward based on the "
    "combined development evidence."
)

print(
    "  Incremental superiority over M_E "
    "is NOT treated as established."
)

print(
    "  M_E retained as primary benchmark."
)

print("\nHoldout:")
print("  Used for selection: NO")
print("  Status: UNTOUCHED")

print("\nPolicy thresholds selected:")
print("  NO")

print("\nEvidence provenance:")
for name, evidence in selection[
    "evidence_files"
].items():

    print(
        f"  {name}: "
        f"{evidence['sha256']}"
    )

print("\nOutput:")
print(OUTPUT_FILE)

print(
    "\nA8.9 RELIABILITY ESTIMATOR "
    "SELECTION: FROZEN"
)