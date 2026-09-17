import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


# ============================================================
# Paths
# ============================================================

ROOT = Path(".")

ANALYSIS = Path(
    "research/diagnostic_reliability/analysis"
)

OUTPUTS = ANALYSIS / "outputs"

A8_6 = (
    OUTPUTS
    / "a8_6_outcome_uncertainty_construction"
)

A8_7 = (
    OUTPUTS
    / "a8_7_reliability_modelling"
)

A8_8 = (
    OUTPUTS
    / "a8_8_evaluation_and_calibration"
)

A8_9 = (
    OUTPUTS
    / "a8_9_reliability_estimator_selection"
)

A8_10 = (
    OUTPUTS
    / "a8_10_post_inference_reliability_policy"
)

A8_11 = (
    OUTPUTS
    / "a8_11_homepot_integration"
)

A8_12 = (
    OUTPUTS
    / "a8_12_feature_level_evaluation"
)

A8_13 = (
    OUTPUTS
    / "a8_13_final_technical_freeze"
)

A8_13.mkdir(
    parents=True,
    exist_ok=True,
)


FREEZE_FILE = (
    A8_13
    / "a8_13_final_technical_freeze.json"
)

MANIFEST_FILE = (
    A8_13
    / "a8_13_final_technical_freeze_manifest.json"
)


# ============================================================
# Frozen evidence inputs
# ============================================================

FILES = {
    "a8_6_dataset":
        A8_6
        / "a8_6_evidence_case_reliability_dataset.csv",

    "a8_6_manifest":
        A8_6
        / "a8_6_outcome_uncertainty_construction_manifest.json",

    "a8_7_partition":
        A8_7
        / "a8_7_scenario_development_holdout_assignment.csv",

    "a8_7_partition_manifest":
        A8_7
        / "a8_7_scenario_partition_manifest.json",

    "a8_7_oof_predictions":
        A8_7
        / "a8_7_grouped_cv_out_of_fold_reliability_predictions.csv",

    "a8_8_protocol":
        A8_8
        / "a8_8_evaluation_protocol.json",

    "a8_8_metrics":
        A8_8
        / "a8_8_oof_model_performance_metrics.csv",

    "a8_8_bootstrap":
        A8_8
        / "a8_8_scenario_bootstrap_paired_model_differences.csv",

    "a8_8_calibration":
        A8_8
        / "a8_8_detailed_calibration_assessment.json",

    "a8_9_selection":
        A8_9
        / "a8_9_reliability_estimator_selection.json",

    "a8_10_policy":
        A8_10
        / "a8_10_frozen_post_inference_reliability_policy.json",

    "a8_11_parameters":
        A8_11
        / "a8_11_final_development_meu_parameters.json",

    "a8_12_protocol":
        A8_12
        / "a8_12_pre_holdout_evaluation_protocol.json",

    "a8_12_estimator_summary":
        A8_12
        / "a8_12_holdout_estimator_evaluation_summary.json",

    "a8_12_bootstrap":
        A8_12
        / "a8_12_holdout_scenario_bootstrap_paired_differences.csv",

    "a8_12_feature":
        A8_12
        / "a8_12_holdout_feature_policy_summary.json",

    "a8_12_synthesis":
        A8_12
        / "a8_12_feature_level_evaluation_synthesis.json",
}


ANALYSIS_SCRIPTS = [
    ANALYSIS
    / "construct_a8_6_reliability_analysis_dataset.py",

    ANALYSIS
    / "create_a8_7_scenario_partition.py",

    ANALYSIS
    / "create_a8_7_grouped_cv_folds.py",

    ANALYSIS
    / "check_a8_7_partition_compatibility.py",

    ANALYSIS
    / "check_a8_7_predictor_structure_and_separation.py",

    ANALYSIS
    / "check_a8_7_grouped_cv_analytical_compatibility.py",

    ANALYSIS
    / "fit_a8_7_grouped_cv_reliability_models.py",

    ANALYSIS
    / "check_a8_7_grouped_cv_model_integrity.py",

    ANALYSIS
    / "create_a8_8_evaluation_protocol.py",

    ANALYSIS
    / "evaluate_a8_8_oof_reliability_models.py",

    ANALYSIS
    / "bootstrap_a8_8_scenario_level_model_evaluation.py",

    ANALYSIS
    / "assess_a8_8_detailed_calibration.py",

    ANALYSIS
    / "create_a8_9_reliability_estimator_selection.py",

    ANALYSIS
    / "analyse_a8_10_selective_reliability_policy.py",

    ANALYSIS
    / "bootstrap_a8_10_policy_operating_points.py",

    ANALYSIS
    / "create_a8_10_reliability_policy.py",

    ANALYSIS
    / "fit_a8_11_final_development_reliability_estimator.py",

    ANALYSIS
    / "prepare_a8_12_pre_holdout_evaluation.py",

    ANALYSIS
    / "evaluate_a8_12_holdout_estimators.py",

    ANALYSIS
    / "bootstrap_a8_12_holdout_estimator_evaluation.py",

    ANALYSIS
    / "evaluate_a8_12_holdout_feature_policy.py",

    ANALYSIS
    / "create_a8_12_feature_level_evaluation_synthesis.py",
]


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


def load_json(path):
    with path.open(
        encoding="utf-8",
    ) as f:
        return json.load(f)


def git_output(*args):
    result = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    )

    return result.stdout.strip()


# ============================================================
# Validate evidence files
# ============================================================

print("=" * 80)
print("A8.13.2 FINAL TECHNICAL FREEZE")
print("=" * 80)


for name, path in FILES.items():
    if not path.exists():
        raise FileNotFoundError(
            f"{name}: {path}"
        )


for path in ANALYSIS_SCRIPTS:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing analysis script: {path}"
        )


# ============================================================
# Load authoritative frozen state
# ============================================================

parameters = load_json(
    FILES["a8_11_parameters"]
)

holdout = load_json(
    FILES["a8_12_estimator_summary"]
)

feature = load_json(
    FILES["a8_12_feature"]
)

synthesis = load_json(
    FILES["a8_12_synthesis"]
)


# ============================================================
# Integrity assertions
# ============================================================

assert parameters["model"] == "M_EU"

assert (
    parameters["status"]
    == "FROZEN_FOR_PROTOTYPE_INTEGRATION"
)

assert (
    parameters["estimator"]
    == "Firth logistic regression + FLIC"
)

assert (
    float(
        parameters[
            "policy"
        ][
            "issue_threshold"
        ]
    )
    == 0.64
)

assert (
    parameters[
        "unknown_semantics"
    ][
        "handling"
    ]
    == "ABSTAIN"
)

assert (
    holdout[
        "holdout_now_opened"
    ]
    is True
)

assert (
    holdout[
        "post_holdout_tuning_permitted"
    ]
    is False
)

assert (
    feature[
        "population"
    ][
        "holdout_cases"
    ]
    == 120
)

assert (
    feature[
        "population"
    ][
        "substantive_cases"
    ]
    == 54
)

assert (
    feature[
        "population"
    ][
        "unknown_cases"
    ]
    == 66
)

assert (
    synthesis["primary_metric"]
    == "Brier score"
)

assert (
    synthesis["status"]
    == "COMPLETE_AND_READY_FOR_TECHNICAL_FREEZE"
)


# ============================================================
# Git state
# ============================================================

git_commit = git_output(
    "rev-parse",
    "HEAD",
)

git_branch = git_output(
    "branch",
    "--show-current",
)


# ============================================================
# Final frozen technical state
# ============================================================

predictors = {
    item["name"]:
        item["coefficient"]
    for item
    in parameters["predictors"]
}


freeze = {
    "stage":
        "A8.13",

    "substage":
        "A8.13.2",

    "name":
        "Final Technical Freeze",

    "status":
        "FROZEN_PENDING_FINAL_REGRESSION_AND_COMMIT",

    "freeze_created_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "repository": {
        "branch":
            git_branch,

        "pre_freeze_commit":
            git_commit,
    },

    "research_question":
        synthesis[
            "research_question"
        ],

    "experimental_design": {
        "total_scenarios":
            150,

        "total_evidence_cases":
            600,

        "generations_per_case":
            7,

        "total_generations":
            4200,

        "development_scenarios":
            120,

        "holdout_scenarios":
            30,

        "primary_holdout_eligible_cases":
            54,

        "primary_holdout_correct":
            11,

        "primary_holdout_incorrect":
            43,

        "primary_holdout_represented_scenarios":
            15,

        "all_holdout_cases":
            120,

        "holdout_unknown_cases":
            66,
    },

    "measurement_definition": {
        "upstream_signal_E":
            (
                "Binary evidence readiness "
                "derived from EnvelopeResult."
                "is_actionable"
            ),

        "downstream_uncertainty":
            [
                "mean_self_confidence",
                "diagnostic_entropy",
            ],

        "primary_outcome":
            "diagnostic_correctness (Y_D)",

        "secondary_outcome":
            "decision_appropriateness (Y_A)",

        "unknown_semantics":
            (
                "UNKNOWN is an abstention "
                "decision state and is not "
                "converted to Y_D=0."
            ),
    },

    "selected_prototype_estimator": {
        "model":
            "M_EU",

        "estimator":
            "Firth logistic regression + FLIC",

        "flic_intercept":
            parameters[
                "flic_intercept"
            ],

        "coefficients":
            predictors,

        "equation":
            parameters[
                "prediction_equation"
            ],
    },

    "frozen_policy": {
        "UNKNOWN":
            "ABSTAIN",

        "E_equals_0":
            "WITHHOLD",

        "E_equals_1_below_0_64":
            "QUALIFY",

        "E_equals_1_at_or_above_0_64":
            "ISSUE",

        "issue_threshold":
            0.64,
    },

    "development_result": {
        "M_E_brier":
            0.050774,

        "M_EU_brier":
            0.048415,

        "M_EU_minus_M_E_brier":
            -0.002359,

        "paired_brier_ci_95": [
            -0.011065,
            0.006915,
        ],

        "interpretation":
            (
                "M_EU showed the strongest "
                "development point estimates, "
                "but its incremental advantage "
                "over M_E was not conclusive "
                "across the pre-specified "
                "evaluation metrics."
            ),
    },

    "final_holdout_result": {
        "M_E": {
            "brier":
                0.044469,

            "auroc":
                0.965116,

            "auprc":
                0.785714,

            "ece":
                0.026849,
        },

        "M_EU": {
            "brier":
                0.058370,

            "auroc":
                0.952431,

            "auprc":
                0.722859,

            "ece":
                0.056679,
        },

        "paired_M_EU_minus_M_E": {
            "brier":
                0.013900,

            "brier_ci_95": [
                -0.000009,
                0.032344,
            ],

            "auroc":
                -0.012685,

            "auprc":
                -0.062855,
        },
    },

    "feature_level_result": {
        "ABSTAIN":
            66,

        "WITHHOLD":
            40,

        "QUALIFY":
            3,

        "ISSUE":
            11,

        "issue_accuracy":
            0.727273,

        "issue_selective_risk":
            0.272727,

        "abstain_appropriateness":
            0.757576,
    },

    "final_empirical_synthesis":
        synthesis[
            "research_question_synthesis"
        ][
            "overall"
        ],

    "limitations":
        synthesis[
            "limitations"
        ],

    "methodological_safeguards":
        synthesis[
            "methodological_safeguards"
        ],

    "technical_freeze_rules": [
        (
            "Do not rerun the main experiment "
            "in response to final results."
        ),
        (
            "Do not refit or reselect the "
            "reliability estimator using the "
            "holdout."
        ),
        (
            "Do not change the 0.64 policy "
            "threshold using holdout results."
        ),
        (
            "Do not post-hoc recalibrate using "
            "the holdout."
        ),
        (
            "Do not change the frozen prompt, "
            "temperature, taxonomy or generation "
            "configuration in response to "
            "holdout performance."
        ),
        (
            "Preserve UNKNOWN as an abstention "
            "state rather than diagnostic "
            "incorrectness."
        ),
        (
            "Future dissertation analysis must "
            "report the final holdout result "
            "without retrospective optimisation."
        ),
    ],

    "next_stage":
        "A9 Results",
}


# ============================================================
# Hash evidence and scripts
# ============================================================

evidence_hashes = {
    name:
        sha256(path)
    for name, path
    in FILES.items()
}

script_hashes = {
    str(path):
        sha256(path)
    for path
    in ANALYSIS_SCRIPTS
}


freeze[
    "evidence_hashes"
] = evidence_hashes

freeze[
    "analysis_script_hashes"
] = script_hashes


with FREEZE_FILE.open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        freeze,
        f,
        indent=2,
    )


manifest = {
    "stage":
        "A8.13",

    "substage":
        "A8.13.2",

    "status":
        "FREEZE_RECORD_CREATED",

    "freeze_file":
        str(FREEZE_FILE),

    "freeze_file_sha256":
        sha256(
            FREEZE_FILE
        ),

    "evidence_files_hashed":
        len(
            evidence_hashes
        ),

    "analysis_scripts_hashed":
        len(
            script_hashes
        ),

    "pre_freeze_git_commit":
        git_commit,

    "branch":
        git_branch,

    "final_regression_pending":
        True,

    "final_commit_pending":
        True,
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

print("\nRepository:")
print(
    "  Branch:",
    git_branch,
)
print(
    "  Pre-freeze commit:",
    git_commit,
)

print("\nFrozen experiment:")
print(
    "  Scenarios: 150"
)
print(
    "  Evidence cases: 600"
)
print(
    "  Generations: 4200"
)
print(
    "  Development scenarios: 120"
)
print(
    "  Holdout scenarios: 30"
)

print("\nFrozen estimator:")
print(
    "  Model: M_EU"
)
print(
    "  Estimator: Firth logistic + FLIC"
)
print(
    "  FLIC intercept:",
    parameters[
        "flic_intercept"
    ],
)
print(
    "  beta_E:",
    predictors["E"],
)
print(
    "  beta_C:",
    predictors[
        "mean_self_confidence"
    ],
)
print(
    "  beta_H:",
    predictors[
        "diagnostic_entropy"
    ],
)

print("\nFrozen policy:")
print(
    "  UNKNOWN -> ABSTAIN"
)
print(
    "  E=0 -> WITHHOLD"
)
print(
    "  E=1, p<0.64 -> QUALIFY"
)
print(
    "  E=1, p>=0.64 -> ISSUE"
)

print("\nFinal holdout:")
print(
    "  M_E Brier: 0.044469"
)
print(
    "  M_EU Brier: 0.058370"
)
print(
    "  Delta Brier: +0.013900"
)
print(
    "  Delta Brier 95% CI: "
    "[-0.000009, 0.032344]"
)

print("\nFeature behaviour:")
print(
    "  ABSTAIN: 66"
)
print(
    "  WITHHOLD: 40"
)
print(
    "  QUALIFY: 3"
)
print(
    "  ISSUE: 11"
)
print(
    "  ISSUE accuracy: 0.727273"
)
print(
    "  ABSTAIN appropriateness: 0.757576"
)

print("\nFreeze artefacts:")
print(
    FREEZE_FILE
)
print(
    MANIFEST_FILE
)

print("\nValidation:")
print(
    "Evidence chain present: PASS"
)
print(
    "Analysis scripts present: PASS"
)
print(
    "Frozen estimator retained: PASS"
)
print(
    "Frozen policy retained: PASS"
)
print(
    "Final holdout retained: PASS"
)
print(
    "UNKNOWN semantics retained: PASS"
)
print(
    "No post-holdout optimisation recorded: PASS"
)

print(
    "\nA8.13.2 FINAL TECHNICAL "
    "FREEZE RECORD: CREATED"
)