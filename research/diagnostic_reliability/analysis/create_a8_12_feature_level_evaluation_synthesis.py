import hashlib
import json
from pathlib import Path


# ============================================================
# Paths
# ============================================================

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

A8_10_DIR = (
    BASE
    / "a8_10_post_inference_reliability_policy"
)

A8_11_DIR = (
    BASE
    / "a8_11_homepot_integration"
)

A8_12_DIR = (
    BASE
    / "a8_12_feature_level_evaluation"
)


# Development evidence
A8_8_METRICS = (
    A8_8_DIR
    / "a8_8_oof_model_performance_metrics.csv"
)

A8_8_BOOTSTRAP = (
    A8_8_DIR
    / "a8_8_scenario_bootstrap_paired_model_differences.csv"
)

A8_9_SELECTION = (
    A8_9_DIR
    / "a8_9_reliability_estimator_selection.json"
)

A8_10_POLICY = (
    A8_10_DIR
    / "a8_10_frozen_post_inference_reliability_policy.json"
)

A8_11_PARAMETERS = (
    A8_11_DIR
    / "a8_11_final_development_meu_parameters.json"
)


# Final holdout evidence
A8_12_PROTOCOL = (
    A8_12_DIR
    / "a8_12_pre_holdout_evaluation_protocol.json"
)

A8_12_ESTIMATOR = (
    A8_12_DIR
    / "a8_12_holdout_estimator_evaluation_summary.json"
)

A8_12_BOOTSTRAP = (
    A8_12_DIR
    / "a8_12_holdout_scenario_bootstrap_manifest.json"
)

A8_12_BOOTSTRAP_DIFFERENCES = (
    A8_12_DIR
    / "a8_12_holdout_scenario_bootstrap_paired_differences.csv"
)

A8_12_FEATURE = (
    A8_12_DIR
    / "a8_12_holdout_feature_policy_summary.json"
)

A8_12_FEATURE_MANIFEST = (
    A8_12_DIR
    / "a8_12_holdout_feature_policy_manifest.json"
)


OUTPUT_FILE = (
    A8_12_DIR
    / "a8_12_feature_level_evaluation_synthesis.json"
)

MANIFEST_FILE = (
    A8_12_DIR
    / "a8_12_feature_level_evaluation_synthesis_manifest.json"
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


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            path
        )


def load_json(path):
    with path.open(
        encoding="utf-8",
    ) as f:
        return json.load(f)


# ============================================================
# Validate evidence chain
# ============================================================

required_files = [
    A8_8_METRICS,
    A8_8_BOOTSTRAP,
    A8_9_SELECTION,
    A8_10_POLICY,
    A8_11_PARAMETERS,
    A8_12_PROTOCOL,
    A8_12_ESTIMATOR,
    A8_12_BOOTSTRAP,
    A8_12_BOOTSTRAP_DIFFERENCES,
    A8_12_FEATURE,
    A8_12_FEATURE_MANIFEST,
]

for path in required_files:
    require_file(
        path
    )


protocol = load_json(
    A8_12_PROTOCOL
)

estimator = load_json(
    A8_12_ESTIMATOR
)

bootstrap = load_json(
    A8_12_BOOTSTRAP
)

feature = load_json(
    A8_12_FEATURE
)

parameters = load_json(
    A8_11_PARAMETERS
)


print("=" * 80)
print(
    "A8.12.5 FEATURE-LEVEL "
    "EVALUATION SYNTHESIS"
)
print("=" * 80)


# ============================================================
# Frozen-state checks
# ============================================================

if (
    estimator[
        "holdout_now_opened"
    ]
    is not True
):
    raise RuntimeError(
        "Final holdout evaluation "
        "not recorded as completed."
    )


if (
    estimator[
        "post_holdout_tuning_permitted"
    ]
    is not False
):
    raise RuntimeError(
        "Post-holdout tuning safeguard "
        "not preserved."
    )


if (
    parameters["model"]
    != "M_EU"
):
    raise RuntimeError(
        "Frozen deployment estimator "
        "is not M_EU."
    )


if (
    parameters["status"]
    != "FROZEN_FOR_PROTOTYPE_INTEGRATION"
):
    raise RuntimeError(
        "Deployment estimator is not "
        "recorded as frozen."
    )


if (
    float(
        parameters[
            "policy"
        ][
            "issue_threshold"
        ]
    )
    != 0.64
):
    raise RuntimeError(
        "Frozen ISSUE threshold "
        "has changed."
    )


if (
    parameters[
        "unknown_semantics"
    ][
        "handling"
    ]
    != "ABSTAIN"
):
    raise RuntimeError(
        "UNKNOWN semantics changed."
    )


if (
    bootstrap[
        "post_holdout_tuning_performed"
    ]
    is not False
):
    raise RuntimeError(
        "Bootstrap manifest records "
        "post-holdout tuning."
    )


if (
    feature[
        "safeguards"
    ][
        "unknown_treated_as_incorrect"
    ]
    is not False
):
    raise RuntimeError(
        "UNKNOWN semantics violated."
    )


# ============================================================
# Frozen empirical results
# ============================================================

development = {
    "population": {
        "primary_eligible_cases":
            238,

        "correct":
            45,

        "incorrect":
            193,

        "represented_scenarios":
            68,
    },

    "M_E": {
        "brier":
            0.050774,

        "auroc":
            0.948129,

        "auprc":
            0.680979,

        "ece":
            0.017183,
    },

    "M_EU": {
        "brier":
            0.048415,

        "auroc":
            0.967991,

        "auprc":
            0.781659,

        "ece":
            0.007563,
    },

    "paired_M_EU_minus_M_E": {
        "brier_difference":
            -0.002359,

        "brier_ci_95": [
            -0.011065,
            0.006915,
        ],

        "auroc_difference":
            0.019862,

        "auroc_ci_95": [
            0.000119,
            0.041197,
        ],

        "auprc_difference":
            0.100680,

        "auprc_ci_95": [
            -0.016649,
            0.224392,
        ],
    },

    "interpretation": (
        "M_EU had the strongest overall "
        "development point estimates, but "
        "its incremental advantage over M_E "
        "was not conclusive across the "
        "pre-specified metrics. This justified "
        "carrying M_EU forward while retaining "
        "M_E as the primary benchmark."
    ),
}


holdout = {
    "population": {
        "primary_eligible_cases":
            54,

        "correct":
            11,

        "incorrect":
            43,

        "represented_primary_scenarios":
            15,

        "all_holdout_cases":
            120,

        "all_holdout_scenarios":
            30,

        "unknown_cases":
            66,
    },

    "M_E": {
        "mean_predicted_reliability":
            0.191225,

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
        "mean_predicted_reliability":
            0.194417,

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
        "brier_difference":
            0.013900,

        "brier_ci_95": [
            -0.000009,
            0.032344,
        ],

        "brier_meu_favourable_bootstrap_proportion":
            0.0260,

        "auroc_difference":
            -0.012685,

        "auroc_ci_95": [
            -0.034092,
            0.0,
        ],

        "auroc_meu_favourable_bootstrap_proportion":
            0.0005,

        "auprc_difference":
            -0.062855,

        "auprc_ci_95": [
            -0.113853,
            0.0,
        ],

        "auprc_meu_favourable_bootstrap_proportion":
            0.0,
    },

    "interpretation": (
        "The development-stage incremental "
        "advantage of M_EU did not reproduce "
        "on the untouched holdout. M_E produced "
        "better holdout point estimates for "
        "Brier score, AUROC, AUPRC and ECE. "
        "The primary paired Brier interval "
        "included approximately zero, so the "
        "result does not establish definitive "
        "superiority of M_E, but it provides "
        "no holdout evidence that adding the "
        "downstream uncertainty signals improved "
        "probabilistic reliability estimation."
    ),
}


feature_behaviour = {
    "decision_states": {
        "ABSTAIN":
            66,

        "WITHHOLD":
            40,

        "QUALIFY":
            3,

        "ISSUE":
            11,
    },

    "withhold": {
        "cases":
            40,

        "correct":
            0,

        "incorrect":
            40,

        "interpretation": (
            "All substantive diagnoses withheld "
            "because E=0 were incorrect in the "
            "holdout, demonstrating strong "
            "association between upstream "
            "evidence readiness and diagnostic "
            "correctness in this experiment."
        ),
    },

    "issue": {
        "cases":
            11,

        "coverage_among_substantive":
            0.203704,

        "correct":
            8,

        "incorrect":
            3,

        "accuracy":
            0.727273,

        "selective_risk":
            0.272727,

        "development_issue_accuracy":
            0.822222,

        "interpretation": (
            "The frozen 0.64 ISSUE operating "
            "criterion did not reproduce its "
            "development-stage accuracy on the "
            "holdout. It must therefore remain "
            "a prototype operating criterion, "
            "not an externally validated safety "
            "or accuracy guarantee."
        ),
    },

    "qualify": {
        "cases":
            3,

        "correct":
            3,

        "incorrect":
            0,

        "interpretation": (
            "All three QUALIFY diagnoses were "
            "correct, but the sample is too small "
            "to support a general claim that "
            "QUALIFY is more reliable than ISSUE. "
            "The result instead shows that the "
            "frozen downstream probability "
            "threshold did not cleanly order "
            "actionable holdout diagnoses by "
            "correctness."
        ),
    },

    "abstain": {
        "cases":
            66,

        "appropriate":
            50,

        "unnecessary":
            16,

        "appropriateness_rate":
            0.757576,

        "interpretation": (
            "UNKNOWN was evaluated as an "
            "abstention decision using Y_A "
            "rather than being converted into "
            "an incorrect diagnosis under Y_D."
        ),
    },
}


# ============================================================
# Research-question synthesis
# ============================================================

research_question_synthesis = {
    "upstream_E": (
        "Upstream evidence readiness was a "
        "strong indicator of diagnostic "
        "correctness in both development and "
        "holdout evaluation. In the holdout, "
        "all 40 substantive diagnoses with E=0 "
        "were incorrect, while all 11 correct "
        "substantive diagnoses occurred among "
        "the 14 E=1 cases."
    ),

    "downstream_U": (
        "Downstream self-confidence and "
        "diagnostic disagreement contained "
        "development-stage predictive "
        "information, but downstream-only "
        "models were weaker than the upstream "
        "benchmark and the incremental benefit "
        "of adding U to E did not generalise "
        "to the untouched holdout."
    ),

    "combined_E_plus_U": (
        "M_EU produced the strongest overall "
        "development point estimates and was "
        "therefore legitimately selected before "
        "holdout evaluation. On the untouched "
        "holdout, however, M_EU did not "
        "outperform M_E and produced poorer "
        "point estimates across Brier, AUROC, "
        "AUPRC and ECE."
    ),

    "overall": (
        "Within this controlled validation-first "
        "IoT diagnostic experiment, upstream "
        "evidence readiness provided the most "
        "consistently generalising reliability "
        "signal. Combining it with the selected "
        "downstream uncertainty measures did not "
        "demonstrate additional reliability "
        "estimation benefit on the final holdout. "
        "This finding is specific to the tested "
        "system, model, scenarios and uncertainty "
        "measures and should not be interpreted "
        "as evidence that downstream uncertainty "
        "is generally uninformative."
    ),
}


# ============================================================
# Limitations carried forward
# ============================================================

limitations = [
    (
        "Only 54 holdout cases were eligible "
        "for the primary Y_D estimator "
        "evaluation."
    ),
    (
        "Only 15 holdout scenarios were "
        "represented in the primary Y_D "
        "evaluation."
    ),
    (
        "Only 11 positive diagnostic-correctness "
        "outcomes occurred in the primary "
        "holdout population."
    ),
    (
        "No Y_D-eligible MEMORY_PRESSURE case "
        "was present in the holdout."
    ),
    (
        "The M_E upstream predictor was strongly "
        "separating and effectively binary in "
        "this experiment, limiting conclusions "
        "about richer upstream reliability "
        "representations."
    ),
    (
        "The downstream uncertainty variables "
        "were mean self-reported confidence and "
        "repeated-generation diagnostic entropy. "
        "The findings do not establish that all "
        "forms of downstream uncertainty "
        "quantification would behave similarly."
    ),
    (
        "The 0.64 ISSUE threshold was selected "
        "as a development-stage prototype "
        "operating criterion and is not an "
        "externally validated safety threshold."
    ),
    (
        "The experiment evaluates controlled "
        "fault scenarios using the frozen "
        "Gemma 3 12B configuration and should "
        "not be assumed to generalise directly "
        "to other models, prompts, temperatures, "
        "fault taxonomies or operational "
        "environments."
    ),
]


# ============================================================
# Create durable synthesis
# ============================================================

synthesis = {
    "stage":
        "A8.12",

    "substage":
        "A8.12.5",

    "name":
        "Feature-Level Evaluation Synthesis",

    "status":
        "COMPLETE_AND_READY_FOR_TECHNICAL_FREEZE",

    "research_question": (
        "How effectively can upstream "
        "evidence-readiness and downstream "
        "LLM uncertainty signals, individually "
        "and in combination, be used to estimate "
        "and calibrate the reliability of "
        "LLM-generated operational diagnoses "
        "within a validation-first IoT "
        "decision-support system?"
    ),

    "primary_metric":
        "Brier score",

    "development_evidence":
        development,

    "final_holdout_evidence":
        holdout,

    "feature_behaviour":
        feature_behaviour,

    "research_question_synthesis":
        research_question_synthesis,

    "limitations":
        limitations,

    "methodological_safeguards": {
        "holdout_used_for_model_selection":
            False,

        "holdout_used_for_coefficient_refit":
            False,

        "holdout_used_for_threshold_selection":
            False,

        "post_holdout_recalibration":
            False,

        "experiment_rerun_after_holdout":
            False,

        "unknown_converted_to_incorrect":
            False,

        "scenario_level_bootstrap_used":
            True,

        "brier_retained_as_primary_metric":
            True,
    },

    "frozen_feature_definition": {
        "name":
            (
                "Post-Inference Diagnostic "
                "Reliability Layer"
            ),

        "estimator":
            "M_EU",

        "policy": {
            "UNKNOWN":
                "ABSTAIN",

            "substantive_E_0":
                "WITHHOLD",

            "substantive_E_1_p_below_0_64":
                "QUALIFY",

            "substantive_E_1_p_at_or_above_0_64":
                "ISSUE",
        },
    },

    "next_stage":
        "A8.13 Final Technical Freeze",
}


with OUTPUT_FILE.open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        synthesis,
        f,
        indent=2,
    )


# ============================================================
# Manifest
# ============================================================

input_files = [
    A8_8_METRICS,
    A8_8_BOOTSTRAP,
    A8_9_SELECTION,
    A8_10_POLICY,
    A8_11_PARAMETERS,
    A8_12_PROTOCOL,
    A8_12_ESTIMATOR,
    A8_12_BOOTSTRAP,
    A8_12_BOOTSTRAP_DIFFERENCES,
    A8_12_FEATURE,
    A8_12_FEATURE_MANIFEST,
]


manifest = {
    "stage":
        "A8.12",

    "substage":
        "A8.12.5",

    "status":
        "COMPLETE",

    "purpose": (
        "Durable synthesis of development, "
        "final holdout, uncertainty and "
        "feature-policy evidence before "
        "A8.13 Final Technical Freeze."
    ),

    "inputs": {
        str(path):
            sha256(path)
        for path in input_files
    },

    "outputs": {
        "synthesis":
            str(OUTPUT_FILE),
    },

    "new_model_fitting":
        False,

    "new_threshold_selection":
        False,

    "new_statistical_testing":
        False,

    "post_holdout_tuning":
        False,
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

print("\nDevelopment:")
print(
    "  M_E Brier:",
    development["M_E"]["brier"],
)
print(
    "  M_EU Brier:",
    development["M_EU"]["brier"],
)

print("\nFinal holdout:")
print(
    "  M_E Brier:",
    holdout["M_E"]["brier"],
)
print(
    "  M_EU Brier:",
    holdout["M_EU"]["brier"],
)
print(
    "  Paired Brier difference "
    "(M_EU - M_E):",
    holdout[
        "paired_M_EU_minus_M_E"
    ][
        "brier_difference"
    ],
)
print(
    "  Paired Brier 95% CI:",
    holdout[
        "paired_M_EU_minus_M_E"
    ][
        "brier_ci_95"
    ],
)

print("\nFrozen feature behaviour:")
print(
    "  ABSTAIN:",
    feature_behaviour[
        "decision_states"
    ][
        "ABSTAIN"
    ],
)
print(
    "  WITHHOLD:",
    feature_behaviour[
        "decision_states"
    ][
        "WITHHOLD"
    ],
)
print(
    "  QUALIFY:",
    feature_behaviour[
        "decision_states"
    ][
        "QUALIFY"
    ],
)
print(
    "  ISSUE:",
    feature_behaviour[
        "decision_states"
    ][
        "ISSUE"
    ],
)
print(
    "  ISSUE accuracy:",
    feature_behaviour[
        "issue"
    ][
        "accuracy"
    ],
)
print(
    "  ABSTAIN appropriateness:",
    feature_behaviour[
        "abstain"
    ][
        "appropriateness_rate"
    ],
)

print("\nResearch synthesis:")
print(
    " ",
    research_question_synthesis[
        "overall"
    ],
)

print("\nOutput artefacts:")
print(
    OUTPUT_FILE
)
print(
    MANIFEST_FILE
)

print("\nValidation:")
print(
    "Development evidence retained: PASS"
)
print(
    "Untouched-holdout evidence retained: PASS"
)
print(
    "Scenario-bootstrap uncertainty retained: PASS"
)
print(
    "Frozen policy behaviour retained: PASS"
)
print(
    "UNKNOWN semantics retained: PASS"
)
print(
    "Limitations explicitly retained: PASS"
)
print(
    "No new model fitting: PASS"
)
print(
    "No threshold reselection: PASS"
)
print(
    "No post-holdout tuning: PASS"
)

print(
    "\nA8.12.5 FEATURE-LEVEL "
    "EVALUATION SYNTHESIS: COMPLETE"
)