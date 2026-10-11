from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

RESEARCH_ROOT = Path("research/diagnostic_reliability")

ANALYSIS_OUTPUTS = RESEARCH_ROOT / "analysis" / "outputs"
DISSERTATION_OUTPUTS = RESEARCH_ROOT / "dissertation_outputs"

TABLES_DIR = DISSERTATION_OUTPUTS / "tables"
CONSOLIDATED_TABLES_DIR = TABLES_DIR / "consolidated"
FIGURES_DIR = DISSERTATION_OUTPUTS / "figures"
MANIFESTS_DIR = DISSERTATION_OUTPUTS / "manifests"

for directory in (
    TABLES_DIR,
    CONSOLIDATED_TABLES_DIR,
    FIGURES_DIR,
    MANIFESTS_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)


A86_DIR = ANALYSIS_OUTPUTS / "a8_6_outcome_uncertainty_construction"
A87_DIR = ANALYSIS_OUTPUTS / "a8_7_reliability_modelling"
A88_DIR = ANALYSIS_OUTPUTS / "a8_8_evaluation_and_calibration"
A89_DIR = ANALYSIS_OUTPUTS / "a8_9_reliability_estimator_selection"
A810_DIR = ANALYSIS_OUTPUTS / "a8_10_post_inference_reliability_policy"
A811_DIR = ANALYSIS_OUTPUTS / "a8_11_homepot_integration"
A812_DIR = ANALYSIS_OUTPUTS / "a8_12_feature_level_evaluation"

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required frozen artefact not found: {path}")
    return pd.read_csv(path)


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required frozen artefact not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_table(df: pd.DataFrame, filename: str) -> Path:
    path = TABLES_DIR / filename
    df.to_csv(path, index=False)
    print(f"[TABLE] {path}")
    return path

def save_consolidated_table(
    df: pd.DataFrame,
    filename: str,
) -> Path:
    path = CONSOLIDATED_TABLES_DIR / filename
    df.to_csv(path, index=False)
    print(f"[CONSOLIDATED TABLE] {path}")
    return path

def normalise_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )


# ---------------------------------------------------------------------
# Load frozen source artefacts
# ---------------------------------------------------------------------

cases = read_csv(
    A86_DIR / "a8_6_evidence_case_reliability_dataset.csv"
)

partition = read_csv(
    A87_DIR / "a8_7_scenario_development_holdout_assignment.csv"
)

cv_coefficients = read_csv(
    A87_DIR / "a8_7_grouped_cv_flic_model_coefficients.csv"
)

oof_predictions = read_csv(
    A87_DIR / "a8_7_grouped_cv_out_of_fold_reliability_predictions.csv"
)

development_metrics = read_csv(
    A88_DIR / "a8_8_oof_model_performance_metrics.csv"
)

development_calibration = read_csv(
    A88_DIR / "a8_8_detailed_calibration_assessment.csv"
)

development_bootstrap = read_csv(
    A88_DIR / "a8_8_scenario_bootstrap_metric_intervals.csv"
)

development_paired = read_csv(
    A88_DIR / "a8_8_scenario_bootstrap_paired_model_differences.csv"
)

policy_summary = read_json(
    A810_DIR / "a8_10_development_selective_risk_coverage_summary.json"
)

holdout_metrics = read_csv(
    A812_DIR / "a8_12_holdout_estimator_performance_metrics.csv"
)

holdout_bootstrap = read_csv(
    A812_DIR / "a8_12_holdout_scenario_bootstrap_metric_intervals.csv"
)

holdout_paired = read_csv(
    A812_DIR / "a8_12_holdout_scenario_bootstrap_paired_differences.csv"
)

holdout_policy_cases = read_csv(
    A812_DIR / "a8_12_holdout_feature_policy_cases.csv"
)

holdout_policy_states = read_csv(
    A812_DIR / "a8_12_holdout_feature_policy_state_summary.csv"
)


# ---------------------------------------------------------------------
# Defensive normalisation
# ---------------------------------------------------------------------

cases["actionable"] = normalise_bool(cases["actionable"])
cases["primary_correctness_eligible"] = normalise_bool(
    cases["primary_correctness_eligible"]
)


# =====================================================================
# TABLE 1
# Experimental dataset and primary outcome summary
# =====================================================================

n_scenarios = cases["scenario_id"].nunique()
n_cases = len(cases)

generations_per_case = 7
n_generations = n_cases * generations_per_case

eligible = cases["primary_correctness_eligible"].fillna(False)
unknown = cases["modal_diagnosis"].eq("UNKNOWN")

correct = cases["diagnostic_correctness"].eq(1)
incorrect = cases["diagnostic_correctness"].eq(0)

table_01 = pd.DataFrame(
    [
        ["Independent scenarios", n_scenarios],
        ["Evidence cases", n_cases],
        ["Generations per case", generations_per_case],
        ["Total generations", n_generations],
        ["Modal UNKNOWN cases", int(unknown.sum())],
        ["Primary Y_D eligible cases", int(eligible.sum())],
        ["Correct eligible diagnoses", int(correct.sum())],
        ["Incorrect eligible diagnoses", int(incorrect.sum())],
        [
            "Eligible diagnostic accuracy",
            float(correct.sum() / eligible.sum()),
        ],
    ],
    columns=["Measure", "Value"],
)

save_table(
    table_01,
    "table_01_experimental_dataset_and_primary_outcome_summary.csv",
)


# =====================================================================
# TABLE 2
# Modal diagnosis distribution
# =====================================================================

diagnosis_order = [
    "UNKNOWN",
    "NETWORK_DEGRADATION",
    "CPU_SATURATION",
    "NORMAL",
    "MEMORY_PRESSURE",
    "APPLICATION_FAILURE",
]

diagnosis_counts = (
    cases["modal_diagnosis"]
    .value_counts()
    .reindex(diagnosis_order, fill_value=0)
)

table_02 = diagnosis_counts.rename_axis("Modal diagnosis").reset_index(
    name="Cases"
)

save_table(
    table_02,
    "table_02_modal_diagnosis_distribution.csv",
)


# =====================================================================
# TABLE 3
# Decision appropriateness
# =====================================================================

table_03 = (
    cases["decision_appropriateness"]
    .value_counts()
    .reindex([1, 0], fill_value=0)
    .rename_axis("Y_A")
    .reset_index(name="Cases")
)

table_03["Interpretation"] = table_03["Y_A"].map(
    {1: "Appropriate", 0: "Inappropriate"}
)

table_03 = table_03[
    ["Y_A", "Interpretation", "Cases"]
]

save_table(
    table_03,
    "table_03_decision_appropriateness.csv",
)


# =====================================================================
# TABLE 4
# Error category distribution
# =====================================================================

error_order = [
    "APPROPRIATE_ABSTENTION",
    "UNSUPPORTED_DIAGNOSIS",
    "UNNECESSARY_ABSTENTION",
    "CORRECT",
    "WRONG_FAULT",
    "FALSE_NORMAL",
    "FALSE_FAULT",
]

error_counts = (
    cases["error_category"]
    .value_counts()
    .reindex(error_order, fill_value=0)
)

table_04 = error_counts.rename_axis("Error category").reset_index(
    name="Cases"
)

save_table(
    table_04,
    "table_04_error_category_distribution.csv",
)


# =====================================================================
# TABLE 5
# Primary outcome by evidence condition
# =====================================================================

q_order = [
    "COMPLETE_FRESH",
    "DISCONTINUOUS",
    "INCOMPLETE",
    "STALE",
]

eligible_cases = cases.loc[eligible].copy()

table_05 = (
    eligible_cases.groupby("evidence_condition", observed=False)
    .agg(
        Eligible_cases=("case_id", "size"),
        Correct=("diagnostic_correctness", lambda x: int((x == 1).sum())),
        Incorrect=("diagnostic_correctness", lambda x: int((x == 0).sum())),
    )
    .reindex(q_order, fill_value=0)
    .reset_index()
)

save_table(
    table_05,
    "table_05_primary_outcome_by_evidence_condition.csv",
)


# =====================================================================
# TABLE 6
# Primary outcome by fault class
# =====================================================================

fault_order = [
    "APPLICATION_FAILURE",
    "CPU_SATURATION",
    "MEMORY_PRESSURE",
    "NETWORK_DEGRADATION",
    "NORMAL",
]

table_06 = (
    eligible_cases.groupby("ground_truth", observed=False)
    .agg(
        Eligible_cases=("case_id", "size"),
        Correct=("diagnostic_correctness", lambda x: int((x == 1).sum())),
        Incorrect=("diagnostic_correctness", lambda x: int((x == 0).sum())),
    )
    .reindex(fault_order, fill_value=0)
    .reset_index()
    .rename(columns={"ground_truth": "Fault class"})
)

save_table(
    table_06,
    "table_06_primary_outcome_by_fault_class.csv",
)


# =====================================================================
# TABLE 7
# Eligible-case uncertainty summary
# =====================================================================

table_07 = pd.DataFrame(
    [
        [
            "Mean self-confidence",
            eligible_cases["mean_self_confidence"].mean(),
        ],
        [
            "Diagnostic entropy",
            eligible_cases["diagnostic_entropy"].mean(),
        ],
        [
            "Modal agreement",
            eligible_cases["modal_agreement"].mean(),
        ],
    ],
    columns=["Measure", "Mean"],
)

save_table(
    table_07,
    "table_07_eligible_case_uncertainty_summary.csv",
)


# =====================================================================
# TABLE 8
# Development / holdout partition
# =====================================================================

case_partition = cases.merge(
    partition[["scenario_id", "partition"]],
    on="scenario_id",
    how="left",
    validate="many_to_one",
)

if case_partition["partition"].isna().any():
    raise ValueError("Some evidence cases do not have a partition assignment.")

partition_rows = []

for partition_name in ["DEVELOPMENT", "HOLDOUT"]:
    subset = case_partition[
        case_partition["partition"].eq(partition_name)
    ]

    subset_eligible = subset[
        subset["primary_correctness_eligible"].fillna(False)
    ]

    partition_rows.append(
        {
            "Partition": partition_name.title(),
            "Scenarios": subset["scenario_id"].nunique(),
            "Cases": len(subset),
            "Y_D eligible": len(subset_eligible),
            "Correct": int(
                subset_eligible["diagnostic_correctness"].eq(1).sum()
            ),
            "Incorrect": int(
                subset_eligible["diagnostic_correctness"].eq(0).sum()
            ),
        }
    )

table_08 = pd.DataFrame(partition_rows)

save_table(
    table_08,
    "table_08_development_holdout_partition.csv",
)


# =====================================================================
# TABLE 9
# Development E x Y_D
# =====================================================================

development_cases = case_partition[
    case_partition["partition"].eq("DEVELOPMENT")
]

development_eligible_cases = development_cases[
    development_cases["primary_correctness_eligible"].fillna(False)
].copy()

table_09 = pd.crosstab(
    development_eligible_cases["actionable"].astype(int),
    development_eligible_cases["diagnostic_correctness"].astype(int),
)

table_09 = (
    table_09.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    .rename(columns={0: "Incorrect", 1: "Correct"})
    .rename_axis("E")
    .reset_index()
)

save_table(
    table_09,
    "table_09_development_evidence_readiness_by_correctness.csv",
)


# =====================================================================
# TABLE 10
# Development model performance
# =====================================================================

table_10 = development_metrics[
    [
        "model",
        "n",
        "Y_D_0",
        "Y_D_1",
        "observed_correctness_rate",
        "mean_predicted_reliability",
        "brier_score",
        "auroc",
        "auprc",
        "ece_5_quantile_bins",
    ]
].copy()

save_table(
    table_10,
    "table_10_development_model_performance.csv",
)


# =====================================================================
# TABLE 11
# Development detailed calibration
# =====================================================================

table_11 = development_calibration.copy()

save_table(
    table_11,
    "table_11_development_detailed_calibration.csv",
)


# =====================================================================
# TABLE 12
# Development bootstrap intervals
# =====================================================================

table_12 = development_bootstrap.copy()

save_table(
    table_12,
    "table_12_development_scenario_bootstrap_intervals.csv",
)


# =====================================================================
# TABLE 13
# Development paired model differences
# =====================================================================

table_13 = development_paired.copy()

save_table(
    table_13,
    "table_13_development_paired_model_differences.csv",
)


# =====================================================================
# TABLE 14
# M_EU grouped-CV coefficient stability
# =====================================================================

meu_coefficients = cv_coefficients[
    cv_coefficients["model"].eq("M_EU")
].copy()

# Keep actual slope predictors only.
meu_coefficients = meu_coefficients[
    meu_coefficients["parameter"].isin(
        ["E", "mean_self_confidence", "diagnostic_entropy"]
    )
]

table_14 = (
    meu_coefficients.groupby("parameter")
    .agg(
        Mean_coefficient=("value", "mean"),
        SD_coefficient=("value", "std"),
        Minimum=("value", "min"),
        Maximum=("value", "max"),
        Folds=("cv_fold", "nunique"),
    )
    .reset_index()
)

save_table(
    table_14,
    "table_14_meu_grouped_cv_coefficient_stability.csv",
)


# =====================================================================
# TABLE 15
# Development policy summary
# =====================================================================

model_summaries = policy_summary["model_summaries"]

table_15 = pd.DataFrame(
    [
        {
            "Model": model_name,
            "Cases": values["n_cases"],
            "Observed correctness": values[
                "observed_correctness_rate"
            ],
            "Prediction minimum": values["prediction_min"],
            "Prediction maximum": values["prediction_max"],
            "Highest non-empty threshold": values[
                "highest_nonempty_threshold"
            ],
            "Highest observed accuracy": values[
                "highest_accuracy_observed"
            ],
            "Coverage at highest observed accuracy": values[
                "coverage_at_highest_accuracy"
            ],
            "Threshold at highest observed accuracy": values[
                "threshold_at_highest_accuracy"
            ],
        }
        for model_name, values in model_summaries.items()
    ]
)

save_table(
    table_15,
    "table_15_development_selective_risk_summary.csv",
)


# =====================================================================
# TABLE 16
# Final holdout estimator performance
# =====================================================================

table_16 = holdout_metrics.copy()

save_table(
    table_16,
    "table_16_final_holdout_estimator_performance.csv",
)


# =====================================================================
# TABLE 17
# Final holdout bootstrap intervals
# =====================================================================

table_17 = holdout_bootstrap.copy()

save_table(
    table_17,
    "table_17_final_holdout_bootstrap_intervals.csv",
)


# =====================================================================
# TABLE 18
# Final holdout paired differences
# =====================================================================

table_18 = holdout_paired.copy()

save_table(
    table_18,
    "table_18_final_holdout_paired_differences.csv",
)


# =====================================================================
# TABLE 19
# Holdout E x Y_D
# =====================================================================

holdout_eligible = holdout_policy_cases[
    holdout_policy_cases["Y_D"].notna()
].copy()

table_19 = pd.crosstab(
    holdout_eligible["E"].astype(int),
    holdout_eligible["Y_D"].astype(int),
)

table_19 = (
    table_19.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    .rename(columns={0: "Incorrect", 1: "Correct"})
    .rename_axis("E")
    .reset_index()
)

save_table(
    table_19,
    "table_19_holdout_evidence_readiness_by_correctness.csv",
)


# =====================================================================
# TABLE 20
# Holdout PIDRL policy-state evaluation
# =====================================================================

state_order = ["ABSTAIN", "WITHHOLD", "QUALIFY", "ISSUE"]

table_20 = (
    holdout_policy_states
    .set_index("decision_state")
    .reindex(state_order)
    .reset_index()
)

save_table(
    table_20,
    "table_20_holdout_pidrl_policy_state_evaluation.csv",
)


# =====================================================================
# TABLE 21
# Holdout modal diagnosis distribution
# =====================================================================

holdout_diagnosis_counts = (
    holdout_policy_cases["modal_diagnosis"]
    .value_counts()
    .reindex(diagnosis_order, fill_value=0)
)

table_21 = (
    holdout_diagnosis_counts
    .rename_axis("Modal diagnosis")
    .reset_index(name="Cases")
)

save_table(
    table_21,
    "table_21_holdout_modal_diagnosis_distribution.csv",
)


# =====================================================================
# TABLE 22
# Holdout decision appropriateness
# =====================================================================

table_22 = (
    holdout_policy_cases["Y_A"]
    .value_counts()
    .reindex([1, 0], fill_value=0)
    .rename_axis("Y_A")
    .reset_index(name="Cases")
)

table_22["Interpretation"] = table_22["Y_A"].map(
    {1: "Appropriate", 0: "Inappropriate"}
)

table_22 = table_22[
    ["Y_A", "Interpretation", "Cases"]
]

save_table(
    table_22,
    "table_22_holdout_decision_appropriateness.csv",
)


# =====================================================================
# TABLE 23
# Holdout error categories
# =====================================================================

holdout_error_counts = (
    holdout_policy_cases["error_category"]
    .value_counts()
    .reindex(error_order, fill_value=0)
)

table_23 = (
    holdout_error_counts
    .rename_axis("Error category")
    .reset_index(name="Cases")
)

save_table(
    table_23,
    "table_23_holdout_error_category_distribution.csv",
)

# ============================================================
# ADDITIONAL DISSERTATION EVIDENCE TABLES
# Completes frozen A8.6-A8.12 result coverage
# ============================================================

import json

OUTPUTS_ROOT = ANALYSIS_OUTPUTS

A8_6_DIR = A86_DIR
A8_7_DIR = A87_DIR
A8_9_DIR = A89_DIR
A8_10_DIR = A810_DIR
A8_11_DIR = A811_DIR
A8_12_DIR = A812_DIR


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


# ------------------------------------------------------------
# Load canonical frozen sources
# ------------------------------------------------------------

case_df = pd.read_csv(
    A8_6_DIR / "a8_6_evidence_case_reliability_dataset.csv"
)

generation_df = load_jsonl(
    A8_6_DIR / "a8_6_generation_level_diagnostic_outcomes.jsonl"
)

partition_df = pd.read_csv(
    A8_7_DIR / "a8_7_scenario_development_holdout_assignment.csv"
)

selection = load_json(
    A8_9_DIR / "a8_9_reliability_estimator_selection.json"
)

policy = load_json(
    A8_10_DIR / "a8_10_frozen_post_inference_reliability_policy.json"
)

meu_parameters = load_json(
    A8_11_DIR / "a8_11_final_development_meu_parameters.json"
)

me_parameters = load_json(
    A8_12_DIR / "a8_12_final_development_me_parameters.json"
)

holdout_policy_df = pd.read_csv(
    A8_12_DIR / "a8_12_holdout_feature_policy_cases.csv"
)


# ------------------------------------------------------------
# Table 24
# Generation-level outcome summary
# ------------------------------------------------------------

generation_summary_rows = []

generation_summary_rows.append({
    "outcome_family": "Diagnostic correctness Y_D",
    "outcome": "Correct",
    "n": int((generation_df["diagnostic_correctness"] == 1).sum()),
})

generation_summary_rows.append({
    "outcome_family": "Diagnostic correctness Y_D",
    "outcome": "Incorrect",
    "n": int((generation_df["diagnostic_correctness"] == 0).sum()),
})

generation_summary_rows.append({
    "outcome_family": "Diagnostic correctness Y_D",
    "outcome": "UNKNOWN / not eligible",
    "n": int(generation_df["diagnostic_correctness"].isna().sum()),
})

generation_summary_rows.append({
    "outcome_family": "Decision appropriateness Y_A",
    "outcome": "Appropriate",
    "n": int((generation_df["decision_appropriateness"] == 1).sum()),
})

generation_summary_rows.append({
    "outcome_family": "Decision appropriateness Y_A",
    "outcome": "Inappropriate",
    "n": int((generation_df["decision_appropriateness"] == 0).sum()),
})

table_24 = pd.DataFrame(generation_summary_rows)
table_24["total_generations"] = len(generation_df)
table_24["analysis_note"] = (
    "Repeated generations are nested within evidence cases and are not "
    "treated as independent observations in the primary reliability modelling."
)

save_table(
    table_24,
    "table_24_generation_level_outcome_summary.csv",
)


# ------------------------------------------------------------
# Table 25
# Outcome-construction integrity summary
# ------------------------------------------------------------

generations_per_case = (
    generation_df.groupby("case_id")
    .size()
)

table_25 = pd.DataFrame([
    {
        "measure": "Evidence cases",
        "value": len(case_df),
    },
    {
        "measure": "Generation records",
        "value": len(generation_df),
    },
    {
        "measure": "Minimum generations per case",
        "value": int(generations_per_case.min()),
    },
    {
        "measure": "Maximum generations per case",
        "value": int(generations_per_case.max()),
    },
    {
        "measure": "Modal diagnosis ties",
        "value": int(case_df["modal_tie"].astype(bool).sum()),
    },
    {
        "measure": "Primary Y_D eligible cases",
        "value": int(case_df["primary_correctness_eligible"].astype(bool).sum()),
    },
    {
        "measure": "UNKNOWN modal diagnoses",
        "value": int((case_df["modal_diagnosis"] == "UNKNOWN").sum()),
    },
])

save_table(
    table_25,
    "table_25_outcome_construction_integrity_summary.csv",
)


# ------------------------------------------------------------
# Table 26
# Development and holdout analytical sample summary
# ------------------------------------------------------------

case_with_partition = case_df.merge(
    partition_df[["scenario_id", "partition"]],
    on="scenario_id",
    how="left",
    validate="many_to_one",
)

analytical_rows = []

for partition_name in ["development", "holdout"]:
    subset = case_with_partition[
        case_with_partition["partition"].str.lower() == partition_name
    ]

    eligible = subset[
        subset["primary_correctness_eligible"].astype(bool)
    ]

    analytical_rows.append({
        "partition": partition_name.capitalize(),
        "independent_scenarios": int(subset["scenario_id"].nunique()),
        "evidence_cases": len(subset),
        "Y_D_eligible_cases": len(eligible),
        "eligible_represented_scenarios": int(
            eligible["scenario_id"].nunique()
        ),
        "Y_D_correct": int(
            (eligible["diagnostic_correctness"] == 1).sum()
        ),
        "Y_D_incorrect": int(
            (eligible["diagnostic_correctness"] == 0).sum()
        ),
    })

table_26 = pd.DataFrame(analytical_rows)

save_table(
    table_26,
    "table_26_development_holdout_analytical_sample_summary.csv",
)


# =====================================================================
# TABLE 27
# Frozen reliability estimator specifications
# =====================================================================

me_predictor_map = {
    predictor["name"]: predictor
    for predictor in me_parameters["predictors"]
}

meu_predictor_map = {
    predictor["name"]: predictor
    for predictor in meu_parameters["predictors"]
}

me_intercept = float(me_parameters["flic_intercept"])

me_e_coefficient = float(
    me_predictor_map["E"]["coefficient"]
)

meu_intercept = float(meu_parameters["flic_intercept"])

meu_e_coefficient = float(
    meu_predictor_map["E"]["coefficient"]
)

meu_confidence_coefficient = float(
    meu_predictor_map["mean_self_confidence"]["coefficient"]
)

meu_entropy_coefficient = float(
    meu_predictor_map["diagnostic_entropy"]["coefficient"]
)

table_27 = pd.DataFrame(
    [
        {
            "Estimator": me_parameters["model"],
            "Target": "Y_D",
            "Predictors": ", ".join(
                predictor["name"]
                for predictor in me_parameters["predictors"]
            ),
            "Estimation method": me_parameters["estimator"],
            "Intercept": me_intercept,
            "E coefficient": me_e_coefficient,
            "Mean self-confidence coefficient": np.nan,
            "Diagnostic entropy coefficient": np.nan,
            "Equation": (
                f"logit(p) = {me_intercept:.9f} "
                f"+ {me_e_coefficient:.9f} E"
            ),
            "Role": me_parameters["role"],
        },
        {
            "Estimator": meu_parameters["model"],
            "Target": "Y_D",
            "Predictors": ", ".join(
                predictor["name"]
                for predictor in meu_parameters["predictors"]
            ),
            "Estimation method": meu_parameters["estimator"],
            "Intercept": meu_intercept,
            "E coefficient": meu_e_coefficient,
            "Mean self-confidence coefficient": (
                meu_confidence_coefficient
            ),
            "Diagnostic entropy coefficient": (
                meu_entropy_coefficient
            ),
            "Equation": (
                f"logit(p) = {meu_intercept:.9f} "
                f"+ {meu_e_coefficient:.9f} E "
                f"+ {meu_confidence_coefficient:.9f} C_self_bar "
                f"+ {meu_entropy_coefficient:.9f} H_D"
            ),
            "Role": "Selected combined development estimator",
        },
    ]
)

save_table(
    table_27,
    "table_27_frozen_reliability_estimator_specifications.csv",
)


# =====================================================================
# TABLE 28
# Frozen post-inference reliability policy
# =====================================================================

issue_threshold = float(
    policy["issue_threshold"]["value"]
)

threshold_selection_rule = (
    policy["issue_threshold"]["selection_rule"]
)

criterion_interpretation = (
    policy["issue_threshold"]["criterion_interpretation"]
)

policy_logic = policy["policy_logic"]

table_28 = pd.DataFrame(
    [
        {
            "Condition": "Modal diagnosis = UNKNOWN",
            "Decision state": "ABSTAIN",
            "Interpretation": (
                meu_parameters["unknown_semantics"]["handling"]
                + ": primary reliability estimation is not applied "
                "because UNKNOWN is not a substantive fault diagnosis."
            ),
        },
        {
            "Condition": policy_logic["WITHHOLD"]["condition"],
            "Decision state": "WITHHOLD",
            "Interpretation": policy_logic["WITHHOLD"]["meaning"],
        },
        {
            "Condition": policy_logic["QUALIFY"]["condition"],
            "Decision state": "QUALIFY",
            "Interpretation": policy_logic["QUALIFY"]["meaning"],
        },
        {
            "Condition": policy_logic["ISSUE"]["condition"],
            "Decision state": "ISSUE",
            "Interpretation": policy_logic["ISSUE"]["meaning"],
        },
    ]
)

table_28["Issue threshold"] = issue_threshold
table_28["Threshold selection rule"] = threshold_selection_rule
table_28["Claim boundary"] = criterion_interpretation

save_table(
    table_28,
    "table_28_frozen_post_inference_reliability_policy.csv",
)


# ------------------------------------------------------------
# Table 29
# Holdout evidence-readiness distribution and correctness
# ------------------------------------------------------------

holdout_actionability_rows = []

for e_value in [0, 1]:
    all_e = holdout_policy_df[
        holdout_policy_df["E"] == e_value
    ]

    substantive_e = all_e[
        all_e["Y_D"].notna()
    ]

    holdout_actionability_rows.append({
        "E": e_value,
        "all_holdout_cases": len(all_e),
        "substantive_Y_D_eligible_cases": len(substantive_e),
        "Y_D_correct": int((substantive_e["Y_D"] == 1).sum()),
        "Y_D_incorrect": int((substantive_e["Y_D"] == 0).sum()),
        "substantive_accuracy": (
            float((substantive_e["Y_D"] == 1).mean())
            if len(substantive_e) > 0
            else None
        ),
    })

holdout_actionability = pd.DataFrame(
    holdout_actionability_rows
)

save_table(
    holdout_actionability,
    "table_29_holdout_evidence_readiness_and_correctness.csv"
)

# =====================================================================
# CONSOLIDATED DISSERTATION-READY TABLES
#
# These tables consolidate the frozen numerical evidence into a smaller
# set of presentation tables suitable for the dissertation main body.
# They do not perform new scientific analysis or alter frozen results.
# =====================================================================


# =====================================================================
# CONSOLIDATED TABLE C1
# Experimental sample and primary analytical outcome
# =====================================================================

development_partition_row = table_26[
    table_26["partition"].str.lower().eq("development")
].iloc[0]

holdout_partition_row = table_26[
    table_26["partition"].str.lower().eq("holdout")
].iloc[0]

full_summary = dict(
    zip(table_01["Measure"], table_01["Value"])
)

table_c1 = pd.DataFrame(
    [
        {
            "Stage / measure": "Full experiment",
            "Independent scenarios": int(
                full_summary["Independent scenarios"]
            ),
            "Evidence cases": int(
                full_summary["Evidence cases"]
            ),
            "Y_D eligible cases": int(
                full_summary["Primary Y_D eligible cases"]
            ),
            "Correct": int(
                full_summary["Correct eligible diagnoses"]
            ),
            "Incorrect": int(
                full_summary["Incorrect eligible diagnoses"]
            ),
            "Eligible accuracy": float(
                full_summary["Eligible diagnostic accuracy"]
            ),
        },
        {
            "Stage / measure": "Development partition",
            "Independent scenarios": int(
                development_partition_row["independent_scenarios"]
            ),
            "Evidence cases": int(
                development_partition_row["evidence_cases"]
            ),
            "Y_D eligible cases": int(
                development_partition_row["Y_D_eligible_cases"]
            ),
            "Correct": int(
                development_partition_row["Y_D_correct"]
            ),
            "Incorrect": int(
                development_partition_row["Y_D_incorrect"]
            ),
            "Eligible accuracy": float(
                development_partition_row["Y_D_correct"]
                / development_partition_row["Y_D_eligible_cases"]
            ),
        },
        {
            "Stage / measure": "Final holdout partition",
            "Independent scenarios": int(
                holdout_partition_row["independent_scenarios"]
            ),
            "Evidence cases": int(
                holdout_partition_row["evidence_cases"]
            ),
            "Y_D eligible cases": int(
                holdout_partition_row["Y_D_eligible_cases"]
            ),
            "Correct": int(
                holdout_partition_row["Y_D_correct"]
            ),
            "Incorrect": int(
                holdout_partition_row["Y_D_incorrect"]
            ),
            "Eligible accuracy": float(
                holdout_partition_row["Y_D_correct"]
                / holdout_partition_row["Y_D_eligible_cases"]
            ),
        },
    ]
)

table_c1["Additional context"] = [
    (
        f"{n_generations} total generations; "
        f"{int(unknown.sum())} modal UNKNOWN cases; "
        f"{int(table_25.loc[table_25['measure'].eq('Modal diagnosis ties'), 'value'].iloc[0])} "
        "modal ties"
    ),
    (
        f"{int(development_partition_row['eligible_represented_scenarios'])} "
        "scenarios represented among Y_D-eligible cases"
    ),
    (
        f"{int(holdout_partition_row['eligible_represented_scenarios'])} "
        "scenarios represented among Y_D-eligible cases"
    ),
]

save_consolidated_table(
    table_c1,
    "consolidated_table_c1_experimental_sample_and_primary_outcome.csv",
)


# =====================================================================
# CONSOLIDATED TABLE C2
# Development reliability estimator performance
# =====================================================================

model_order = ["M_E", "M_C", "M_H", "M_U", "M_EU"]

table_c2 = (
    table_10[
        [
            "model",
            "n",
            "observed_correctness_rate",
            "brier_score",
            "auroc",
            "auprc",
            "ece_5_quantile_bins",
        ]
    ]
    .set_index("model")
    .reindex(model_order)
    .reset_index()
    .rename(
        columns={
            "model": "Estimator",
            "n": "N",
            "observed_correctness_rate": "Observed correctness",
            "brier_score": "Brier score",
            "auroc": "AUROC",
            "auprc": "AUPRC",
            "ece_5_quantile_bins": "ECE",
        }
    )
)

save_consolidated_table(
    table_c2,
    "consolidated_table_c2_development_estimator_performance.csv",
)


# =====================================================================
# CONSOLIDATED TABLE C3
# Development comparison of M_E and M_EU
# =====================================================================

development_focus = (
    table_10[
        table_10["model"].isin(["M_E", "M_EU"])
    ]
    .set_index("model")
    .reindex(["M_E", "M_EU"])
)

development_bootstrap_focus = table_12[
    table_12["model"].isin(["M_E", "M_EU"])
].copy()

development_bootstrap_lookup = {
    (row["model"], row["metric"]): row
    for _, row in development_bootstrap_focus.iterrows()
}

development_paired_focus = table_13[
    table_13["reference_model"].eq("M_E")
    & table_13["comparison_model"].eq("M_EU")
].copy()

development_paired_lookup = {
    row["metric"]: row
    for _, row in development_paired_focus.iterrows()
}

metric_specs = [
    ("Brier score", "brier_score"),
    ("AUROC", "auroc"),
    ("AUPRC", "auprc"),
]

c3_rows = []

for display_metric, source_metric in metric_specs:
    me_row = development_focus.loc["M_E"]
    meu_row = development_focus.loc["M_EU"]

    me_bootstrap = development_bootstrap_lookup[
        ("M_E", source_metric)
    ]
    meu_bootstrap = development_bootstrap_lookup[
        ("M_EU", source_metric)
    ]
    paired_row = development_paired_lookup[source_metric]

    c3_rows.append(
        {
            "Metric": display_metric,
            "M_E point estimate": me_row[source_metric],
            "M_E 95% CI lower": me_bootstrap["ci_lower_95"],
            "M_E 95% CI upper": me_bootstrap["ci_upper_95"],
            "M_EU point estimate": meu_row[source_metric],
            "M_EU 95% CI lower": meu_bootstrap["ci_lower_95"],
            "M_EU 95% CI upper": meu_bootstrap["ci_upper_95"],
            "M_EU - M_E difference": paired_row[
                "point_difference"
            ],
            "Difference 95% CI lower": paired_row[
                "ci_lower_95"
            ],
            "Difference 95% CI upper": paired_row[
                "ci_upper_95"
            ],
        }
    )

table_c3 = pd.DataFrame(c3_rows)

save_consolidated_table(
    table_c3,
    "consolidated_table_c3_development_me_vs_meu_comparison.csv",
)


# =====================================================================
# CONSOLIDATED TABLE C4
# Final holdout comparison of M_E and M_EU
# =====================================================================

holdout_focus = (
    table_16[
        table_16["model"].isin(["M_E", "M_EU"])
    ]
    .set_index("model")
    .reindex(["M_E", "M_EU"])
)

holdout_bootstrap_focus = table_17[
    table_17["model"].isin(["M_E", "M_EU"])
].copy()

holdout_bootstrap_lookup = {
    (row["model"], row["metric"]): row
    for _, row in holdout_bootstrap_focus.iterrows()
}

holdout_paired_focus = table_18[
    table_18["reference_model"].eq("M_E")
    & table_18["comparison_model"].eq("M_EU")
].copy()

holdout_paired_lookup = {
    row["metric"]: row
    for _, row in holdout_paired_focus.iterrows()
}

c4_rows = []

for display_metric, source_metric in metric_specs:
    me_row = holdout_focus.loc["M_E"]
    meu_row = holdout_focus.loc["M_EU"]

    me_bootstrap = holdout_bootstrap_lookup[
        ("M_E", source_metric)
    ]
    meu_bootstrap = holdout_bootstrap_lookup[
        ("M_EU", source_metric)
    ]
    paired_row = holdout_paired_lookup[source_metric]

    c4_rows.append(
        {
            "Metric": display_metric,
            "M_E point estimate": me_row[source_metric],
            "M_E 95% CI lower": me_bootstrap["ci_lower_95"],
            "M_E 95% CI upper": me_bootstrap["ci_upper_95"],
            "M_EU point estimate": meu_row[source_metric],
            "M_EU 95% CI lower": meu_bootstrap["ci_lower_95"],
            "M_EU 95% CI upper": meu_bootstrap["ci_upper_95"],
            "M_EU - M_E difference": paired_row[
                "point_difference"
            ],
            "Difference 95% CI lower": paired_row[
                "ci_lower_95"
            ],
            "Difference 95% CI upper": paired_row[
                "ci_upper_95"
            ],
        }
    )

table_c4 = pd.DataFrame(c4_rows)

save_consolidated_table(
    table_c4,
    "consolidated_table_c4_final_holdout_me_vs_meu_evaluation.csv",
)


# =====================================================================
# CONSOLIDATED TABLE C5
# PIDRL prototype policy evaluation
# =====================================================================

development_operating_point = policy[
    "development_operating_point"
]

table_c5 = pd.DataFrame(
    [
        {
            "Evaluation": "Development frozen ISSUE operating point",
            "Decision state": "ISSUE",
            "Cases": int(
                development_operating_point["n_accepted"]
            ),
            "Coverage": float(
                development_operating_point["coverage"]
            ),
            "Diagnostic accuracy": float(
                development_operating_point["accepted_accuracy"]
            ),
            "Selective risk": float(
                development_operating_point["selective_risk"]
            ),
            "Y_A appropriateness rate": np.nan,
            "Mean predicted reliability": np.nan,
        },
        {
            "Evaluation": "Final holdout",
            "Decision state": "ABSTAIN",
            "Cases": int(
                table_20.loc[
                    table_20["decision_state"].eq("ABSTAIN"),
                    "n_cases",
                ].iloc[0]
            ),
            "Coverage": float(
                table_20.loc[
                    table_20["decision_state"].eq("ABSTAIN"),
                    "fraction_of_all_holdout",
                ].iloc[0]
            ),
            "Diagnostic accuracy": np.nan,
            "Selective risk": np.nan,
            "Y_A appropriateness rate": float(
                table_20.loc[
                    table_20["decision_state"].eq("ABSTAIN"),
                    "Y_A_appropriateness_rate",
                ].iloc[0]
            ),
            "Mean predicted reliability": np.nan,
        },
        {
            "Evaluation": "Final holdout",
            "Decision state": "WITHHOLD",
            "Cases": int(
                table_20.loc[
                    table_20["decision_state"].eq("WITHHOLD"),
                    "n_cases",
                ].iloc[0]
            ),
            "Coverage": float(
                table_20.loc[
                    table_20["decision_state"].eq("WITHHOLD"),
                    "fraction_of_all_holdout",
                ].iloc[0]
            ),
            "Diagnostic accuracy": float(
                table_20.loc[
                    table_20["decision_state"].eq("WITHHOLD"),
                    "Y_D_accuracy",
                ].iloc[0]
            ),
            "Selective risk": np.nan,
            "Y_A appropriateness rate": float(
                table_20.loc[
                    table_20["decision_state"].eq("WITHHOLD"),
                    "Y_A_appropriateness_rate",
                ].iloc[0]
            ),
            "Mean predicted reliability": float(
                table_20.loc[
                    table_20["decision_state"].eq("WITHHOLD"),
                    "mean_predicted_reliability",
                ].iloc[0]
            ),
        },
        {
            "Evaluation": "Final holdout",
            "Decision state": "QUALIFY",
            "Cases": int(
                table_20.loc[
                    table_20["decision_state"].eq("QUALIFY"),
                    "n_cases",
                ].iloc[0]
            ),
            "Coverage": float(
                table_20.loc[
                    table_20["decision_state"].eq("QUALIFY"),
                    "fraction_of_all_holdout",
                ].iloc[0]
            ),
            "Diagnostic accuracy": float(
                table_20.loc[
                    table_20["decision_state"].eq("QUALIFY"),
                    "Y_D_accuracy",
                ].iloc[0]
            ),
            "Selective risk": np.nan,
            "Y_A appropriateness rate": float(
                table_20.loc[
                    table_20["decision_state"].eq("QUALIFY"),
                    "Y_A_appropriateness_rate",
                ].iloc[0]
            ),
            "Mean predicted reliability": float(
                table_20.loc[
                    table_20["decision_state"].eq("QUALIFY"),
                    "mean_predicted_reliability",
                ].iloc[0]
            ),
        },
        {
            "Evaluation": "Final holdout",
            "Decision state": "ISSUE",
            "Cases": int(
                table_20.loc[
                    table_20["decision_state"].eq("ISSUE"),
                    "n_cases",
                ].iloc[0]
            ),
            "Coverage": float(
                table_20.loc[
                    table_20["decision_state"].eq("ISSUE"),
                    "fraction_of_all_holdout",
                ].iloc[0]
            ),
            "Diagnostic accuracy": float(
                table_20.loc[
                    table_20["decision_state"].eq("ISSUE"),
                    "Y_D_accuracy",
                ].iloc[0]
            ),
            "Selective risk": float(
                1.0
                - table_20.loc[
                    table_20["decision_state"].eq("ISSUE"),
                    "Y_D_accuracy",
                ].iloc[0]
            ),
            "Y_A appropriateness rate": float(
                table_20.loc[
                    table_20["decision_state"].eq("ISSUE"),
                    "Y_A_appropriateness_rate",
                ].iloc[0]
            ),
            "Mean predicted reliability": float(
                table_20.loc[
                    table_20["decision_state"].eq("ISSUE"),
                    "mean_predicted_reliability",
                ].iloc[0]
            ),
        },
    ]
)

table_c5["Frozen ISSUE threshold"] = issue_threshold

save_consolidated_table(
    table_c5,
    "consolidated_table_c5_pidrl_holdout_policy_evaluation.csv",
)

# =====================================================================
# Manifest
# =====================================================================

generated_tables = sorted(
    p.name for p in TABLES_DIR.glob("*.csv")
)

manifest = {
    "name": "Dissertation Results Artefact Generation",
    "source_status": "Frozen A8 analysis outputs",
    "scientific_experiment_rerun": False,
    "models_retrained": False,
    "holdout_reanalysed_for_selection": False,
    "generated_table_count": len(generated_tables),
    "generated_tables": generated_tables,
    "notes": [
        "Tables are derived from frozen A8.6-A8.12 analysis artefacts.",
        "Generation of dissertation tables does not alter scientific results.",
        "Main-body versus appendix placement is handled separately from numerical derivation.",
    ],
}

manifest_path = (
    MANIFESTS_DIR
    / "dissertation_results_artifact_generation_manifest.json"
)

with manifest_path.open("w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print(f"[MANIFEST] {manifest_path}")

print()
print("=" * 72)
print("DISSERTATION RESULTS TABLE GENERATION COMPLETE")
print(f"Tables generated: {len(generated_tables)}")
print("=" * 72)