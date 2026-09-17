import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


# ============================================================
# Frozen configuration
# ============================================================

SEED = 20260917
N_BOOTSTRAP = 2000
CONFIDENCE_LEVEL = 0.95

MODEL_ORDER = [
    "M_E",
    "M_C",
    "M_H",
    "M_U",
    "M_EU",
]

REFERENCE_MODEL = "M_E"


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

A8_8_DIR = (
    BASE
    / "a8_8_evaluation_and_calibration"
)

PREDICTIONS_FILE = (
    A8_7_DIR
    / "a8_7_grouped_cv_out_of_fold_reliability_predictions.csv"
)

PROTOCOL_FILE = (
    A8_8_DIR
    / "a8_8_evaluation_protocol.json"
)

BOOTSTRAP_SUMMARY_FILE = (
    A8_8_DIR
    / "a8_8_scenario_bootstrap_metric_intervals.csv"
)

PAIRED_DIFFERENCES_FILE = (
    A8_8_DIR
    / "a8_8_scenario_bootstrap_paired_model_differences.csv"
)

BOOTSTRAP_MANIFEST_FILE = (
    A8_8_DIR
    / "a8_8_scenario_bootstrap_manifest.json"
)


# ============================================================
# Utilities
# ============================================================

def load_csv(path):
    with path.open(
        newline="",
        encoding="utf-8",
    ) as f:
        return list(csv.DictReader(f))


def percentile_interval(values):
    alpha = 1.0 - CONFIDENCE_LEVEL

    lower = float(
        np.quantile(
            values,
            alpha / 2.0,
        )
    )

    upper = float(
        np.quantile(
            values,
            1.0 - alpha / 2.0,
        )
    )

    return lower, upper


def calculate_metrics(y, p):
    if len(np.unique(y)) < 2:
        return None

    return {
        "brier_score": float(
            brier_score_loss(y, p)
        ),
        "auroc": float(
            roc_auc_score(y, p)
        ),
        "auprc": float(
            average_precision_score(y, p)
        ),
    }


# ============================================================
# Load inputs and frozen protocol
# ============================================================

rows = load_csv(PREDICTIONS_FILE)

with PROTOCOL_FILE.open(
    encoding="utf-8",
) as f:
    protocol = json.load(f)


print("=" * 80)
print("A8.8.3 SCENARIO-LEVEL BOOTSTRAP UNCERTAINTY ANALYSIS")
print("=" * 80)


# ============================================================
# Verify frozen protocol
# ============================================================

bootstrap_protocol = protocol[
    "uncertainty_estimation"
]

if (
    bootstrap_protocol["method"]
    != "scenario_level_bootstrap"
):
    raise RuntimeError(
        "Unexpected bootstrap method in frozen protocol."
    )

if (
    bootstrap_protocol["experimental_unit"]
    != "scenario_id"
):
    raise RuntimeError(
        "Frozen experimental unit is not scenario_id."
    )

if (
    int(bootstrap_protocol["replicates"])
    != N_BOOTSTRAP
):
    raise RuntimeError(
        "Bootstrap replicate count differs "
        "from frozen protocol."
    )

if (
    int(bootstrap_protocol["seed"])
    != SEED
):
    raise RuntimeError(
        "Bootstrap seed differs from frozen protocol."
    )

if len(rows) != 1190:
    raise RuntimeError(
        f"Expected 1,190 predictions, found {len(rows)}."
    )


print("\nFrozen protocol:")
print("Method: scenario-level bootstrap")
print("Replicates:", N_BOOTSTRAP)
print("Seed:", SEED)
print("Confidence level:", CONFIDENCE_LEVEL)
print("Holdout used: NO")


# ============================================================
# Build case-level aligned representation
#
# Each case has five model predictions. We construct one
# analytical row per case so the same bootstrap sample is
# applied to every candidate model.
# ============================================================

by_case = defaultdict(list)

for row in rows:
    by_case[row["case_id"]].append(row)

if len(by_case) != 238:
    raise RuntimeError(
        f"Expected 238 unique cases, found {len(by_case)}."
    )


case_records = []

for case_id, case_rows in by_case.items():

    if len(case_rows) != 5:
        raise RuntimeError(
            f"{case_id}: expected five model predictions."
        )

    models = {
        row["model"]
        for row in case_rows
    }

    if models != set(MODEL_ORDER):
        raise RuntimeError(
            f"{case_id}: incomplete model set."
        )

    scenario_ids = {
        row["scenario_id"]
        for row in case_rows
    }

    outcomes = {
        int(row["Y_D"])
        for row in case_rows
    }

    if len(scenario_ids) != 1:
        raise RuntimeError(
            f"{case_id}: inconsistent scenario IDs."
        )

    if len(outcomes) != 1:
        raise RuntimeError(
            f"{case_id}: inconsistent Y_D."
        )

    prediction_map = {
        row["model"]:
            float(row["predicted_reliability"])
        for row in case_rows
    }

    case_records.append(
        {
            "case_id": case_id,
            "scenario_id":
                next(iter(scenario_ids)),
            "Y_D":
                next(iter(outcomes)),
            "predictions":
                prediction_map,
        }
    )


# ============================================================
# Group eligible cases by scenario
# ============================================================

scenario_to_cases = defaultdict(list)

for record in case_records:
    scenario_to_cases[
        record["scenario_id"]
    ].append(record)

scenario_ids = sorted(
    scenario_to_cases
)

n_scenarios = len(scenario_ids)

print("\nAnalytical structure:")
print("Eligible cases:", len(case_records))
print("Represented scenarios:", n_scenarios)

if n_scenarios != 68:
    raise RuntimeError(
        f"Expected 68 represented DEVELOPMENT scenarios, "
        f"found {n_scenarios}."
    )


# ============================================================
# Original point estimates
# ============================================================

original_metrics = {}

for model in MODEL_ORDER:

    y = np.array(
        [
            record["Y_D"]
            for record in case_records
        ],
        dtype=int,
    )

    p = np.array(
        [
            record["predictions"][model]
            for record in case_records
        ],
        dtype=float,
    )

    metrics = calculate_metrics(
        y,
        p,
    )

    if metrics is None:
        raise RuntimeError(
            "Original dataset unexpectedly "
            "contains only one outcome class."
        )

    original_metrics[model] = metrics


print("\nOriginal point estimates:")

for model in MODEL_ORDER:
    m = original_metrics[model]

    print(
        f"{model}: "
        f"Brier={m['brier_score']:.6f}, "
        f"AUROC={m['auroc']:.6f}, "
        f"AUPRC={m['auprc']:.6f}"
    )


# ============================================================
# Scenario-level bootstrap
# ============================================================

rng = np.random.default_rng(SEED)

bootstrap_results = {
    model: {
        "brier_score": [],
        "auroc": [],
        "auprc": [],
    }
    for model in MODEL_ORDER
}

valid_replicates = 0
skipped_single_class = 0


for replicate in range(N_BOOTSTRAP):

    sampled_scenarios = rng.choice(
        scenario_ids,
        size=n_scenarios,
        replace=True,
    )

    sampled_cases = []

    # IMPORTANT:
    # Do not deduplicate sampled scenario IDs.
    # Repeated scenario selections must preserve
    # bootstrap multiplicity.
    for scenario_id in sampled_scenarios:
        sampled_cases.extend(
            scenario_to_cases[scenario_id]
        )

    y = np.array(
        [
            record["Y_D"]
            for record in sampled_cases
        ],
        dtype=int,
    )

    if len(np.unique(y)) < 2:
        skipped_single_class += 1
        continue

    replicate_metrics = {}

    for model in MODEL_ORDER:

        p = np.array(
            [
                record["predictions"][model]
                for record in sampled_cases
            ],
            dtype=float,
        )

        metrics = calculate_metrics(
            y,
            p,
        )

        if metrics is None:
            raise RuntimeError(
                "Unexpected single-class metric failure."
            )

        replicate_metrics[model] = metrics

        for metric_name, value in metrics.items():
            bootstrap_results[
                model
            ][
                metric_name
            ].append(value)

    valid_replicates += 1


print("\nBootstrap execution:")
print("Requested replicates:", N_BOOTSTRAP)
print("Valid replicates:", valid_replicates)
print(
    "Skipped single-class replicates:",
    skipped_single_class,
)


if valid_replicates < 1900:
    raise RuntimeError(
        "Unexpectedly high number of invalid "
        "bootstrap replicates."
    )


# ============================================================
# Metric confidence intervals
# ============================================================

summary_records = []

print("\n" + "=" * 80)
print("BOOTSTRAP METRIC INTERVALS")
print("=" * 80)


for model in MODEL_ORDER:

    print(f"\n{model}")

    for metric_name in [
        "brier_score",
        "auroc",
        "auprc",
    ]:

        values = np.array(
            bootstrap_results[
                model
            ][
                metric_name
            ],
            dtype=float,
        )

        lower, upper = percentile_interval(
            values
        )

        point = original_metrics[
            model
        ][
            metric_name
        ]

        bootstrap_mean = float(
            np.mean(values)
        )

        bootstrap_sd = float(
            np.std(
                values,
                ddof=1,
            )
        )

        summary_records.append(
            {
                "model": model,
                "metric": metric_name,
                "point_estimate": point,
                "bootstrap_mean":
                    bootstrap_mean,
                "bootstrap_sd":
                    bootstrap_sd,
                "ci_lower_95": lower,
                "ci_upper_95": upper,
                "valid_replicates":
                    valid_replicates,
            }
        )

        print(
            f"  {metric_name}: "
            f"point={point:.6f}, "
            f"bootstrap_mean={bootstrap_mean:.6f}, "
            f"95% CI="
            f"[{lower:.6f}, {upper:.6f}]"
        )


# ============================================================
# Paired differences against M_E
#
# Same scenario bootstrap samples are used for all models,
# making these paired comparisons.
#
# Difference orientation:
#
# Brier:
#   comparison - M_E
#   negative => comparison has lower Brier
#
# AUROC/AUPRC:
#   comparison - M_E
#   positive => comparison has higher discrimination
# ============================================================

paired_records = []

print("\n" + "=" * 80)
print("PAIRED BOOTSTRAP DIFFERENCES AGAINST M_E")
print("=" * 80)


for comparison_model in [
    "M_C",
    "M_H",
    "M_U",
    "M_EU",
]:

    print(f"\n{comparison_model} versus M_E")

    for metric_name in [
        "brier_score",
        "auroc",
        "auprc",
    ]:

        comparison_values = np.array(
            bootstrap_results[
                comparison_model
            ][
                metric_name
            ],
            dtype=float,
        )

        reference_values = np.array(
            bootstrap_results[
                REFERENCE_MODEL
            ][
                metric_name
            ],
            dtype=float,
        )

        differences = (
            comparison_values
            - reference_values
        )

        lower, upper = percentile_interval(
            differences
        )

        point_difference = (
            original_metrics[
                comparison_model
            ][
                metric_name
            ]
            - original_metrics[
                REFERENCE_MODEL
            ][
                metric_name
            ]
        )

        mean_difference = float(
            np.mean(differences)
        )

        if metric_name == "brier_score":
            favourable_probability = float(
                np.mean(
                    differences < 0.0
                )
            )
            favourable_definition = (
                "P(comparison Brier < M_E Brier)"
            )
        else:
            favourable_probability = float(
                np.mean(
                    differences > 0.0
                )
            )
            favourable_definition = (
                f"P(comparison {metric_name} "
                f"> M_E {metric_name})"
            )

        paired_records.append(
            {
                "reference_model":
                    REFERENCE_MODEL,
                "comparison_model":
                    comparison_model,
                "metric":
                    metric_name,
                "difference_definition":
                    (
                        f"{comparison_model} "
                        f"- {REFERENCE_MODEL}"
                    ),
                "point_difference":
                    point_difference,
                "bootstrap_mean_difference":
                    mean_difference,
                "ci_lower_95":
                    lower,
                "ci_upper_95":
                    upper,
                "favourable_probability":
                    favourable_probability,
                "favourable_definition":
                    favourable_definition,
                "valid_replicates":
                    valid_replicates,
            }
        )

        print(
            f"  {metric_name}: "
            f"point_diff="
            f"{point_difference:.6f}, "
            f"95% CI="
            f"[{lower:.6f}, {upper:.6f}], "
            f"favourable proportion="
            f"{favourable_probability:.4f}"
        )


# ============================================================
# Write durable outputs
# ============================================================

with BOOTSTRAP_SUMMARY_FILE.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    fieldnames = [
        "model",
        "metric",
        "point_estimate",
        "bootstrap_mean",
        "bootstrap_sd",
        "ci_lower_95",
        "ci_upper_95",
        "valid_replicates",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(
        summary_records
    )


with PAIRED_DIFFERENCES_FILE.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    fieldnames = [
        "reference_model",
        "comparison_model",
        "metric",
        "difference_definition",
        "point_difference",
        "bootstrap_mean_difference",
        "ci_lower_95",
        "ci_upper_95",
        "favourable_probability",
        "favourable_definition",
        "valid_replicates",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(
        paired_records
    )


manifest = {
    "stage": "A8.8",
    "substage": "A8.8.3",
    "name": (
        "Scenario-Level Bootstrap "
        "Uncertainty Analysis"
    ),
    "status": "COMPLETED_AFTER_VALIDATION",
    "bootstrap": {
        "unit": "scenario_id",
        "represented_scenarios": n_scenarios,
        "requested_replicates": N_BOOTSTRAP,
        "valid_replicates": valid_replicates,
        "skipped_single_class_replicates":
            skipped_single_class,
        "seed": SEED,
        "confidence_level":
            CONFIDENCE_LEVEL,
        "interval_method":
            "percentile",
        "cluster_multiplicity_preserved":
            True,
    },
    "metrics": [
        "brier_score",
        "auroc",
        "auprc",
    ],
    "paired_reference_model":
        REFERENCE_MODEL,
    "holdout_used": False,
    "model_selection_performed": False,
    "ece_bootstrapped": False,
    "ece_reason": (
        "ECE remains a supporting binned "
        "calibration measure and is not included "
        "in the primary scenario-level bootstrap."
    ),
    "outputs": {
        "metric_intervals":
            str(BOOTSTRAP_SUMMARY_FILE),
        "paired_differences":
            str(PAIRED_DIFFERENCES_FILE),
    },
}


with BOOTSTRAP_MANIFEST_FILE.open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        manifest,
        f,
        indent=2,
    )


print("\n" + "=" * 80)
print("OUTPUT ARTEFACTS")
print("=" * 80)

print(BOOTSTRAP_SUMMARY_FILE)
print(PAIRED_DIFFERENCES_FILE)
print(BOOTSTRAP_MANIFEST_FILE)

print("\nValidation:")
print("Frozen bootstrap protocol respected: PASS")
print("Scenario-level resampling: PASS")
print("Scenario multiplicity preserved: PASS")
print("Same bootstrap samples used across models: PASS")
print("Metric intervals generated: PASS")
print("Paired model differences generated: PASS")
print("Holdout untouched: PASS")
print("No model selection performed: PASS")

print(
    "\nA8.8.3 SCENARIO-LEVEL BOOTSTRAP "
    "UNCERTAINTY ANALYSIS: PASS"
)