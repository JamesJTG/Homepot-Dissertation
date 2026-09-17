import csv
import hashlib
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
    "M_EU",
]

REFERENCE_MODEL = "M_E"
COMPARISON_MODEL = "M_EU"


# ============================================================
# Paths
# ============================================================

BASE = Path(
    "research/diagnostic_reliability/analysis/outputs"
)

A8_12_DIR = (
    BASE
    / "a8_12_feature_level_evaluation"
)

PREDICTIONS_FILE = (
    A8_12_DIR
    / "a8_12_holdout_estimator_predictions.csv"
)

EVALUATION_SUMMARY_FILE = (
    A8_12_DIR
    / "a8_12_holdout_estimator_evaluation_summary.json"
)

PROTOCOL_FILE = (
    A8_12_DIR
    / "a8_12_pre_holdout_evaluation_protocol.json"
)

BOOTSTRAP_SUMMARY_FILE = (
    A8_12_DIR
    / "a8_12_holdout_scenario_bootstrap_metric_intervals.csv"
)

PAIRED_DIFFERENCES_FILE = (
    A8_12_DIR
    / "a8_12_holdout_scenario_bootstrap_paired_differences.csv"
)

BOOTSTRAP_MANIFEST_FILE = (
    A8_12_DIR
    / "a8_12_holdout_scenario_bootstrap_manifest.json"
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


def percentile_interval(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    if len(values) == 0:
        raise RuntimeError(
            "Cannot calculate interval "
            "from zero values."
        )

    alpha = (
        1.0
        - CONFIDENCE_LEVEL
    )

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


def calculate_brier(y, p):
    return float(
        brier_score_loss(
            y,
            p,
        )
    )


def calculate_discrimination(y, p):
    """
    AUROC requires both outcome classes.

    AUPRC is evaluated on the same two-class
    replicate set so the paired discrimination
    intervals use a consistent replicate population.
    """

    if len(
        np.unique(y)
    ) < 2:
        return None

    return {
        "auroc":
            float(
                roc_auc_score(
                    y,
                    p,
                )
            ),

        "auprc":
            float(
                average_precision_score(
                    y,
                    p,
                )
            ),
    }


# ============================================================
# Load frozen inputs
# ============================================================

required_files = [
    PREDICTIONS_FILE,
    EVALUATION_SUMMARY_FILE,
    PROTOCOL_FILE,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(
            path
        )


rows = load_csv(
    PREDICTIONS_FILE
)


with EVALUATION_SUMMARY_FILE.open(
    encoding="utf-8",
) as f:
    evaluation_summary = json.load(f)


with PROTOCOL_FILE.open(
    encoding="utf-8",
) as f:
    protocol = json.load(f)


print("=" * 80)
print(
    "A8.12.3 SCENARIO-LEVEL HOLDOUT "
    "UNCERTAINTY ANALYSIS"
)
print("=" * 80)


# ============================================================
# Verify frozen protocol
# ============================================================

uncertainty_protocol = protocol[
    "uncertainty_analysis"
]


if (
    uncertainty_protocol[
        "resampling_unit"
    ]
    != "scenario"
):
    raise RuntimeError(
        "Frozen resampling unit "
        "is not scenario."
    )


if (
    int(
        uncertainty_protocol[
            "bootstrap_replicates"
        ]
    )
    != N_BOOTSTRAP
):
    raise RuntimeError(
        "Bootstrap replicate count "
        "differs from frozen protocol."
    )


if (
    int(
        uncertainty_protocol[
            "seed"
        ]
    )
    != SEED
):
    raise RuntimeError(
        "Bootstrap seed differs "
        "from frozen protocol."
    )


if abs(
    float(
        uncertainty_protocol[
            "confidence_level"
        ]
    )
    - CONFIDENCE_LEVEL
) > 1e-12:
    raise RuntimeError(
        "Confidence level differs "
        "from frozen protocol."
    )


if (
    uncertainty_protocol[
        "same_bootstrap_samples_across_models"
    ]
    is not True
):
    raise RuntimeError(
        "Frozen protocol does not require "
        "paired bootstrap samples."
    )


if (
    evaluation_summary[
        "holdout_now_opened"
    ]
    is not True
):
    raise RuntimeError(
        "A8.12.2 final holdout evaluation "
        "has not been completed."
    )


if (
    evaluation_summary[
        "post_holdout_tuning_permitted"
    ]
    is not False
):
    raise RuntimeError(
        "Post-holdout tuning safeguard "
        "is not frozen."
    )


# ============================================================
# Validate holdout prediction dataset
# ============================================================

if len(rows) != 54:
    raise RuntimeError(
        f"Expected 54 holdout cases, "
        f"found {len(rows)}."
    )


case_ids = [
    row["case_id"]
    for row in rows
]


if len(
    set(case_ids)
) != 54:
    raise RuntimeError(
        "Duplicate or missing "
        "holdout case IDs."
    )


case_records = []


for row in rows:

    y = int(
        row["Y_D"]
    )

    if y not in {
        0,
        1,
    }:
        raise RuntimeError(
            f"Unexpected Y_D: {y}"
        )

    me_prediction = float(
        row[
            "M_E_predicted_reliability"
        ]
    )

    meu_prediction = float(
        row[
            "M_EU_predicted_reliability"
        ]
    )

    for model, probability in [
        (
            "M_E",
            me_prediction,
        ),
        (
            "M_EU",
            meu_prediction,
        ),
    ]:

        if not np.isfinite(
            probability
        ):
            raise RuntimeError(
                f"{model}: non-finite "
                "prediction."
            )

        if not (
            0.0
            < probability
            < 1.0
        ):
            raise RuntimeError(
                f"{model}: prediction "
                "outside (0,1)."
            )

    case_records.append({
        "case_id":
            row["case_id"],

        "scenario_id":
            row["scenario_id"],

        "Y_D":
            y,

        "predictions": {
            "M_E":
                me_prediction,

            "M_EU":
                meu_prediction,
        },
    })


y_original = np.asarray(
    [
        record["Y_D"]
        for record in case_records
    ],
    dtype=int,
)


if int(
    np.sum(
        y_original == 1
    )
) != 11:
    raise RuntimeError(
        "Expected 11 positive "
        "holdout outcomes."
    )


if int(
    np.sum(
        y_original == 0
    )
) != 43:
    raise RuntimeError(
        "Expected 43 negative "
        "holdout outcomes."
    )


# ============================================================
# Group eligible cases by scenario
# ============================================================

scenario_to_cases = defaultdict(
    list
)


for record in case_records:
    scenario_to_cases[
        record["scenario_id"]
    ].append(record)


scenario_ids = sorted(
    scenario_to_cases
)

n_scenarios = len(
    scenario_ids
)


if n_scenarios != 15:
    raise RuntimeError(
        f"Expected 15 represented "
        f"holdout scenarios, found "
        f"{n_scenarios}."
    )


print("\nFrozen bootstrap protocol:")
print("  Unit: scenario_id")
print(
    "  Replicates:",
    N_BOOTSTRAP,
)
print(
    "  Seed:",
    SEED,
)
print(
    "  Confidence level:",
    CONFIDENCE_LEVEL,
)

print("\nAnalytical structure:")
print(
    "  Eligible cases:",
    len(case_records),
)
print(
    "  Represented scenarios:",
    n_scenarios,
)
print(
    "  Y_D=0:",
    int(
        np.sum(
            y_original == 0
        )
    ),
)
print(
    "  Y_D=1:",
    int(
        np.sum(
            y_original == 1
        )
    ),
)


# ============================================================
# Original point estimates
# ============================================================

original_metrics = {}


for model in MODEL_ORDER:

    p = np.asarray(
        [
            record[
                "predictions"
            ][
                model
            ]
            for record in case_records
        ],
        dtype=float,
    )

    discrimination = (
        calculate_discrimination(
            y_original,
            p,
        )
    )

    if discrimination is None:
        raise RuntimeError(
            "Original holdout unexpectedly "
            "contains only one outcome class."
        )

    original_metrics[
        model
    ] = {
        "brier_score":
            calculate_brier(
                y_original,
                p,
            ),

        **discrimination,
    }


print("\nOriginal point estimates:")


for model in MODEL_ORDER:

    metrics = original_metrics[
        model
    ]

    print(
        f"  {model}: "
        f"Brier="
        f"{metrics['brier_score']:.6f}, "
        f"AUROC="
        f"{metrics['auroc']:.6f}, "
        f"AUPRC="
        f"{metrics['auprc']:.6f}"
    )


# ============================================================
# Scenario-level bootstrap
# ============================================================

rng = np.random.default_rng(
    SEED
)


bootstrap_results = {
    model: {
        "brier_score": [],
        "auroc": [],
        "auprc": [],
    }
    for model in MODEL_ORDER
}


brier_replicates = 0
discrimination_replicates = 0
single_class_replicates = 0


for replicate in range(
    N_BOOTSTRAP
):

    sampled_scenarios = (
        rng.choice(
            scenario_ids,
            size=n_scenarios,
            replace=True,
        )
    )

    sampled_cases = []

    # Preserve scenario multiplicity.
    for scenario_id in (
        sampled_scenarios
    ):
        sampled_cases.extend(
            scenario_to_cases[
                scenario_id
            ]
        )

    y = np.asarray(
        [
            record["Y_D"]
            for record in sampled_cases
        ],
        dtype=int,
    )

    # Brier is valid even if a bootstrap
    # replicate contains one outcome class.
    for model in MODEL_ORDER:

        p = np.asarray(
            [
                record[
                    "predictions"
                ][
                    model
                ]
                for record
                in sampled_cases
            ],
            dtype=float,
        )

        bootstrap_results[
            model
        ][
            "brier_score"
        ].append(
            calculate_brier(
                y,
                p,
            )
        )

    brier_replicates += 1

    # Discrimination requires both classes.
    if len(
        np.unique(y)
    ) < 2:
        single_class_replicates += 1
        continue

    for model in MODEL_ORDER:

        p = np.asarray(
            [
                record[
                    "predictions"
                ][
                    model
                ]
                for record
                in sampled_cases
            ],
            dtype=float,
        )

        discrimination = (
            calculate_discrimination(
                y,
                p,
            )
        )

        if discrimination is None:
            raise RuntimeError(
                "Unexpected discrimination "
                "metric failure."
            )

        bootstrap_results[
            model
        ][
            "auroc"
        ].append(
            discrimination[
                "auroc"
            ]
        )

        bootstrap_results[
            model
        ][
            "auprc"
        ].append(
            discrimination[
                "auprc"
            ]
        )

    discrimination_replicates += 1


# ============================================================
# Bootstrap integrity checks
# ============================================================

if (
    brier_replicates
    != N_BOOTSTRAP
):
    raise RuntimeError(
        "Not all bootstrap replicates "
        "contributed to Brier."
    )


if (
    discrimination_replicates
    + single_class_replicates
    != N_BOOTSTRAP
):
    raise RuntimeError(
        "Bootstrap replicate accounting "
        "does not sum correctly."
    )


for model in MODEL_ORDER:

    if len(
        bootstrap_results[
            model
        ][
            "brier_score"
        ]
    ) != N_BOOTSTRAP:
        raise RuntimeError(
            f"{model}: incorrect Brier "
            "replicate count."
        )

    if len(
        bootstrap_results[
            model
        ][
            "auroc"
        ]
    ) != discrimination_replicates:
        raise RuntimeError(
            f"{model}: incorrect AUROC "
            "replicate count."
        )

    if len(
        bootstrap_results[
            model
        ][
            "auprc"
        ]
    ) != discrimination_replicates:
        raise RuntimeError(
            f"{model}: incorrect AUPRC "
            "replicate count."
        )


if (
    discrimination_replicates
    == 0
):
    raise RuntimeError(
        "No valid discrimination "
        "replicates."
    )


print("\nBootstrap execution:")
print(
    "  Requested replicates:",
    N_BOOTSTRAP,
)
print(
    "  Valid Brier replicates:",
    brier_replicates,
)
print(
    "  Valid discrimination replicates:",
    discrimination_replicates,
)
print(
    "  Single-class discrimination "
    "replicates:",
    single_class_replicates,
)


# ============================================================
# Metric confidence intervals
# ============================================================

summary_records = []


print("\n" + "=" * 80)
print(
    "HOLDOUT BOOTSTRAP METRIC INTERVALS"
)
print("=" * 80)


for model in MODEL_ORDER:

    print(f"\n{model}")

    for metric_name in [
        "brier_score",
        "auroc",
        "auprc",
    ]:

        values = np.asarray(
            bootstrap_results[
                model
            ][
                metric_name
            ],
            dtype=float,
        )

        lower, upper = (
            percentile_interval(
                values
            )
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

        valid_replicates = (
            brier_replicates
            if metric_name
            == "brier_score"
            else discrimination_replicates
        )

        summary_records.append({
            "model":
                model,

            "metric":
                metric_name,

            "point_estimate":
                point,

            "bootstrap_mean":
                bootstrap_mean,

            "bootstrap_sd":
                bootstrap_sd,

            "ci_lower_95":
                lower,

            "ci_upper_95":
                upper,

            "valid_replicates":
                valid_replicates,
        })

        print(
            f"  {metric_name}: "
            f"point={point:.6f}, "
            f"bootstrap_mean="
            f"{bootstrap_mean:.6f}, "
            f"95% CI="
            f"[{lower:.6f}, "
            f"{upper:.6f}], "
            f"valid="
            f"{valid_replicates}"
        )


# ============================================================
# Paired M_EU minus M_E differences
# ============================================================

paired_records = []


print("\n" + "=" * 80)
print(
    "PAIRED HOLDOUT BOOTSTRAP "
    "DIFFERENCES: M_EU MINUS M_E"
)
print("=" * 80)


for metric_name in [
    "brier_score",
    "auroc",
    "auprc",
]:

    comparison_values = np.asarray(
        bootstrap_results[
            COMPARISON_MODEL
        ][
            metric_name
        ],
        dtype=float,
    )

    reference_values = np.asarray(
        bootstrap_results[
            REFERENCE_MODEL
        ][
            metric_name
        ],
        dtype=float,
    )


    if (
        len(comparison_values)
        != len(reference_values)
    ):
        raise RuntimeError(
            "Paired bootstrap arrays "
            "are not aligned."
        )


    differences = (
        comparison_values
        - reference_values
    )


    lower, upper = (
        percentile_interval(
            differences
        )
    )


    point_difference = (
        original_metrics[
            COMPARISON_MODEL
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
        np.mean(
            differences
        )
    )


    if (
        metric_name
        == "brier_score"
    ):

        favourable_proportion = float(
            np.mean(
                differences < 0.0
            )
        )

        favourable_definition = (
            "Proportion of bootstrap "
            "replicates where M_EU "
            "Brier < M_E Brier"
        )

    else:

        favourable_proportion = float(
            np.mean(
                differences > 0.0
            )
        )

        favourable_definition = (
            "Proportion of bootstrap "
            f"replicates where M_EU "
            f"{metric_name} > M_E "
            f"{metric_name}"
        )


    valid_replicates = (
        brier_replicates
        if metric_name
        == "brier_score"
        else discrimination_replicates
    )


    paired_records.append({
        "reference_model":
            REFERENCE_MODEL,

        "comparison_model":
            COMPARISON_MODEL,

        "metric":
            metric_name,

        "difference_definition":
            "M_EU - M_E",

        "point_difference":
            point_difference,

        "bootstrap_mean_difference":
            mean_difference,

        "ci_lower_95":
            lower,

        "ci_upper_95":
            upper,

        "favourable_proportion":
            favourable_proportion,

        "favourable_definition":
            favourable_definition,

        "valid_replicates":
            valid_replicates,
    })


    print(
        f"\n  {metric_name}:"
    )

    print(
        "    point difference:",
        f"{point_difference:.6f}",
    )

    print(
        "    bootstrap mean difference:",
        f"{mean_difference:.6f}",
    )

    print(
        "    95% CI:",
        f"[{lower:.6f}, "
        f"{upper:.6f}]",
    )

    print(
        "    favourable proportion:",
        f"{favourable_proportion:.4f}",
    )


# ============================================================
# Export metric intervals
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


# ============================================================
# Export paired differences
# ============================================================

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
        "favourable_proportion",
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


# ============================================================
# Durable manifest
# ============================================================

manifest = {
    "stage":
        "A8.12",

    "substage":
        "A8.12.3",

    "name":
        (
            "Scenario-Level Holdout "
            "Uncertainty Analysis"
        ),

    "status":
        "COMPLETE",

    "bootstrap": {
        "unit":
            "scenario_id",

        "represented_scenarios":
            n_scenarios,

        "requested_replicates":
            N_BOOTSTRAP,

        "brier_valid_replicates":
            brier_replicates,

        "discrimination_valid_replicates":
            discrimination_replicates,

        "single_class_discrimination_replicates":
            single_class_replicates,

        "seed":
            SEED,

        "confidence_level":
            CONFIDENCE_LEVEL,

        "interval_method":
            "percentile",

        "cluster_multiplicity_preserved":
            True,

        "same_samples_across_models":
            True,
    },

    "models": [
        "M_E",
        "M_EU",
    ],

    "metrics": [
        "brier_score",
        "auroc",
        "auprc",
    ],

    "paired_reference_model":
        REFERENCE_MODEL,

    "paired_comparison_model":
        COMPARISON_MODEL,

    "difference_orientation":
        "M_EU - M_E",

    "ece_bootstrapped":
        False,

    "ece_reason":
        (
            "ECE remains a supporting "
            "binned calibration measure "
            "and was not pre-specified "
            "for scenario-level bootstrap."
        ),

    "model_selection_performed":
        False,

    "parameter_refitting_performed":
        False,

    "threshold_selection_performed":
        False,

    "post_holdout_tuning_performed":
        False,

    "inputs": {
        str(PREDICTIONS_FILE):
            sha256(
                PREDICTIONS_FILE
            ),

        str(EVALUATION_SUMMARY_FILE):
            sha256(
                EVALUATION_SUMMARY_FILE
            ),

        str(PROTOCOL_FILE):
            sha256(
                PROTOCOL_FILE
            ),
    },

    "outputs": {
        "metric_intervals":
            str(
                BOOTSTRAP_SUMMARY_FILE
            ),

        "paired_differences":
            str(
                PAIRED_DIFFERENCES_FILE
            ),
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


# ============================================================
# Report
# ============================================================

print("\n" + "=" * 80)
print("OUTPUT ARTEFACTS")
print("=" * 80)

print(
    BOOTSTRAP_SUMMARY_FILE
)
print(
    PAIRED_DIFFERENCES_FILE
)
print(
    BOOTSTRAP_MANIFEST_FILE
)

print("\nValidation:")
print(
    "Frozen A8.12 protocol respected: PASS"
)
print(
    "Scenario-level resampling: PASS"
)
print(
    "Scenario multiplicity preserved: PASS"
)
print(
    "Same samples used across models: PASS"
)
print(
    "All 2,000 replicates used for Brier: PASS"
)
print(
    "Single-class discrimination "
    "replicates separately handled: PASS"
)
print(
    "Metric intervals generated: PASS"
)
print(
    "Paired M_EU vs M_E differences "
    "generated: PASS"
)
print(
    "No model refitting: PASS"
)
print(
    "No threshold selection: PASS"
)
print(
    "No post-holdout tuning: PASS"
)

print(
    "\nA8.12.3 SCENARIO-LEVEL "
    "HOLDOUT UNCERTAINTY ANALYSIS: "
    "COMPLETE"
)