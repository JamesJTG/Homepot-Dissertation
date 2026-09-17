import csv
import hashlib
import json
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

CANDIDATE_MODEL = "M_EU"
BENCHMARK_MODEL = "M_E"

MODEL_ORDER = [
    BENCHMARK_MODEL,
    CANDIDATE_MODEL,
]

OUTCOME = "diagnostic_correctness"

N_CALIBRATION_BINS = 5


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

PARTITION_FILE = (
    OUTPUTS
    / "a8_7_reliability_modelling"
    / "a8_7_scenario_development_holdout_assignment.csv"
)

A8_11_PARAMETERS_FILE = (
    OUTPUTS
    / "a8_11_homepot_integration"
    / "a8_11_final_development_meu_parameters.json"
)

A8_12_DIR = (
    OUTPUTS
    / "a8_12_feature_level_evaluation"
)

BENCHMARK_PARAMETERS_FILE = (
    A8_12_DIR
    / "a8_12_final_development_me_parameters.json"
)

PROTOCOL_FILE = (
    A8_12_DIR
    / "a8_12_pre_holdout_evaluation_protocol.json"
)

PREDICTIONS_FILE = (
    A8_12_DIR
    / "a8_12_holdout_estimator_predictions.csv"
)

METRICS_FILE = (
    A8_12_DIR
    / "a8_12_holdout_estimator_performance_metrics.csv"
)

CALIBRATION_FILE = (
    A8_12_DIR
    / "a8_12_holdout_reliability_calibration_bins.csv"
)

SUMMARY_FILE = (
    A8_12_DIR
    / "a8_12_holdout_estimator_evaluation_summary.json"
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


def quantile_calibration_bins(
    y,
    p,
    n_bins=5,
):
    """
    Frozen A8.8 calibration-bin procedure.

    Sort probabilities deterministically using a stable
    sort and divide observations into approximately
    equal-frequency bins with numpy.array_split.
    """

    order = np.argsort(
        p,
        kind="stable",
    )

    groups = np.array_split(
        order,
        n_bins,
    )

    records = []

    for bin_number, indices in enumerate(
        groups,
        start=1,
    ):
        if len(indices) == 0:
            raise RuntimeError(
                f"Calibration bin "
                f"{bin_number} is empty."
            )

        y_bin = y[indices]
        p_bin = p[indices]

        records.append({
            "bin":
                bin_number,

            "n":
                int(
                    len(indices)
                ),

            "mean_predicted_reliability":
                float(
                    np.mean(p_bin)
                ),

            "observed_correctness":
                float(
                    np.mean(y_bin)
                ),

            "min_predicted_reliability":
                float(
                    np.min(p_bin)
                ),

            "max_predicted_reliability":
                float(
                    np.max(p_bin)
                ),
        })

    return records


def expected_calibration_error(
    calibration_bins,
    total_n,
):
    return float(
        sum(
            (
                row["n"]
                / total_n
            )
            * abs(
                row[
                    "observed_correctness"
                ]
                - row[
                    "mean_predicted_reliability"
                ]
            )
            for row in calibration_bins
        )
    )


def get_coefficient(
    parameters,
    name,
):
    matches = [
        float(
            predictor[
                "coefficient"
            ]
        )
        for predictor
        in parameters[
            "predictors"
        ]
        if predictor[
            "name"
        ] == name
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one "
            f"coefficient for {name}."
        )

    return matches[0]


# ============================================================
# Load and validate frozen inputs
# ============================================================

required_files = [
    A8_6_FILE,
    PARTITION_FILE,
    A8_11_PARAMETERS_FILE,
    BENCHMARK_PARAMETERS_FILE,
    PROTOCOL_FILE,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(
            path
        )


with A8_11_PARAMETERS_FILE.open(
    encoding="utf-8",
) as f:
    meu_parameters = json.load(f)


with BENCHMARK_PARAMETERS_FILE.open(
    encoding="utf-8",
) as f:
    me_parameters = json.load(f)


with PROTOCOL_FILE.open(
    encoding="utf-8",
) as f:
    protocol = json.load(f)


if (
    protocol[
        "holdout_performance_seen"
    ]
    is not False
):
    raise RuntimeError(
        "Pre-holdout protocol does not "
        "record an unseen holdout."
    )


if (
    protocol[
        "candidate_model"
    ][
        "model"
    ]
    != CANDIDATE_MODEL
):
    raise RuntimeError(
        "Unexpected candidate model."
    )


if (
    protocol[
        "primary_benchmark"
    ][
        "model"
    ]
    != BENCHMARK_MODEL
):
    raise RuntimeError(
        "Unexpected primary benchmark."
    )


if (
    meu_parameters["model"]
    != CANDIDATE_MODEL
):
    raise RuntimeError(
        "Unexpected M_EU parameter file."
    )


if (
    me_parameters["model"]
    != BENCHMARK_MODEL
):
    raise RuntimeError(
        "Unexpected M_E parameter file."
    )


if (
    meu_parameters[
        "holdout"
    ][
        "used_for_evaluation"
    ]
    is not False
):
    raise RuntimeError(
        "M_EU parameter artefact does not "
        "record untouched holdout."
    )


if (
    me_parameters[
        "holdout"
    ][
        "used_for_evaluation"
    ]
    is not False
):
    raise RuntimeError(
        "M_E parameter artefact does not "
        "record untouched holdout."
    )


# ============================================================
# Recover frozen coefficients
# ============================================================

me_intercept = float(
    me_parameters[
        "flic_intercept"
    ]
)

me_beta_E = get_coefficient(
    me_parameters,
    "E",
)


meu_intercept = float(
    meu_parameters[
        "flic_intercept"
    ]
)

meu_beta_E = get_coefficient(
    meu_parameters,
    "E",
)

meu_beta_C = get_coefficient(
    meu_parameters,
    "mean_self_confidence",
)

meu_beta_H = get_coefficient(
    meu_parameters,
    "diagnostic_entropy",
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
        "overlap detected."
    )


# ============================================================
# Construct final holdout evaluation population
# ============================================================

rows = load_csv(
    A8_6_FILE
)

holdout_rows = []


for row in rows:

    scenario_id = (
        row["scenario_id"]
    )

    if (
        scenario_id
        not in holdout_scenarios
    ):
        continue

    y_raw = (
        row[
            OUTCOME
        ]
        .strip()
    )

    # Frozen primary Y_D population:
    # UNKNOWN/non-substantive outcomes remain excluded.
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

    confidence = float(
        row[
            "mean_self_confidence"
        ]
    )

    entropy = float(
        row[
            "diagnostic_entropy"
        ]
    )

    if not (
        0.0
        <= confidence
        <= 1.0
    ):
        raise RuntimeError(
            "Confidence outside [0,1]."
        )

    if not (
        0.0
        <= entropy
        <= 1.0
    ):
        raise RuntimeError(
            "Entropy outside [0,1]."
        )

    holdout_rows.append({
        "case_id":
            row["case_id"],

        "scenario_id":
            scenario_id,

        "evidence_condition":
            row[
                "evidence_condition"
            ],

        "ground_truth":
            row[
                "ground_truth"
            ],

        "E":
            E,

        "mean_self_confidence":
            confidence,

        "diagnostic_entropy":
            entropy,

        "Y_D":
            y,
    })


# ============================================================
# Holdout integrity checks
# ============================================================

if len(
    holdout_rows
) != 54:
    raise RuntimeError(
        f"Expected 54 eligible "
        f"holdout cases, found "
        f"{len(holdout_rows)}."
    )


represented_scenarios = {
    row["scenario_id"]
    for row in holdout_rows
}


if len(
    represented_scenarios
) != 15:
    raise RuntimeError(
        "Expected 15 represented "
        "Y_D-eligible holdout scenarios."
    )


if (
    represented_scenarios
    & development_scenarios
):
    raise RuntimeError(
        "Development contamination "
        "detected."
    )


y = np.asarray(
    [
        row["Y_D"]
        for row in holdout_rows
    ],
    dtype=int,
)


if int(
    np.sum(y == 1)
) != 11:
    raise RuntimeError(
        "Expected 11 correct "
        "holdout outcomes."
    )


if int(
    np.sum(y == 0)
) != 43:
    raise RuntimeError(
        "Expected 43 incorrect "
        "holdout outcomes."
    )


E = np.asarray(
    [
        row["E"]
        for row in holdout_rows
    ],
    dtype=float,
)

C = np.asarray(
    [
        row[
            "mean_self_confidence"
        ]
        for row in holdout_rows
    ],
    dtype=float,
)

H = np.asarray(
    [
        row[
            "diagnostic_entropy"
        ]
        for row in holdout_rows
    ],
    dtype=float,
)


# ============================================================
# Apply frozen models exactly once
# ============================================================

predictions = {
    BENCHMARK_MODEL:
        logistic(
            me_intercept
            + me_beta_E * E
        ),

    CANDIDATE_MODEL:
        logistic(
            meu_intercept
            + meu_beta_E * E
            + meu_beta_C * C
            + meu_beta_H * H
        ),
}


for model, p in predictions.items():

    if not np.all(
        np.isfinite(p)
    ):
        raise RuntimeError(
            f"{model}: non-finite "
            "probabilities."
        )

    if np.any(
        (p <= 0.0)
        | (p >= 1.0)
    ):
        raise RuntimeError(
            f"{model}: probabilities "
            "outside (0,1)."
        )


# ============================================================
# Evaluate frozen estimators
# ============================================================

metric_records = []
calibration_records = []


print("=" * 80)
print(
    "A8.12.2 FINAL HOLDOUT "
    "ESTIMATOR EVALUATION"
)
print("=" * 80)

print("\nHoldout population:")
print(
    "  Eligible cases:",
    len(holdout_rows),
)
print(
    "  Represented scenarios:",
    len(represented_scenarios),
)
print(
    "  Correct:",
    int(
        np.sum(y == 1)
    ),
)
print(
    "  Incorrect:",
    int(
        np.sum(y == 0)
    ),
)
print(
    "  Observed correctness:",
    f"{np.mean(y):.6f}",
)


for model in MODEL_ORDER:

    p = predictions[
        model
    ]

    brier = float(
        brier_score_loss(
            y,
            p,
        )
    )

    auroc = float(
        roc_auc_score(
            y,
            p,
        )
    )

    auprc = float(
        average_precision_score(
            y,
            p,
        )
    )

    bins = quantile_calibration_bins(
        y,
        p,
        n_bins=N_CALIBRATION_BINS,
    )

    ece = expected_calibration_error(
        bins,
        len(y),
    )

    mean_probability = float(
        np.mean(p)
    )

    observed_rate = float(
        np.mean(y)
    )

    metric_records.append({
        "model":
            model,

        "n":
            len(y),

        "Y_D_0":
            int(
                np.sum(y == 0)
            ),

        "Y_D_1":
            int(
                np.sum(y == 1)
            ),

        "observed_correctness_rate":
            observed_rate,

        "mean_predicted_reliability":
            mean_probability,

        "brier_score":
            brier,

        "auroc":
            auroc,

        "auprc":
            auprc,

        "ece_5_quantile_bins":
            ece,
    })

    for record in bins:
        calibration_records.append({
            "model":
                model,

            **record,
        })

    print(f"\n{model}")
    print("-" * 40)

    print(
        "  Mean predicted reliability:",
        f"{mean_probability:.6f}",
    )

    print(
        "  Brier score:",
        f"{brier:.6f}",
    )

    print(
        "  AUROC:",
        f"{auroc:.6f}",
    )

    print(
        "  AUPRC:",
        f"{auprc:.6f}",
    )

    print(
        "  ECE (5 quantile bins):",
        f"{ece:.6f}",
    )

    print("  Calibration bins:")

    for record in bins:
        print(
            f"    Bin {record['bin']}: "
            f"n={record['n']}, "
            f"mean_p="
            f"{record['mean_predicted_reliability']:.6f}, "
            f"observed="
            f"{record['observed_correctness']:.6f}"
        )


# ============================================================
# Paired point-estimate differences
# ============================================================

metrics_by_model = {
    row["model"]:
        row
    for row in metric_records
}


me_metrics = metrics_by_model[
    BENCHMARK_MODEL
]

meu_metrics = metrics_by_model[
    CANDIDATE_MODEL
]


paired_differences = {
    "comparison":
        "M_EU minus M_E",

    "brier_score_difference":
        float(
            meu_metrics[
                "brier_score"
            ]
            - me_metrics[
                "brier_score"
            ]
        ),

    "auroc_difference":
        float(
            meu_metrics[
                "auroc"
            ]
            - me_metrics[
                "auroc"
            ]
        ),

    "auprc_difference":
        float(
            meu_metrics[
                "auprc"
            ]
            - me_metrics[
                "auprc"
            ]
        ),

    "ece_difference":
        float(
            meu_metrics[
                "ece_5_quantile_bins"
            ]
            - me_metrics[
                "ece_5_quantile_bins"
            ]
        ),
}


print(
    "\nM_EU minus M_E point-estimate "
    "differences:"
)

print(
    "  Brier:",
    f"{paired_differences['brier_score_difference']:.6f}",
)

print(
    "  AUROC:",
    f"{paired_differences['auroc_difference']:.6f}",
)

print(
    "  AUPRC:",
    f"{paired_differences['auprc_difference']:.6f}",
)

print(
    "  ECE:",
    f"{paired_differences['ece_difference']:.6f}",
)


# ============================================================
# Export case-level predictions
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
        "ground_truth",
        "E",
        "mean_self_confidence",
        "diagnostic_entropy",
        "Y_D",
        "M_E_predicted_reliability",
        "M_EU_predicted_reliability",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    for index, row in enumerate(
        holdout_rows
    ):
        writer.writerow({
            **row,

            "M_E_predicted_reliability":
                float(
                    predictions[
                        BENCHMARK_MODEL
                    ][index]
                ),

            "M_EU_predicted_reliability":
                float(
                    predictions[
                        CANDIDATE_MODEL
                    ][index]
                ),
        })


# ============================================================
# Export metrics
# ============================================================

with METRICS_FILE.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    fieldnames = [
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

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(
        metric_records
    )


# ============================================================
# Export calibration bins
# ============================================================

with CALIBRATION_FILE.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    fieldnames = [
        "model",
        "bin",
        "n",
        "mean_predicted_reliability",
        "observed_correctness",
        "min_predicted_reliability",
        "max_predicted_reliability",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(
        calibration_records
    )


# ============================================================
# Durable evaluation summary
# ============================================================

summary = {
    "stage":
        "A8.12.2",

    "status":
        "FINAL_HOLDOUT_ESTIMATOR_EVALUATION_COMPLETE",

    "evaluation_type":
        "one-shot frozen holdout evaluation",

    "holdout_now_opened":
        True,

    "holdout_population": {
        "eligible_cases":
            len(
                holdout_rows
            ),

        "represented_scenarios":
            len(
                represented_scenarios
            ),

        "Y_D_0":
            int(
                np.sum(y == 0)
            ),

        "Y_D_1":
            int(
                np.sum(y == 1)
            ),

        "observed_correctness_rate":
            float(
                np.mean(y)
            ),
    },

    "models": {
        row["model"]:
            {
                key:
                    value
                for key, value
                in row.items()
                if key != "model"
            }
        for row in metric_records
    },

    "paired_point_estimate_differences":
        paired_differences,

    "primary_metric":
        "Brier score",

    "supporting_metrics": [
        "AUROC",
        "AUPRC",
        "ECE",
        "5-bin reliability analysis",
    ],

    "frozen_before_holdout": {
        "M_E_parameters":
            True,

        "M_EU_parameters":
            True,

        "issue_threshold_0_64":
            True,

        "evaluation_protocol":
            True,
    },

    "post_holdout_tuning_permitted":
        False,

    "inputs": {
        str(A8_6_FILE):
            sha256(
                A8_6_FILE
            ),

        str(PARTITION_FILE):
            sha256(
                PARTITION_FILE
            ),

        str(A8_11_PARAMETERS_FILE):
            sha256(
                A8_11_PARAMETERS_FILE
            ),

        str(BENCHMARK_PARAMETERS_FILE):
            sha256(
                BENCHMARK_PARAMETERS_FILE
            ),

        str(PROTOCOL_FILE):
            sha256(
                PROTOCOL_FILE
            ),
    },
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


print("\nOutput artefacts:")
print(PREDICTIONS_FILE)
print(METRICS_FILE)
print(CALIBRATION_FILE)
print(SUMMARY_FILE)

print("\nValidation:")
print("Frozen M_E parameters loaded: PASS")
print("Frozen M_EU parameters loaded: PASS")
print("Pre-holdout protocol loaded: PASS")
print("30-scenario holdout partition retained: PASS")
print("54 primary eligible cases: PASS")
print("11 correct / 43 incorrect: PASS")
print("15 represented eligible scenarios: PASS")
print("No parameter fitting on holdout: PASS")
print("No threshold selection on holdout: PASS")
print("Brier remains primary metric: PASS")
print("Holdout is now OPENED: YES")

print(
    "\nA8.12.2 FINAL HOLDOUT ESTIMATOR "
    "EVALUATION: COMPLETE"
)