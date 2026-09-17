import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
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

METRICS_FILE = (
    A8_8_DIR
    / "a8_8_oof_model_performance_metrics.csv"
)

CALIBRATION_FILE = (
    A8_8_DIR
    / "a8_8_oof_reliability_calibration_bins.csv"
)

SUMMARY_FILE = (
    A8_8_DIR
    / "a8_8_oof_evaluation_summary.json"
)


MODEL_ORDER = [
    "M_E",
    "M_C",
    "M_H",
    "M_U",
    "M_EU",
]


# ============================================================
# Utilities
# ============================================================

def load_csv(path):
    with path.open(
        newline="",
        encoding="utf-8",
    ) as f:
        return list(csv.DictReader(f))


def quantile_calibration_bins(y, p, n_bins=5):
    """
    Construct approximately equal-frequency calibration bins.

    Sorting is deterministic. The observations are split using
    numpy.array_split so that every observation belongs to
    exactly one bin.

    This avoids empty bins caused by the very different
    probability ranges of the five candidate models.
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
                f"Calibration bin {bin_number} is empty."
            )

        y_bin = y[indices]
        p_bin = p[indices]

        records.append(
            {
                "bin": bin_number,
                "n": int(len(indices)),
                "mean_predicted_reliability":
                    float(np.mean(p_bin)),
                "observed_correctness":
                    float(np.mean(y_bin)),
                "min_predicted_reliability":
                    float(np.min(p_bin)),
                "max_predicted_reliability":
                    float(np.max(p_bin)),
            }
        )

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


# ============================================================
# Load frozen inputs
# ============================================================

rows = load_csv(PREDICTIONS_FILE)

with PROTOCOL_FILE.open(
    encoding="utf-8",
) as f:
    protocol = json.load(f)


print("=" * 78)
print("A8.8.2 OOF PERFORMANCE AND CALIBRATION EVALUATION")
print("=" * 78)


# ============================================================
# Input integrity
# ============================================================

if len(rows) != 1190:
    raise RuntimeError(
        f"Expected 1,190 OOF prediction rows, "
        f"found {len(rows)}."
    )

model_counts = Counter(
    row["model"]
    for row in rows
)

for model in MODEL_ORDER:
    if model_counts[model] != 238:
        raise RuntimeError(
            f"{model}: expected 238 predictions, "
            f"found {model_counts[model]}."
        )


if (
    protocol[
        "comparison_policy"
    ][
        "no_holdout_use"
    ]
    is not True
):
    raise RuntimeError(
        "Frozen evaluation protocol does not "
        "exclude the holdout."
    )


print("\nInput validation:")
print("1,190 OOF predictions: PASS")
print("238 predictions per model: PASS")
print("Frozen evaluation protocol loaded: PASS")
print("Holdout exclusion policy confirmed: PASS")


# ============================================================
# Evaluate each model
# ============================================================

metric_records = []
calibration_records = []

print("\n" + "=" * 78)
print("MODEL RESULTS")
print("=" * 78)


for model in MODEL_ORDER:

    model_rows = [
        row
        for row in rows
        if row["model"] == model
    ]

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

    case_ids = [
        row["case_id"]
        for row in model_rows
    ]

    if len(set(case_ids)) != 238:
        raise RuntimeError(
            f"{model}: duplicate or missing cases."
        )

    if not np.isfinite(p).all():
        raise RuntimeError(
            f"{model}: non-finite probabilities."
        )

    if np.any(
        (p <= 0.0)
        | (p >= 1.0)
    ):
        raise RuntimeError(
            f"{model}: probability outside (0,1)."
        )

    if set(np.unique(y)) != {0, 1}:
        raise RuntimeError(
            f"{model}: both Y_D classes required."
        )

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
        n_bins=5,
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

    metric_records.append(
        {
            "model": model,
            "n": len(y),
            "Y_D_0": int(
                np.sum(y == 0)
            ),
            "Y_D_1": int(
                np.sum(y == 1)
            ),
            "observed_correctness_rate":
                observed_rate,
            "mean_predicted_reliability":
                mean_probability,
            "brier_score": brier,
            "auroc": auroc,
            "auprc": auprc,
            "ece_5_quantile_bins": ece,
        }
    )

    for record in bins:
        calibration_records.append(
            {
                "model": model,
                **record,
            }
        )

    print(f"\n{model}")
    print("-" * 40)

    print(
        f"N: {len(y)} "
        f"(Y_D=0: {np.sum(y == 0)}, "
        f"Y_D=1: {np.sum(y == 1)})"
    )

    print(
        "Observed correctness rate:",
        f"{observed_rate:.6f}",
    )

    print(
        "Mean predicted reliability:",
        f"{mean_probability:.6f}",
    )

    print(
        "Brier score:",
        f"{brier:.6f}",
    )

    print(
        "AUROC:",
        f"{auroc:.6f}",
    )

    print(
        "AUPRC:",
        f"{auprc:.6f}",
    )

    print(
        "ECE (5 quantile bins):",
        f"{ece:.6f}",
    )

    print("Calibration bins:")

    for record in bins:
        print(
            f"  Bin {record['bin']}: "
            f"n={record['n']}, "
            f"mean_p="
            f"{record['mean_predicted_reliability']:.6f}, "
            f"observed="
            f"{record['observed_correctness']:.6f}, "
            f"range="
            f"{record['min_predicted_reliability']:.6f}"
            f" to "
            f"{record['max_predicted_reliability']:.6f}"
        )


# ============================================================
# Cross-model outcome consistency
# ============================================================

print("\n" + "=" * 78)
print("CROSS-MODEL CONSISTENCY")
print("=" * 78)

reference = {
    row["case_id"]: int(row["Y_D"])
    for row in rows
    if row["model"] == "M_E"
}

for model in MODEL_ORDER:

    current = {
        row["case_id"]: int(row["Y_D"])
        for row in rows
        if row["model"] == model
    }

    if current != reference:
        raise RuntimeError(
            f"{model}: case/outcome set differs "
            "from reference."
        )

    print(
        f"{model}: identical 238-case "
        "outcome set PASS"
    )


# ============================================================
# Write artefacts
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
    writer.writerows(metric_records)


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


summary = {
    "stage": "A8.8",
    "substage": "A8.8.2",
    "name": (
        "OOF Performance and "
        "Calibration Point Estimates"
    ),
    "evaluation_cases": 238,
    "candidate_models": MODEL_ORDER,
    "metrics": metric_records,
    "calibration_bins": (
        "5 deterministic equal-frequency "
        "bins per model"
    ),
    "holdout_used": False,
    "model_selection_performed": False,
    "threshold_selection_performed": False,
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


print("\n" + "=" * 78)
print("OUTPUT ARTEFACTS")
print("=" * 78)

print(METRICS_FILE)
print(CALIBRATION_FILE)
print(SUMMARY_FILE)

print("\nValidation:")
print("Five models evaluated: PASS")
print("Identical Y_D outcome set: PASS")
print("Brier score calculated: PASS")
print("AUROC calculated: PASS")
print("AUPRC calculated: PASS")
print("5-bin reliability analysis calculated: PASS")
print("ECE calculated: PASS")
print("Holdout untouched: PASS")
print("No model selection performed: PASS")

print(
    "\nA8.8.2 OOF PERFORMANCE AND "
    "CALIBRATION EVALUATION: PASS"
)