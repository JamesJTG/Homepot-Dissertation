import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


# ============================================================
# Frozen configuration
# ============================================================

MODEL = "M_EU"

BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260917
CONFIDENCE_LEVEL = 0.95

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

OUTPUT_FILE = (
    A8_10_DIR
    / "a8_10_scenario_bootstrap_policy_operating_points.csv"
)

SUMMARY_FILE = (
    A8_10_DIR
    / "a8_10_scenario_bootstrap_policy_operating_points.json"
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


def percentile_interval(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    lower = 100.0 * (
        1.0 - CONFIDENCE_LEVEL
    ) / 2.0

    upper = 100.0 - lower

    return (
        float(
            np.percentile(
                values,
                lower,
            )
        ),
        float(
            np.percentile(
                values,
                upper,
            )
        ),
    )


# ============================================================
# Validate frozen selection
# ============================================================

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
    != MODEL
):
    raise RuntimeError(
        "Frozen selected model is not M_EU."
    )


# ============================================================
# Load OOF predictions
# ============================================================

rows = load_csv(
    PREDICTIONS_FILE
)

model_rows = [
    row
    for row in rows
    if row["model"] == MODEL
]


if len(model_rows) != 238:
    raise RuntimeError(
        f"Expected 238 {MODEL} OOF rows, "
        f"found {len(model_rows)}."
    )


# ============================================================
# Parse analytical records
# ============================================================

records = []

for row in model_rows:

    record = {
        "case_id":
            row["case_id"],

        "scenario_id":
            row["scenario_id"],

        "Y_D":
            int(row["Y_D"]),

        "predicted_reliability":
            float(
                row[
                    "predicted_reliability"
                ]
            ),
    }

    records.append(record)


scenario_groups = defaultdict(list)

for record in records:
    scenario_groups[
        record["scenario_id"]
    ].append(record)


scenario_ids = sorted(
    scenario_groups.keys()
)


if len(scenario_ids) != 68:
    raise RuntimeError(
        f"Expected 68 represented "
        f"development scenarios, "
        f"found {len(scenario_ids)}."
    )


# ============================================================
# Identify thresholds that change acceptance set
# ============================================================

original_profiles = {}

previous_case_set = None
change_thresholds = []


for threshold in THRESHOLDS:

    accepted = [
        r
        for r in records
        if (
            r["predicted_reliability"]
            >= threshold
        )
    ]

    accepted_case_set = frozenset(
        r["case_id"]
        for r in accepted
    )

    n_accepted = len(accepted)

    coverage = (
        n_accepted
        / len(records)
    )

    if n_accepted > 0:

        n_correct = sum(
            r["Y_D"]
            for r in accepted
        )

        accepted_accuracy = (
            n_correct
            / n_accepted
        )

        selective_risk = (
            1.0
            - accepted_accuracy
        )

    else:

        n_correct = 0
        accepted_accuracy = None
        selective_risk = None

    original_profiles[
        float(threshold)
    ] = {
        "n_accepted":
            n_accepted,

        "coverage":
            coverage,

        "accepted_accuracy":
            accepted_accuracy,

        "selective_risk":
            selective_risk,
    }

    if (
        accepted_case_set
        != previous_case_set
    ):
        change_thresholds.append(
            float(threshold)
        )

    previous_case_set = (
        accepted_case_set
    )


# ============================================================
# Bootstrap whole scenarios
# ============================================================

rng = np.random.default_rng(
    BOOTSTRAP_SEED
)


bootstrap_values = {
    float(threshold): {
        "coverage": [],
        "accepted_accuracy": [],
        "selective_risk": [],
    }
    for threshold in THRESHOLDS
}


valid_replicates = 0


for _ in range(
    BOOTSTRAP_REPLICATES
):

    sampled_scenario_ids = rng.choice(
        scenario_ids,
        size=len(scenario_ids),
        replace=True,
    )

    # IMPORTANT:
    # preserve scenario multiplicity.
    sampled_records = []

    for scenario_id in sampled_scenario_ids:
        sampled_records.extend(
            scenario_groups[
                scenario_id
            ]
        )

    n_total = len(
        sampled_records
    )

    if n_total == 0:
        continue

    valid_replicates += 1

    y = np.array(
        [
            r["Y_D"]
            for r in sampled_records
        ],
        dtype=int,
    )

    p = np.array(
        [
            r["predicted_reliability"]
            for r in sampled_records
        ],
        dtype=float,
    )

    for threshold in THRESHOLDS:

        threshold = float(
            threshold
        )

        accepted = (
            p >= threshold
        )

        n_accepted = int(
            np.sum(
                accepted
            )
        )

        coverage = (
            n_accepted
            / n_total
        )

        bootstrap_values[
            threshold
        ][
            "coverage"
        ].append(
            coverage
        )

        if n_accepted > 0:

            accepted_accuracy = float(
                np.mean(
                    y[
                        accepted
                    ]
                )
            )

            selective_risk = (
                1.0
                - accepted_accuracy
            )

            bootstrap_values[
                threshold
            ][
                "accepted_accuracy"
            ].append(
                accepted_accuracy
            )

            bootstrap_values[
                threshold
            ][
                "selective_risk"
            ].append(
                selective_risk
            )


if valid_replicates != BOOTSTRAP_REPLICATES:
    raise RuntimeError(
        "Unexpected invalid bootstrap "
        "replicates."
    )


# ============================================================
# Construct output
# ============================================================

output_records = []


for threshold in THRESHOLDS:

    threshold = float(
        threshold
    )

    original = (
        original_profiles[
            threshold
        ]
    )

    coverage_values = (
        bootstrap_values[
            threshold
        ][
            "coverage"
        ]
    )

    accuracy_values = (
        bootstrap_values[
            threshold
        ][
            "accepted_accuracy"
        ]
    )

    risk_values = (
        bootstrap_values[
            threshold
        ][
            "selective_risk"
        ]
    )

    coverage_low, coverage_high = (
        percentile_interval(
            coverage_values
        )
    )

    if len(
        accuracy_values
    ) > 0:

        accuracy_low, accuracy_high = (
            percentile_interval(
                accuracy_values
            )
        )

        risk_low, risk_high = (
            percentile_interval(
                risk_values
            )
        )

        bootstrap_nonempty = len(
            accuracy_values
        )

    else:

        accuracy_low = None
        accuracy_high = None
        risk_low = None
        risk_high = None
        bootstrap_nonempty = 0

    output_records.append({
        "model":
            MODEL,

        "threshold":
            threshold,

        "acceptance_set_changes":
            (
                threshold
                in change_thresholds
            ),

        "n_accepted_original":
            original[
                "n_accepted"
            ],

        "coverage_original":
            original[
                "coverage"
            ],

        "coverage_ci_low":
            coverage_low,

        "coverage_ci_high":
            coverage_high,

        "accepted_accuracy_original":
            original[
                "accepted_accuracy"
            ],

        "accepted_accuracy_ci_low":
            accuracy_low,

        "accepted_accuracy_ci_high":
            accuracy_high,

        "selective_risk_original":
            original[
                "selective_risk"
            ],

        "selective_risk_ci_low":
            risk_low,

        "selective_risk_ci_high":
            risk_high,

        "bootstrap_nonempty_replicates":
            bootstrap_nonempty,
    })


# ============================================================
# Write CSV
# ============================================================

with OUTPUT_FILE.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    fieldnames = [
        "model",
        "threshold",
        "acceptance_set_changes",
        "n_accepted_original",
        "coverage_original",
        "coverage_ci_low",
        "coverage_ci_high",
        "accepted_accuracy_original",
        "accepted_accuracy_ci_low",
        "accepted_accuracy_ci_high",
        "selective_risk_original",
        "selective_risk_ci_low",
        "selective_risk_ci_high",
        "bootstrap_nonempty_replicates",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(
        output_records
    )


# ============================================================
# Print candidate operating region
# ============================================================

print("=" * 80)
print("A8.10.2 SCENARIO-LEVEL POLICY OPERATING-POINT ANALYSIS")
print("=" * 80)

print("\nModel:", MODEL)
print("Development OOF cases:", len(records))
print("Represented scenarios:", len(scenario_ids))
print(
    "Bootstrap replicates:",
    valid_replicates,
)
print("Holdout used: NO")


print("\n" + "=" * 80)
print("DESCRIPTIVE OPERATING POINTS")
print("=" * 80)


for target in [
    0.10,
    0.25,
    0.50,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
]:

    record = next(
        r
        for r in output_records
        if abs(
            r["threshold"]
            - target
        ) < 1e-12
    )

    print(
        f"\nt={target:.2f}"
    )

    print(
        "  Accepted:",
        record[
            "n_accepted_original"
        ],
    )

    print(
        "  Coverage:",
        f"{record['coverage_original']:.6f}",
        "95% CI=",
        (
            f"[{record['coverage_ci_low']:.6f}, "
            f"{record['coverage_ci_high']:.6f}]"
        ),
    )

    if (
        record[
            "accepted_accuracy_original"
        ]
        is None
    ):

        print(
            "  Accepted accuracy: NA"
        )

        print(
            "  Selective risk: NA"
        )

    else:

        print(
            "  Accepted accuracy:",
            (
                f"{record['accepted_accuracy_original']:.6f}"
            ),
            "95% CI=",
            (
                f"[{record['accepted_accuracy_ci_low']:.6f}, "
                f"{record['accepted_accuracy_ci_high']:.6f}]"
            ),
        )

        print(
            "  Selective risk:",
            (
                f"{record['selective_risk_original']:.6f}"
            ),
            "95% CI=",
            (
                f"[{record['selective_risk_ci_low']:.6f}, "
                f"{record['selective_risk_ci_high']:.6f}]"
            ),
        )


# ============================================================
# Durable summary
# ============================================================

summary = {
    "stage": "A8.10",
    "substage": "A8.10.2",
    "name": (
        "Scenario-Level Policy "
        "Operating-Point Analysis"
    ),
    "status":
        "COMPLETED_AFTER_VALIDATION",

    "model":
        MODEL,

    "data": {
        "type":
            "development OOF predictions",

        "eligible_cases":
            len(records),

        "represented_scenarios":
            len(scenario_ids),

        "holdout_used":
            False,
    },

    "bootstrap": {
        "unit":
            "scenario_id",

        "replicates":
            BOOTSTRAP_REPLICATES,

        "valid_replicates":
            valid_replicates,

        "seed":
            BOOTSTRAP_SEED,

        "confidence_level":
            CONFIDENCE_LEVEL,

        "scenario_multiplicity_preserved":
            True,
    },

    "analysis": {
        "threshold_grid":
            "0.00 to 1.00 by 0.01",

        "metrics": [
            "coverage",
            "accepted_accuracy",
            "selective_risk",
        ],

        "acceptance_set_change_thresholds":
            change_thresholds,
    },

    "interpretation_constraints": [
        (
            "This analysis characterises "
            "candidate operating regions."
        ),
        (
            "No Issue, Qualify or Withhold "
            "threshold is selected here."
        ),
        (
            "Observed maximum accuracy is not "
            "used mechanically as a threshold."
        ),
        (
            "Uncertainty is estimated by "
            "resampling whole scenarios."
        ),
        (
            "The final holdout remains untouched."
        ),
    ],

    "thresholds_selected":
        False,

    "holdout_used":
        False,
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


# ============================================================
# Validation
# ============================================================

print("\n" + "=" * 80)
print("OUTPUT ARTEFACTS")
print("=" * 80)

print(OUTPUT_FILE)
print(SUMMARY_FILE)

print("\nValidation:")
print("M_EU only: PASS")
print("238 OOF cases: PASS")
print("68 represented scenarios: PASS")
print("2000 scenario bootstrap replicates: PASS")
print("Scenario multiplicity preserved: PASS")
print("Coverage intervals generated: PASS")
print("Accepted-accuracy intervals generated: PASS")
print("Selective-risk intervals generated: PASS")
print("Holdout untouched: PASS")
print("No policy thresholds selected: PASS")

print(
    "\nA8.10.2 SCENARIO-LEVEL POLICY "
    "OPERATING-POINT ANALYSIS: PASS"
)