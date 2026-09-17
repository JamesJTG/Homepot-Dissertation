import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from firthmodels import FirthLogisticRegression


# ============================================================
# Frozen A8.12 pre-holdout configuration
# ============================================================

CANDIDATE_MODEL = "M_EU"
BENCHMARK_MODEL = "M_E"

BENCHMARK_PREDICTORS = [
    "E",
]

OUTCOME = "diagnostic_correctness"

PRIMARY_METRIC = "Brier score"

SUPPORTING_METRICS = [
    "AUROC",
    "AUPRC",
    "ECE",
    "reliability/calibration assessment",
]

BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260917
CONFIDENCE_LEVEL = 0.95

ISSUE_THRESHOLD = 0.64


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

A8_9_FILE = (
    OUTPUTS
    / "a8_9_reliability_estimator_selection"
    / "a8_9_reliability_estimator_selection.json"
)

A8_10_FILE = (
    OUTPUTS
    / "a8_10_post_inference_reliability_policy"
    / "a8_10_frozen_post_inference_reliability_policy.json"
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

A8_12_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

BENCHMARK_PARAMETERS_FILE = (
    A8_12_DIR
    / "a8_12_final_development_me_parameters.json"
)

PROTOCOL_FILE = (
    A8_12_DIR
    / "a8_12_pre_holdout_evaluation_protocol.json"
)

MANIFEST_FILE = (
    A8_12_DIR
    / "a8_12_pre_holdout_preparation_manifest.json"
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


def solve_flic_intercept(
    X,
    y,
    slopes,
):
    """
    Frozen FLIC correction used in A8.7 and A8.11.

    Retain the Firth slopes and solve only the intercept
    so that mean fitted probability on the training data
    equals the observed event rate.
    """

    linear_without_intercept = (
        X @ slopes
    )

    target = float(
        np.mean(y)
    )

    def objective(intercept):
        probabilities = logistic(
            intercept
            + linear_without_intercept
        )

        return (
            float(
                np.mean(probabilities)
            )
            - target
        )

    return float(
        brentq(
            objective,
            -100.0,
            100.0,
        )
    )


# ============================================================
# Validate frozen upstream artefacts
# ============================================================

required_files = [
    A8_6_FILE,
    PARTITION_FILE,
    A8_9_FILE,
    A8_10_FILE,
    A8_11_PARAMETERS_FILE,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(
            path
        )


with A8_9_FILE.open(
    encoding="utf-8",
) as f:
    selection = json.load(f)


if (
    selection[
        "selected_candidate"
    ][
        "model"
    ]
    != CANDIDATE_MODEL
):
    raise RuntimeError(
        "Frozen A8.9 selected candidate "
        "is not M_EU."
    )


if (
    selection[
        "primary_benchmark"
    ][
        "model"
    ]
    != BENCHMARK_MODEL
):
    raise RuntimeError(
        "Frozen A8.9 primary benchmark "
        "is not M_E."
    )


with A8_10_FILE.open(
    encoding="utf-8",
) as f:
    policy = json.load(f)


if (
    policy[
        "selected_reliability_estimator"
    ]
    != CANDIDATE_MODEL
):
    raise RuntimeError(
        "Frozen A8.10 policy does not "
        "use M_EU."
    )


frozen_threshold = float(
    policy[
        "issue_threshold"
    ][
        "value"
    ]
)


if abs(
    frozen_threshold
    - ISSUE_THRESHOLD
) > 1e-12:
    raise RuntimeError(
        "Unexpected frozen A8.10 "
        "Issue threshold."
    )


with A8_11_PARAMETERS_FILE.open(
    encoding="utf-8",
) as f:
    meu_parameters = json.load(f)


if (
    meu_parameters["model"]
    != CANDIDATE_MODEL
):
    raise RuntimeError(
        "A8.11 deployment parameters "
        "are not for M_EU."
    )


if (
    meu_parameters[
        "holdout"
    ][
        "used_for_evaluation"
    ]
):
    raise RuntimeError(
        "A8.11 indicates that holdout "
        "evaluation has already occurred."
    )


# ============================================================
# Load frozen partition
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
# Construct development-only benchmark dataset
# ============================================================

rows = load_csv(
    A8_6_FILE
)

development_rows = []


for row in rows:

    scenario_id = (
        row["scenario_id"]
    )

    # Critical pre-holdout safeguard:
    # only development rows enter this fit.
    if (
        scenario_id
        not in development_scenarios
    ):
        continue

    y_raw = (
        row[
            OUTCOME
        ]
        .strip()
    )

    # M_E uses the same primary Y_D-eligible
    # substantive population as M_EU.
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

    development_rows.append({
        "case_id":
            row["case_id"],

        "scenario_id":
            scenario_id,

        "evidence_condition":
            row[
                "evidence_condition"
            ],

        "E":
            E,

        "Y_D":
            y,
    })


# ============================================================
# Development-only integrity checks
# ============================================================

if len(
    development_rows
) != 238:
    raise RuntimeError(
        f"Expected 238 eligible "
        f"development cases, found "
        f"{len(development_rows)}."
    )


represented_scenarios = {
    row["scenario_id"]
    for row in development_rows
}


if len(
    represented_scenarios
) != 68:
    raise RuntimeError(
        "Expected 68 represented "
        "development scenarios."
    )


if (
    represented_scenarios
    & holdout_scenarios
):
    raise RuntimeError(
        "Holdout contamination "
        "detected."
    )


X = np.asarray(
    [
        [
            row["E"],
        ]
        for row in development_rows
    ],
    dtype=float,
)

y = np.asarray(
    [
        row["Y_D"]
        for row in development_rows
    ],
    dtype=int,
)


if X.shape != (
    238,
    1,
):
    raise RuntimeError(
        f"Unexpected X shape: "
        f"{X.shape}"
    )


n_positive = int(
    np.sum(y == 1)
)

n_negative = int(
    np.sum(y == 0)
)


if n_positive != 45:
    raise RuntimeError(
        "Expected 45 positive "
        "development outcomes."
    )


if n_negative != 193:
    raise RuntimeError(
        "Expected 193 negative "
        "development outcomes."
    )


# Frozen complete-separation structure.
e0_y1 = int(
    np.sum(
        (X[:, 0] == 0.0)
        & (y == 1)
    )
)

e0_y0 = int(
    np.sum(
        (X[:, 0] == 0.0)
        & (y == 0)
    )
)

e1_y1 = int(
    np.sum(
        (X[:, 0] == 1.0)
        & (y == 1)
    )
)

e1_y0 = int(
    np.sum(
        (X[:, 0] == 1.0)
        & (y == 0)
    )
)


if (
    e0_y0,
    e0_y1,
    e1_y0,
    e1_y1,
) != (
    177,
    0,
    16,
    45,
):
    raise RuntimeError(
        "Unexpected development "
        "E x Y_D structure."
    )


# ============================================================
# Fit frozen M_E benchmark
# ============================================================

firth = FirthLogisticRegression()

firth.fit(
    X,
    y,
)


firth_slopes = np.asarray(
    firth.coef_,
    dtype=float,
).reshape(-1)


if len(
    firth_slopes
) != 1:
    raise RuntimeError(
        "Unexpected number of "
        "M_E Firth slopes."
    )


raw_intercept = float(
    np.asarray(
        firth.intercept_,
        dtype=float,
    ).reshape(-1)[0]
)


if not np.all(
    np.isfinite(
        firth_slopes
    )
):
    raise RuntimeError(
        "Non-finite M_E slope."
    )


if not np.isfinite(
    raw_intercept
):
    raise RuntimeError(
        "Non-finite M_E intercept."
    )


# ============================================================
# Apply frozen FLIC correction
# ============================================================

flic_intercept = (
    solve_flic_intercept(
        X=X,
        y=y,
        slopes=firth_slopes,
    )
)


predicted = logistic(
    flic_intercept
    + X @ firth_slopes
)


observed_event_rate = float(
    np.mean(y)
)

mean_predicted = float(
    np.mean(predicted)
)


if abs(
    mean_predicted
    - observed_event_rate
) > 1e-10:
    raise RuntimeError(
        "M_E FLIC calibration "
        "identity failed."
    )


if not np.all(
    np.isfinite(predicted)
):
    raise RuntimeError(
        "Non-finite M_E fitted "
        "probabilities."
    )


# ============================================================
# Persist final development M_E benchmark
# ============================================================

benchmark_parameters = {
    "stage":
        "A8.12",

    "substage":
        "A8.12.1",

    "model":
        BENCHMARK_MODEL,

    "role":
        "PRIMARY_UPSTREAM_ONLY_BENCHMARK",

    "status":
        "FROZEN_BEFORE_HOLDOUT_EVALUATION",

    "target":
        "diagnostic_correctness",

    "estimator":
        "Firth logistic regression + FLIC",

    "predictors": [
        {
            "name":
                "E",

            "source":
                "actionable",

            "encoding":
                "1=actionable, 0=non-actionable",

            "coefficient":
                float(
                    firth_slopes[0]
                ),
        }
    ],

    "firth_intercept_before_flic":
        raw_intercept,

    "flic_intercept":
        flic_intercept,

    "prediction_equation": (
        "sigmoid("
        "flic_intercept"
        " + beta_E * E"
        ")"
    ),

    "development_fit": {
        "eligible_cases":
            len(
                development_rows
            ),

        "represented_scenarios":
            len(
                represented_scenarios
            ),

        "positive_outcomes":
            n_positive,

        "negative_outcomes":
            n_negative,

        "observed_correctness_rate":
            observed_event_rate,

        "mean_fitted_probability":
            mean_predicted,

        "E_by_Y_D": {
            "E0_Y0":
                e0_y0,

            "E0_Y1":
                e0_y1,

            "E1_Y0":
                e1_y0,

            "E1_Y1":
                e1_y1,
        },
    },

    "holdout": {
        "scenario_count":
            len(
                holdout_scenarios
            ),

        "used_for_fit":
            False,

        "used_for_tuning":
            False,

        "used_for_evaluation":
            False,

        "status":
            "UNTOUCHED",
    },
}


with BENCHMARK_PARAMETERS_FILE.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        benchmark_parameters,
        f,
        indent=2,
    )


# ============================================================
# Freeze A8.12 evaluation protocol
# ============================================================

protocol = {
    "stage":
        "A8.12",

    "substage":
        "A8.12.1",

    "status":
        "FROZEN_BEFORE_HOLDOUT_EVALUATION",

    "purpose": (
        "Evaluate the frozen Post-Inference "
        "Diagnostic Reliability Layer on the "
        "previously untouched scenario-level "
        "holdout partition without further "
        "model or policy development."
    ),

    "research_question_comparison": (
        "Compare upstream evidence-readiness "
        "alone (M_E) with the selected combined "
        "upstream and downstream reliability "
        "estimator (M_EU)."
    ),

    "candidate_model": {
        "model":
            CANDIDATE_MODEL,

        "parameters_source":
            str(
                A8_11_PARAMETERS_FILE
            ),

        "frozen_before_holdout":
            True,
    },

    "primary_benchmark": {
        "model":
            BENCHMARK_MODEL,

        "parameters_source":
            str(
                BENCHMARK_PARAMETERS_FILE
            ),

        "frozen_before_holdout":
            True,
    },

    "primary_outcome": {
        "name":
            "Y_D",

        "source":
            OUTCOME,

        "eligible_values":
            [0, 1],

        "UNKNOWN_excluded":
            True,

        "rationale":
            (
                "Primary diagnostic correctness "
                "is defined only for substantive "
                "modal diagnoses."
            ),
    },

    "evaluation_population": {
        "partition":
            "holdout",

        "scenario_count":
            30,

        "partitioning_unit":
            "scenario",

        "expected_primary_eligible_cases":
            54,

        "expected_primary_positive_cases":
            11,

        "expected_primary_negative_cases":
            43,

        "expected_primary_represented_scenarios":
            15,

        "known_compatibility_limitation":
            (
                "No Y_D-eligible MEMORY_PRESSURE "
                "cases are present in the holdout."
            ),
    },

    "metrics": {
        "primary":
            PRIMARY_METRIC,

        "supporting":
            SUPPORTING_METRICS,

        "calibration_notes": (
            "Calibration diagnostics are "
            "interpreted cautiously because "
            "the holdout contains only 54 "
            "Y_D-eligible cases and 11 "
            "positive outcomes."
        ),
    },

    "paired_model_comparison": {
        "comparison":
            "M_EU minus M_E",

        "same_holdout_cases":
            True,

        "purpose":
            (
                "Assess incremental information "
                "from downstream uncertainty "
                "beyond upstream evidence "
                "readiness."
            ),
    },

    "uncertainty_analysis": {
        "resampling_unit":
            "scenario",

        "bootstrap_replicates":
            BOOTSTRAP_REPLICATES,

        "seed":
            BOOTSTRAP_SEED,

        "confidence_level":
            CONFIDENCE_LEVEL,

        "same_bootstrap_samples_across_models":
            True,

        "invalid_discrimination_replicates":
            (
                "Replicates lacking both outcome "
                "classes are excluded from the "
                "corresponding discrimination "
                "metric interval and counted."
            ),
    },

    "frozen_policy": {
        "estimator":
            CANDIDATE_MODEL,

        "issue_threshold":
            ISSUE_THRESHOLD,

        "rules": {
            "E_equals_0":
                "WITHHOLD",

            "E_equals_1_below_threshold":
                "QUALIFY",

            "E_equals_1_at_or_above_threshold":
                "ISSUE",
        },

        "threshold_reselection_permitted":
            False,
    },

    "unknown_semantics": {
        "modal_UNKNOWN":
            "ABSTAIN",

        "apply_primary_reliability_estimator":
            False,

        "assign_incorrect_Y_D":
            False,
    },

    "feature_level_checks": [
        (
            "UNKNOWN modal diagnosis maps "
            "to ABSTAIN without a reliability "
            "probability."
        ),
        (
            "Substantive E=0 cases map "
            "to WITHHOLD."
        ),
        (
            "Substantive E=1 cases with "
            "predicted reliability below "
            "0.64 map to QUALIFY."
        ),
        (
            "Substantive E=1 cases with "
            "predicted reliability at or "
            "above 0.64 map to ISSUE."
        ),
    ],

    "post_holdout_prohibitions": [
        "model reselection",
        "predictor reselection",
        "coefficient refitting using holdout",
        "threshold reselection",
        "post-hoc recalibration using holdout",
        "prompt tuning",
        "temperature tuning",
        "taxonomy changes",
        "rerunning the main experiment in response to holdout performance",
    ],

    "interpretation_constraints": [
        (
            "Brier score remains the primary "
            "model-comparison metric."
        ),
        (
            "Supporting metrics do not replace "
            "the pre-specified primary metric."
        ),
        (
            "The holdout is a final evaluation "
            "sample, not a new development set."
        ),
        (
            "Failure to outperform M_E on the "
            "holdout is retained as a research "
            "result rather than tuned away."
        ),
        (
            "The 0.64 Issue threshold is a "
            "development-stage prototype "
            "operating criterion, not an "
            "externally validated safety "
            "guarantee."
        ),
    ],

    "holdout_performance_seen":
        False,
}


with PROTOCOL_FILE.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        protocol,
        f,
        indent=2,
    )


# ============================================================
# Durable preparation manifest
# ============================================================

manifest = {
    "stage":
        "A8.12.1",

    "status":
        "COMPLETE_AND_FROZEN",

    "purpose": (
        "Freeze the final development-fitted "
        "M_E benchmark and the A8.12 evaluation "
        "protocol before final holdout "
        "performance is inspected."
    ),

    "model_selection_performed":
        False,

    "predictor_selection_performed":
        False,

    "hyperparameter_tuning_performed":
        False,

    "threshold_selection_performed":
        False,

    "holdout_performance_evaluated":
        False,

    "inputs": {
        str(A8_6_FILE):
            sha256(A8_6_FILE),

        str(PARTITION_FILE):
            sha256(PARTITION_FILE),

        str(A8_9_FILE):
            sha256(A8_9_FILE),

        str(A8_10_FILE):
            sha256(A8_10_FILE),

        str(A8_11_PARAMETERS_FILE):
            sha256(
                A8_11_PARAMETERS_FILE
            ),
    },

    "outputs": {
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

print("=" * 80)
print(
    "A8.12.1 PRE-HOLDOUT "
    "EVALUATION PREPARATION"
)
print("=" * 80)

print("\nFinal benchmark:")
print("  M_E")

print("\nEstimator:")
print(
    "  Firth logistic regression "
    "+ FLIC"
)

print("\nPredictor:")
print("  E")

print("\nDevelopment data:")
print(
    "  Eligible cases:",
    len(development_rows),
)
print(
    "  Represented scenarios:",
    len(represented_scenarios),
)
print(
    "  Correct:",
    n_positive,
)
print(
    "  Incorrect:",
    n_negative,
)

print("\nE x Y_D:")
print(
    "  E=0, Y_D=0:",
    e0_y0,
)
print(
    "  E=0, Y_D=1:",
    e0_y1,
)
print(
    "  E=1, Y_D=0:",
    e1_y0,
)
print(
    "  E=1, Y_D=1:",
    e1_y1,
)

print("\nM_E coefficients:")
print(
    "  Raw Firth intercept:",
    f"{raw_intercept:.9f}",
)
print(
    "  beta_E:",
    f"{firth_slopes[0]:.9f}",
)
print(
    "  FLIC intercept:",
    f"{flic_intercept:.9f}",
)

print("\nFLIC check:")
print(
    "  Observed correctness:",
    f"{observed_event_rate:.9f}",
)
print(
    "  Mean fitted probability:",
    f"{mean_predicted:.9f}",
)

print("\nFrozen A8.12 comparison:")
print("  Candidate: M_EU")
print("  Benchmark: M_E")
print("  Primary metric: Brier score")
print(
    "  Supporting: AUROC, AUPRC, "
    "calibration, ECE"
)

print("\nFrozen policy:")
print("  E=0 -> WITHHOLD")
print(
    "  E=1 and p<0.64 -> QUALIFY"
)
print(
    "  E=1 and p>=0.64 -> ISSUE"
)
print("  UNKNOWN -> ABSTAIN")

print("\nHoldout:")
print("  30 scenarios")
print("  Performance evaluated: NO")
print("  Status: UNTOUCHED")

print("\nOutput artefacts:")
print(BENCHMARK_PARAMETERS_FILE)
print(PROTOCOL_FILE)
print(MANIFEST_FILE)

print("\nValidation:")
print("Frozen M_E benchmark specification: PASS")
print("Development-only fitting: PASS")
print("238 development cases: PASS")
print("45 correct / 193 incorrect: PASS")
print("68 represented scenarios: PASS")
print("E x Y_D structure: PASS")
print("Firth fit finite: PASS")
print("FLIC calibration identity: PASS")
print("M_EU remains selected candidate: PASS")
print("M_E remains primary benchmark: PASS")
print("A8.10 threshold unchanged: PASS")
print("No model selection: PASS")
print("No threshold selection: PASS")
print("Holdout performance unseen: PASS")

print(
    "\nA8.12.1 PRE-HOLDOUT PROTOCOL: "
    "COMPLETE + FROZEN"
)