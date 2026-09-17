import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


# ============================================================
# Frozen configuration
# ============================================================

ISSUE_THRESHOLD = 0.64

EXPECTED_HOLDOUT_SCENARIOS = 30
EXPECTED_HOLDOUT_CASES = 120

EXPECTED_SUBSTANTIVE = 54
EXPECTED_UNKNOWN = 66

EXPECTED_CORRECT = 11
EXPECTED_INCORRECT = 43

EXPECTED_ACTIONABLE = 30
EXPECTED_NON_ACTIONABLE = 90


# ============================================================
# Paths
# ============================================================

BASE = Path(
    "research/diagnostic_reliability/analysis/outputs"
)

A8_6_DIR = (
    BASE
    / "a8_6_outcome_uncertainty_construction"
)

A8_7_DIR = (
    BASE
    / "a8_7_reliability_modelling"
)

A8_11_DIR = (
    BASE
    / "a8_11_homepot_integration"
)

A8_12_DIR = (
    BASE
    / "a8_12_feature_level_evaluation"
)


DATASET_FILE = (
    A8_6_DIR
    / "a8_6_evidence_case_reliability_dataset.csv"
)

PARTITION_FILE = (
    A8_7_DIR
    / "a8_7_scenario_development_holdout_assignment.csv"
)

MEU_PARAMETERS_FILE = (
    A8_11_DIR
    / "a8_11_final_development_meu_parameters.json"
)

PROTOCOL_FILE = (
    A8_12_DIR
    / "a8_12_pre_holdout_evaluation_protocol.json"
)

HOLDOUT_ESTIMATOR_SUMMARY_FILE = (
    A8_12_DIR
    / "a8_12_holdout_estimator_evaluation_summary.json"
)

CASE_OUTPUT_FILE = (
    A8_12_DIR
    / "a8_12_holdout_feature_policy_cases.csv"
)

STATE_SUMMARY_FILE = (
    A8_12_DIR
    / "a8_12_holdout_feature_policy_state_summary.csv"
)

SUMMARY_FILE = (
    A8_12_DIR
    / "a8_12_holdout_feature_policy_summary.json"
)

MANIFEST_FILE = (
    A8_12_DIR
    / "a8_12_holdout_feature_policy_manifest.json"
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


def load_json(path):
    with path.open(
        encoding="utf-8",
    ) as f:
        return json.load(f)


def parse_bool(value):
    value = str(value).strip().lower()

    if value == "true":
        return True

    if value == "false":
        return False

    raise ValueError(
        f"Unexpected boolean value: {value}"
    )


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)

    z = math.exp(x)
    return z / (1.0 + z)


def safe_rate(
    numerator,
    denominator,
):
    if denominator == 0:
        return None

    return (
        numerator
        / denominator
    )


# ============================================================
# Load inputs
# ============================================================

required_files = [
    DATASET_FILE,
    PARTITION_FILE,
    MEU_PARAMETERS_FILE,
    PROTOCOL_FILE,
    HOLDOUT_ESTIMATOR_SUMMARY_FILE,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(
            path
        )


dataset_rows = load_csv(
    DATASET_FILE
)

partition_rows = load_csv(
    PARTITION_FILE
)

parameters = load_json(
    MEU_PARAMETERS_FILE
)

protocol = load_json(
    PROTOCOL_FILE
)

estimator_summary = load_json(
    HOLDOUT_ESTIMATOR_SUMMARY_FILE
)


print("=" * 80)
print(
    "A8.12.4 FROZEN POLICY AND "
    "FEATURE BEHAVIOUR EVALUATION"
)
print("=" * 80)


# ============================================================
# Verify frozen state
# ============================================================

if (
    estimator_summary[
        "holdout_now_opened"
    ]
    is not True
):
    raise RuntimeError(
        "A8.12.2 holdout evaluation "
        "has not been completed."
    )


if (
    estimator_summary[
        "post_holdout_tuning_permitted"
    ]
    is not False
):
    raise RuntimeError(
        "Post-holdout tuning safeguard "
        "is not frozen."
    )


# Recover frozen M_EU parameters from the
# persisted A8.11 deployment artefact.

if parameters["model"] != "M_EU":
    raise RuntimeError(
        "Unexpected frozen estimator."
    )

if (
    parameters["status"]
    != "FROZEN_FOR_PROTOTYPE_INTEGRATION"
):
    raise RuntimeError(
        "M_EU parameters are not recorded "
        "as frozen for prototype integration."
    )

intercept = float(
    parameters[
        "flic_intercept"
    ]
)

predictor_coefficients = {
    predictor["name"]:
        float(
            predictor["coefficient"]
        )
    for predictor
    in parameters["predictors"]
}

expected_predictors = {
    "E",
    "mean_self_confidence",
    "diagnostic_entropy",
}

if (
    set(predictor_coefficients)
    != expected_predictors
):
    raise RuntimeError(
        "Unexpected frozen M_EU "
        "predictor specification."
    )

beta_e = predictor_coefficients[
    "E"
]

beta_c = predictor_coefficients[
    "mean_self_confidence"
]

beta_h = predictor_coefficients[
    "diagnostic_entropy"
]

if abs(
    float(
        parameters[
            "policy"
        ][
            "issue_threshold"
        ]
    )
    - ISSUE_THRESHOLD
) > 1e-12:
    raise RuntimeError(
        "Frozen A8.11 ISSUE threshold "
        "does not equal 0.64."
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
        "Frozen UNKNOWN handling "
        "is not ABSTAIN."
    )


# ============================================================
# Recover holdout
# ============================================================

holdout_scenarios = {
    row["scenario_id"]
    for row in partition_rows
    if (
        row["partition"]
        .strip()
        .lower()
        == "holdout"
    )
}


if (
    len(holdout_scenarios)
    != EXPECTED_HOLDOUT_SCENARIOS
):
    raise RuntimeError(
        "Expected 30 holdout scenarios, "
        f"found {len(holdout_scenarios)}."
    )


holdout_rows = [
    row
    for row in dataset_rows
    if (
        row["scenario_id"]
        in holdout_scenarios
    )
]


if (
    len(holdout_rows)
    != EXPECTED_HOLDOUT_CASES
):
    raise RuntimeError(
        "Expected 120 holdout cases, "
        f"found {len(holdout_rows)}."
    )


# ============================================================
# Apply frozen feature semantics
# ============================================================

case_records = []


for row in holdout_rows:

    modal_diagnosis = (
        row["modal_diagnosis"]
        .strip()
    )

    evidence_ready = parse_bool(
        row["actionable"]
    )

    mean_confidence = float(
        row["mean_self_confidence"]
    )

    entropy = float(
        row["diagnostic_entropy"]
    )

    modal_agreement = float(
        row["modal_agreement"]
    )

    y_a = int(
        row[
            "decision_appropriateness"
        ]
    )

    diagnostic_correctness_raw = (
        row[
            "diagnostic_correctness"
        ].strip()
    )


    if modal_diagnosis == "UNKNOWN":

        if (
            diagnostic_correctness_raw
            != ""
        ):
            raise RuntimeError(
                f"{row['case_id']}: "
                "UNKNOWN unexpectedly has Y_D."
            )

        y_d = None
        predicted_reliability = None
        decision_state = "ABSTAIN"

    else:

        if (
            diagnostic_correctness_raw
            == ""
        ):
            raise RuntimeError(
                f"{row['case_id']}: "
                "substantive diagnosis "
                "missing Y_D."
            )

        y_d = int(
            diagnostic_correctness_raw
        )

        if y_d not in {
            0,
            1,
        }:
            raise RuntimeError(
                f"{row['case_id']}: "
                f"invalid Y_D={y_d}."
            )

        linear_predictor = (
            intercept
            + beta_e
            * int(evidence_ready)
            + beta_c
            * mean_confidence
            + beta_h
            * entropy
        )

        predicted_reliability = (
            sigmoid(
                linear_predictor
            )
        )

        if not evidence_ready:
            decision_state = (
                "WITHHOLD"
            )

        elif (
            predicted_reliability
            < ISSUE_THRESHOLD
        ):
            decision_state = (
                "QUALIFY"
            )

        else:
            decision_state = (
                "ISSUE"
            )


    case_records.append({
        "case_id":
            row["case_id"],

        "scenario_id":
            row["scenario_id"],

        "scenario_family":
            row["scenario_family"],

        "ground_truth":
            row["ground_truth"],

        "evidence_condition":
            row["evidence_condition"],

        "modal_diagnosis":
            modal_diagnosis,

        "E":
            int(evidence_ready),

        "mean_self_confidence":
            mean_confidence,

        "diagnostic_entropy":
            entropy,

        "modal_agreement":
            modal_agreement,

        "predicted_reliability":
            (
                ""
                if predicted_reliability
                is None
                else predicted_reliability
            ),

        "decision_state":
            decision_state,

        "Y_D":
            (
                ""
                if y_d is None
                else y_d
            ),

        "Y_A":
            y_a,

        "error_category":
            row["error_category"],
    })


# ============================================================
# Population validation
# ============================================================

state_counts = Counter(
    record["decision_state"]
    for record in case_records
)


unknown_records = [
    record
    for record in case_records
    if (
        record[
            "modal_diagnosis"
        ]
        == "UNKNOWN"
    )
]


substantive_records = [
    record
    for record in case_records
    if (
        record[
            "modal_diagnosis"
        ]
        != "UNKNOWN"
    )
]


if (
    len(unknown_records)
    != EXPECTED_UNKNOWN
):
    raise RuntimeError(
        "Expected 66 UNKNOWN cases, "
        f"found {len(unknown_records)}."
    )


if (
    len(substantive_records)
    != EXPECTED_SUBSTANTIVE
):
    raise RuntimeError(
        "Expected 54 substantive cases, "
        f"found {len(substantive_records)}."
    )


n_correct = sum(
    record["Y_D"] == 1
    for record in substantive_records
)


n_incorrect = sum(
    record["Y_D"] == 0
    for record in substantive_records
)


if (
    n_correct
    != EXPECTED_CORRECT
):
    raise RuntimeError(
        "Expected 11 correct "
        "substantive diagnoses."
    )


if (
    n_incorrect
    != EXPECTED_INCORRECT
):
    raise RuntimeError(
        "Expected 43 incorrect "
        "substantive diagnoses."
    )


actionable_count = sum(
    record["E"] == 1
    for record in case_records
)


non_actionable_count = sum(
    record["E"] == 0
    for record in case_records
)


if (
    actionable_count
    != EXPECTED_ACTIONABLE
):
    raise RuntimeError(
        "Expected 30 actionable cases."
    )


if (
    non_actionable_count
    != EXPECTED_NON_ACTIONABLE
):
    raise RuntimeError(
        "Expected 90 non-actionable cases."
    )


# Feature-semantic assertions.

for record in unknown_records:
    if (
        record["decision_state"]
        != "ABSTAIN"
    ):
        raise RuntimeError(
            "UNKNOWN did not map "
            "to ABSTAIN."
        )

    if (
        record[
            "predicted_reliability"
        ]
        != ""
    ):
        raise RuntimeError(
            "UNKNOWN unexpectedly received "
            "a reliability probability."
        )


for record in substantive_records:

    if (
        record["E"] == 0
        and record["decision_state"]
        != "WITHHOLD"
    ):
        raise RuntimeError(
            "Substantive E=0 case did "
            "not map to WITHHOLD."
        )

    if record["E"] == 1:

        p = float(
            record[
                "predicted_reliability"
            ]
        )

        expected_state = (
            "ISSUE"
            if p >= ISSUE_THRESHOLD
            else "QUALIFY"
        )

        if (
            record["decision_state"]
            != expected_state
        ):
            raise RuntimeError(
                "Frozen threshold policy "
                "was not reproduced."
            )


# ============================================================
# State-level outcomes
# ============================================================

state_summary_records = []


for state in [
    "ABSTAIN",
    "WITHHOLD",
    "QUALIFY",
    "ISSUE",
]:

    records = [
        record
        for record in case_records
        if (
            record[
                "decision_state"
            ]
            == state
        )
    ]

    n = len(records)

    yd_eligible = [
        record
        for record in records
        if (
            record["Y_D"]
            != ""
        )
    ]

    yd_correct = sum(
        record["Y_D"] == 1
        for record in yd_eligible
    )

    ya_appropriate = sum(
        record["Y_A"] == 1
        for record in records
    )

    probabilities = [
        float(
            record[
                "predicted_reliability"
            ]
        )
        for record in records
        if (
            record[
                "predicted_reliability"
            ]
            != ""
        )
    ]

    state_summary_records.append({
        "decision_state":
            state,

        "n_cases":
            n,

        "fraction_of_all_holdout":
            safe_rate(
                n,
                len(case_records),
            ),

        "Y_D_eligible_cases":
            len(yd_eligible),

        "Y_D_correct_cases":
            yd_correct,

        "Y_D_accuracy":
            safe_rate(
                yd_correct,
                len(yd_eligible),
            ),

        "Y_A_appropriate_cases":
            ya_appropriate,

        "Y_A_appropriateness_rate":
            safe_rate(
                ya_appropriate,
                n,
            ),

        "mean_predicted_reliability":
            (
                None
                if not probabilities
                else sum(
                    probabilities
                ) / len(
                    probabilities
                )
            ),
    })


# ============================================================
# Key feature metrics
# ============================================================

issue_records = [
    record
    for record in substantive_records
    if (
        record[
            "decision_state"
        ]
        == "ISSUE"
    )
]


qualify_records = [
    record
    for record in substantive_records
    if (
        record[
            "decision_state"
        ]
        == "QUALIFY"
    )
]


withhold_records = [
    record
    for record in substantive_records
    if (
        record[
            "decision_state"
        ]
        == "WITHHOLD"
    )
]


issue_correct = sum(
    record["Y_D"] == 1
    for record in issue_records
)


issue_incorrect = sum(
    record["Y_D"] == 0
    for record in issue_records
)


appropriate_abstentions = sum(
    record["Y_A"] == 1
    for record in unknown_records
)


unnecessary_abstentions = sum(
    record["Y_A"] == 0
    for record in unknown_records
)


summary = {
    "stage":
        "A8.12",

    "substage":
        "A8.12.4",

    "name":
        (
            "Frozen Policy and "
            "Feature Behaviour Evaluation"
        ),

    "status":
        "COMPLETE",

    "population": {
        "holdout_scenarios":
            len(
                holdout_scenarios
            ),

        "holdout_cases":
            len(
                case_records
            ),

        "substantive_cases":
            len(
                substantive_records
            ),

        "unknown_cases":
            len(
                unknown_records
            ),

        "substantive_correct":
            n_correct,

        "substantive_incorrect":
            n_incorrect,
    },

    "frozen_policy": {
        "unknown":
            "ABSTAIN",

        "substantive_E_0":
            "WITHHOLD",

        "substantive_E_1_below_threshold":
            "QUALIFY",

        "substantive_E_1_at_or_above_threshold":
            "ISSUE",

        "issue_threshold":
            ISSUE_THRESHOLD,
    },

    "state_counts":
        dict(
            state_counts
        ),

    "issue_behaviour": {
        "n_issue":
            len(
                issue_records
            ),

        "coverage_among_substantive":
            safe_rate(
                len(
                    issue_records
                ),
                len(
                    substantive_records
                ),
            ),

        "correct":
            issue_correct,

        "incorrect":
            issue_incorrect,

        "accuracy":
            safe_rate(
                issue_correct,
                len(
                    issue_records
                ),
            ),

        "selective_risk":
            safe_rate(
                issue_incorrect,
                len(
                    issue_records
                ),
            ),
    },

    "qualify_behaviour": {
        "n_qualify":
            len(
                qualify_records
            ),

        "correct":
            sum(
                record["Y_D"] == 1
                for record
                in qualify_records
            ),

        "incorrect":
            sum(
                record["Y_D"] == 0
                for record
                in qualify_records
            ),
    },

    "withhold_behaviour": {
        "n_withhold":
            len(
                withhold_records
            ),

        "correct":
            sum(
                record["Y_D"] == 1
                for record
                in withhold_records
            ),

        "incorrect":
            sum(
                record["Y_D"] == 0
                for record
                in withhold_records
            ),
    },

    "abstention_behaviour": {
        "n_abstain":
            len(
                unknown_records
            ),

        "appropriate_abstentions":
            appropriate_abstentions,

        "unnecessary_abstentions":
            unnecessary_abstentions,

        "appropriateness_rate":
            safe_rate(
                appropriate_abstentions,
                len(
                    unknown_records
                ),
            ),
    },

    "safeguards": {
        "unknown_treated_as_incorrect":
            False,

        "model_refitting":
            False,

        "threshold_reselection":
            False,

        "post_holdout_recalibration":
            False,

        "prompt_change":
            False,

        "generation_change":
            False,
    },
}


# ============================================================
# Export case-level feature behaviour
# ============================================================

with CASE_OUTPUT_FILE.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    fieldnames = [
        "case_id",
        "scenario_id",
        "scenario_family",
        "ground_truth",
        "evidence_condition",
        "modal_diagnosis",
        "E",
        "mean_self_confidence",
        "diagnostic_entropy",
        "modal_agreement",
        "predicted_reliability",
        "decision_state",
        "Y_D",
        "Y_A",
        "error_category",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(
        case_records
    )


# ============================================================
# Export state summary
# ============================================================

with STATE_SUMMARY_FILE.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    fieldnames = [
        "decision_state",
        "n_cases",
        "fraction_of_all_holdout",
        "Y_D_eligible_cases",
        "Y_D_correct_cases",
        "Y_D_accuracy",
        "Y_A_appropriate_cases",
        "Y_A_appropriateness_rate",
        "mean_predicted_reliability",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(
        state_summary_records
    )


# ============================================================
# Export summary and manifest
# ============================================================

with SUMMARY_FILE.open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
    )


manifest = {
    "stage":
        "A8.12",

    "substage":
        "A8.12.4",

    "status":
        "COMPLETE",

    "feature":
        (
            "Post-Inference Diagnostic "
            "Reliability Layer"
        ),

    "estimator":
        "M_EU",

    "policy_threshold":
        ISSUE_THRESHOLD,

    "inputs": {
        str(DATASET_FILE):
            sha256(
                DATASET_FILE
            ),

        str(PARTITION_FILE):
            sha256(
                PARTITION_FILE
            ),

        str(MEU_PARAMETERS_FILE):
            sha256(
                MEU_PARAMETERS_FILE
            ),

        str(PROTOCOL_FILE):
            sha256(
                PROTOCOL_FILE
            ),

        str(HOLDOUT_ESTIMATOR_SUMMARY_FILE):
            sha256(
                HOLDOUT_ESTIMATOR_SUMMARY_FILE
            ),
    },

    "outputs": {
        "case_level":
            str(
                CASE_OUTPUT_FILE
            ),

        "state_summary":
            str(
                STATE_SUMMARY_FILE
            ),

        "summary":
            str(
                SUMMARY_FILE
            ),
    },

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

print("\nHoldout feature population:")
print(
    "  Cases:",
    len(
        case_records
    ),
)
print(
    "  Scenarios:",
    len(
        holdout_scenarios
    ),
)
print(
    "  Substantive:",
    len(
        substantive_records
    ),
)
print(
    "  UNKNOWN:",
    len(
        unknown_records
    ),
)


print("\nDecision states:")

for state in [
    "ABSTAIN",
    "WITHHOLD",
    "QUALIFY",
    "ISSUE",
]:
    print(
        f"  {state}:",
        state_counts[
            state
        ],
    )


print("\nISSUE behaviour:")
print(
    "  Issued:",
    len(
        issue_records
    ),
)
print(
    "  Coverage among substantive:",
    f"{safe_rate(len(issue_records), len(substantive_records)):.6f}",
)
print(
    "  Correct:",
    issue_correct,
)
print(
    "  Incorrect:",
    issue_incorrect,
)

if issue_records:
    print(
        "  Accuracy:",
        f"{safe_rate(issue_correct, len(issue_records)):.6f}",
    )
    print(
        "  Selective risk:",
        f"{safe_rate(issue_incorrect, len(issue_records)):.6f}",
    )


print("\nQUALIFY behaviour:")
print(
    "  Qualified:",
    len(
        qualify_records
    ),
)
print(
    "  Correct:",
    sum(
        record["Y_D"] == 1
        for record
        in qualify_records
    ),
)
print(
    "  Incorrect:",
    sum(
        record["Y_D"] == 0
        for record
        in qualify_records
    ),
)


print("\nWITHHOLD behaviour:")
print(
    "  Withheld:",
    len(
        withhold_records
    ),
)
print(
    "  Correct:",
    sum(
        record["Y_D"] == 1
        for record
        in withhold_records
    ),
)
print(
    "  Incorrect:",
    sum(
        record["Y_D"] == 0
        for record
        in withhold_records
    ),
)


print("\nABSTAIN behaviour:")
print(
    "  Abstained:",
    len(
        unknown_records
    ),
)
print(
    "  Appropriate:",
    appropriate_abstentions,
)
print(
    "  Unnecessary:",
    unnecessary_abstentions,
)
print(
    "  Appropriateness rate:",
    f"{safe_rate(appropriate_abstentions, len(unknown_records)):.6f}",
)


print("\nOutput artefacts:")
print(
    CASE_OUTPUT_FILE
)
print(
    STATE_SUMMARY_FILE
)
print(
    SUMMARY_FILE
)
print(
    MANIFEST_FILE
)


print("\nValidation:")
print(
    "30 holdout scenarios retained: PASS"
)
print(
    "120 holdout cases retained: PASS"
)
print(
    "54 substantive cases retained: PASS"
)
print(
    "66 UNKNOWN cases retained: PASS"
)
print(
    "UNKNOWN -> ABSTAIN: PASS"
)
print(
    "UNKNOWN receives no reliability probability: PASS"
)
print(
    "Substantive E=0 -> WITHHOLD: PASS"
)
print(
    "Frozen 0.64 ISSUE threshold reproduced: PASS"
)
print(
    "UNKNOWN not converted to Y_D=0: PASS"
)
print(
    "No estimator refitting: PASS"
)
print(
    "No threshold reselection: PASS"
)
print(
    "No post-holdout recalibration: PASS"
)


print(
    "\nA8.12.4 FROZEN POLICY AND "
    "FEATURE BEHAVIOUR EVALUATION: "
    "COMPLETE"
)