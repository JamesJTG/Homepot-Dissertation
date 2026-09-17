import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


# ============================================================
# Frozen policy design
# ============================================================

SELECTED_MODEL = "M_EU"

MIN_ISSUE_ACCURACY = 0.80
MIN_ISSUE_CASES = 40

POLICY_STATES = [
    "WITHHOLD",
    "QUALIFY",
    "ISSUE",
]


# ============================================================
# Paths
# ============================================================

BASE = Path(
    "research/diagnostic_reliability/analysis/outputs"
)

A8_9_DIR = (
    BASE
    / "a8_9_reliability_estimator_selection"
)

A8_10_DIR = (
    BASE
    / "a8_10_post_inference_reliability_policy"
)

SELECTION_FILE = (
    A8_9_DIR
    / "a8_9_reliability_estimator_selection.json"
)

PROFILE_FILE = (
    A8_10_DIR
    / "a8_10_development_selective_risk_coverage_profile.csv"
)

BOOTSTRAP_FILE = (
    A8_10_DIR
    / "a8_10_scenario_bootstrap_policy_operating_points.csv"
)

OUTPUT_FILE = (
    A8_10_DIR
    / "a8_10_frozen_post_inference_reliability_policy.json"
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


# ============================================================
# Validate inputs
# ============================================================

for path in [
    SELECTION_FILE,
    PROFILE_FILE,
    BOOTSTRAP_FILE,
]:
    if not path.exists():
        raise FileNotFoundError(path)


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
    != SELECTED_MODEL
):
    raise RuntimeError(
        "Frozen A8.9 estimator "
        "is not M_EU."
    )


# ============================================================
# Load development policy profile
# ============================================================

profile_rows = load_csv(
    PROFILE_FILE
)

model_rows = [
    row
    for row in profile_rows
    if row["model"] == SELECTED_MODEL
]


if len(model_rows) != 101:
    raise RuntimeError(
        f"Expected 101 threshold rows "
        f"for {SELECTED_MODEL}, "
        f"found {len(model_rows)}."
    )


# ============================================================
# Find first threshold satisfying
# frozen prototype Issue criterion
# ============================================================

eligible_thresholds = []


for row in model_rows:

    if not row["accepted_accuracy"]:
        continue

    threshold = float(
        row["threshold"]
    )

    n_accepted = int(
        row["n_accepted"]
    )

    accuracy = float(
        row["accepted_accuracy"]
    )

    risk = float(
        row["selective_risk"]
    )

    coverage = float(
        row["coverage"]
    )

    if (
        accuracy
        >= MIN_ISSUE_ACCURACY
        and
        n_accepted
        >= MIN_ISSUE_CASES
    ):

        eligible_thresholds.append({
            "threshold":
                threshold,

            "n_accepted":
                n_accepted,

            "coverage":
                coverage,

            "accepted_accuracy":
                accuracy,

            "selective_risk":
                risk,
        })


if not eligible_thresholds:
    raise RuntimeError(
        "No development threshold "
        "satisfies the frozen Issue criterion."
    )


selected = min(
    eligible_thresholds,
    key=lambda x: x["threshold"],
)

ISSUE_THRESHOLD = (
    selected["threshold"]
)


# ============================================================
# Obtain bootstrap interval for selected threshold
# ============================================================

bootstrap_rows = load_csv(
    BOOTSTRAP_FILE
)

matching_bootstrap = [
    row
    for row in bootstrap_rows
    if (
        row["model"]
        == SELECTED_MODEL
        and
        abs(
            float(row["threshold"])
            - ISSUE_THRESHOLD
        )
        < 1e-12
    )
]


if len(matching_bootstrap) != 1:
    raise RuntimeError(
        "Could not uniquely locate "
        "selected threshold in bootstrap output."
    )


bootstrap = (
    matching_bootstrap[0]
)


# ============================================================
# Construct frozen policy
# ============================================================

policy = {
    "stage": "A8.10",
    "substage": "A8.10.3",

    "name": (
        "Validation-Preserving "
        "Post-Inference Reliability Policy"
    ),

    "status":
        "FROZEN_BEFORE_HOLDOUT",

    "created_at_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "selected_reliability_estimator":
        SELECTED_MODEL,

    "policy_states":
        POLICY_STATES,

    "policy_logic": {
        "WITHHOLD": {
            "condition":
                "E == 0",

            "meaning": (
                "Upstream evidence is not "
                "actionable. The generated fault "
                "diagnosis is not presented as "
                "sufficiently reliable for "
                "operational decision support."
            ),
        },

        "QUALIFY": {
            "condition": (
                f"E == 1 and "
                f"predicted_reliability "
                f"< {ISSUE_THRESHOLD:.2f}"
            ),

            "meaning": (
                "Upstream evidence is actionable, "
                "but estimated diagnostic reliability "
                "does not satisfy the prototype Issue "
                "criterion. The diagnosis may be "
                "presented with explicit reliability "
                "qualification and recommendation for "
                "additional verification."
            ),
        },

        "ISSUE": {
            "condition": (
                f"E == 1 and "
                f"predicted_reliability "
                f">= {ISSUE_THRESHOLD:.2f}"
            ),

            "meaning": (
                "Upstream evidence is actionable "
                "and estimated diagnostic reliability "
                "satisfies the frozen prototype "
                "Issue criterion."
            ),
        },
    },

    "issue_threshold": {
        "value":
            ISSUE_THRESHOLD,

        "selection_rule": (
            "Lowest threshold on the frozen "
            "0.01 development grid with accepted "
            "diagnostic accuracy >= 0.80 and "
            "at least 40 accepted cases."
        ),

        "criterion_type":
            "prototype_operating_criterion",

        "criterion_interpretation": (
            "The 0.80 accuracy requirement is "
            "an explicit prototype design criterion, "
            "not an externally validated operational "
            "safety standard."
        ),

        "minimum_accepted_cases":
            MIN_ISSUE_CASES,

        "minimum_development_accuracy":
            MIN_ISSUE_ACCURACY,
    },

    "development_operating_point": {
        "n_accepted":
            selected[
                "n_accepted"
            ],

        "coverage":
            selected[
                "coverage"
            ],

        "accepted_accuracy":
            selected[
                "accepted_accuracy"
            ],

        "selective_risk":
            selected[
                "selective_risk"
            ],

        "coverage_ci_95": [
            float(
                bootstrap[
                    "coverage_ci_low"
                ]
            ),
            float(
                bootstrap[
                    "coverage_ci_high"
                ]
            ),
        ],

        "accepted_accuracy_ci_95": [
            float(
                bootstrap[
                    "accepted_accuracy_ci_low"
                ]
            ),
            float(
                bootstrap[
                    "accepted_accuracy_ci_high"
                ]
            ),
        ],

        "selective_risk_ci_95": [
            float(
                bootstrap[
                    "selective_risk_ci_low"
                ]
            ),
            float(
                bootstrap[
                    "selective_risk_ci_high"
                ]
            ),
        ],
    },

    "architectural_rationale": (
        "The policy preserves HOMEPOT's "
        "validation-first semantics. "
        "Downstream reliability estimation "
        "may refine handling of an actionable "
        "diagnosis but does not override a "
        "non-actionable upstream evidence state."
    ),

    "limitations": [
        (
            "The Issue criterion is a prototype "
            "operating rule rather than a validated "
            "deployment safety threshold."
        ),
        (
            "No human-participant or organisational "
            "utility study was used to derive "
            "decision costs."
        ),
        (
            "The threshold was selected using "
            "development OOF evidence only."
        ),
        (
            "The final holdout has not been used "
            "to alter or validate the threshold."
        ),
        (
            "The policy applies to primary-outcome "
            "eligible substantive diagnoses."
        ),
    ],

    "holdout": {
        "used":
            False,

        "status":
            "UNTOUCHED",
    },

    "evidence_provenance": {
        "a8_9_selection": {
            "path":
                str(SELECTION_FILE),

            "sha256":
                sha256(SELECTION_FILE),
        },

        "a8_10_selective_profile": {
            "path":
                str(PROFILE_FILE),

            "sha256":
                sha256(PROFILE_FILE),
        },

        "a8_10_scenario_bootstrap": {
            "path":
                str(BOOTSTRAP_FILE),

            "sha256":
                sha256(BOOTSTRAP_FILE),
        },
    },
}


# ============================================================
# Write frozen policy
# ============================================================

with OUTPUT_FILE.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        policy,
        f,
        indent=2,
    )


# ============================================================
# Report
# ============================================================

print("=" * 80)
print("A8.10.3 POLICY DEFINITION AND THRESHOLD FREEZE")
print("=" * 80)

print("\nSelected estimator:")
print("  M_EU")

print("\nPolicy:")
print("  E = 0")
print("    -> WITHHOLD")

print(
    f"  E = 1 and p < "
    f"{ISSUE_THRESHOLD:.2f}"
)
print("    -> QUALIFY")

print(
    f"  E = 1 and p >= "
    f"{ISSUE_THRESHOLD:.2f}"
)
print("    -> ISSUE")


print("\nFrozen Issue selection rule:")
print(
    "  Lowest 0.01-grid threshold "
    "with accuracy >= 0.80 "
    "and n accepted >= 40"
)


print("\nSelected development operating point:")

print(
    "  Threshold:",
    f"{ISSUE_THRESHOLD:.2f}",
)

print(
    "  Accepted:",
    selected[
        "n_accepted"
    ],
)

print(
    "  Coverage:",
    f"{selected['coverage']:.6f}",
)

print(
    "  Accepted accuracy:",
    f"{selected['accepted_accuracy']:.6f}",
)

print(
    "  Selective risk:",
    f"{selected['selective_risk']:.6f}",
)

print(
    "  Accuracy 95% CI:",
    (
        f"[{float(bootstrap['accepted_accuracy_ci_low']):.6f}, "
        f"{float(bootstrap['accepted_accuracy_ci_high']):.6f}]"
    ),
)

print(
    "  Risk 95% CI:",
    (
        f"[{float(bootstrap['selective_risk_ci_low']):.6f}, "
        f"{float(bootstrap['selective_risk_ci_high']):.6f}]"
    ),
)


print("\nInterpretation:")
print(
    "  Prototype operating criterion, "
    "NOT an externally validated "
    "safety threshold."
)

print(
    "  Upstream E=0 cannot be overridden "
    "by downstream reliability."
)

print("\nHoldout used: NO")
print("Holdout status: UNTOUCHED")

print("\nOutput:")
print(OUTPUT_FILE)

print(
    "\nA8.10.3 POST-INFERENCE "
    "RELIABILITY POLICY: FROZEN"
)