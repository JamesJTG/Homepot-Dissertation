"""Create the frozen A8.7 development/holdout scenario partition.

The partition implements the pre-specified 120-development / 30-holdout
protocol using only pre-outcome experimental design information.

No LLM diagnosis, correctness outcome, confidence, entropy, agreement,
UNKNOWN status, or other downstream result is used to construct the split.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


RAW_CASES = Path(
    "research/diagnostic_reliability/outputs/main_experiment/cases.jsonl"
)

OUTPUT_DIR = Path(
    "research/diagnostic_reliability/analysis/outputs/"
    "a8_7_reliability_modelling"
)

ASSIGNMENT_CSV = OUTPUT_DIR / (
    "a8_7_scenario_development_holdout_assignment.csv"
)

MANIFEST_JSON = OUTPUT_DIR / (
    "a8_7_scenario_partition_manifest.json"
)

# Fixed once for this partition.
PARTITION_SEED = 20260917


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_unique_scenarios() -> list[dict]:
    scenarios: dict[str, dict] = {}

    with RAW_CASES.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            private = record["private_research"]

            scenario_id = private["scenario_id"]

            scenario = {
                "scenario_id": scenario_id,
                "fault": private["fault"],
                "scenario_family": private["scenario_family"],
                "instance": int(private["instance"]),
                "seed": int(private["seed"]),
            }

            if scenario_id in scenarios:
                if scenarios[scenario_id] != scenario:
                    raise RuntimeError(
                        f"Inconsistent metadata for {scenario_id}"
                    )
            else:
                scenarios[scenario_id] = scenario

    result = list(scenarios.values())

    if len(result) != 150:
        raise RuntimeError(
            f"Expected 150 unique scenarios, found {len(result)}"
        )

    return result


def create_partition(scenarios: list[dict]) -> list[dict]:
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for scenario in scenarios:
        key = (
            scenario["fault"],
            scenario["scenario_family"],
        )
        strata[key].append(scenario)

    if len(strata) != 15:
        raise RuntimeError(
            f"Expected 15 fault-family strata, found {len(strata)}"
        )

    rng = random.Random(PARTITION_SEED)
    assignments: list[dict] = []

    for key in sorted(strata):
        members = sorted(
            strata[key],
            key=lambda row: row["scenario_id"],
        )

        if len(members) != 10:
            raise RuntimeError(
                f"{key} contains {len(members)} scenarios, expected 10"
            )

        holdout_ids = {
            row["scenario_id"]
            for row in rng.sample(members, 2)
        }

        for row in members:
            assignment = dict(row)

            if row["scenario_id"] in holdout_ids:
                assignment["partition"] = "HOLDOUT"
            else:
                assignment["partition"] = "DEVELOPMENT"

            assignments.append(assignment)

    assignments.sort(key=lambda row: row["scenario_id"])

    return assignments


def validate_partition(assignments: list[dict]) -> None:
    counts = Counter(
        row["partition"]
        for row in assignments
    )

    assert len(assignments) == 150
    assert counts["DEVELOPMENT"] == 120
    assert counts["HOLDOUT"] == 30

    strata = defaultdict(Counter)

    for row in assignments:
        key = (
            row["fault"],
            row["scenario_family"],
        )
        strata[key][row["partition"]] += 1

    assert len(strata) == 15

    for key, counts_by_partition in strata.items():
        assert counts_by_partition["DEVELOPMENT"] == 8, key
        assert counts_by_partition["HOLDOUT"] == 2, key


def write_assignment(assignments: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fields = [
        "scenario_id",
        "fault",
        "scenario_family",
        "instance",
        "seed",
        "partition",
    ]

    with ASSIGNMENT_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(assignments)


def write_manifest(assignments: list[dict]) -> None:
    counts = Counter(
        row["partition"]
        for row in assignments
    )

    manifest = {
        "stage": "A8.7",
        "stage_name": "Reliability Modelling",
        "substage": (
            "A8.7.2 Frozen Modelling Protocol Recovery "
            "and Empirical Compatibility Check"
        ),
        "partition_status": "FROZEN",
        "partition_seed": PARTITION_SEED,
        "partition_method": (
            "Fixed-seed stratified random assignment within each "
            "fault x scenario-family stratum. Each of the 15 strata "
            "contains 10 scenarios, with 8 assigned to DEVELOPMENT "
            "and 2 assigned to HOLDOUT."
        ),
        "outcome_blind": True,
        "prohibited_split_inputs": [
            "diagnostic_correctness",
            "decision_appropriateness",
            "modal_diagnosis",
            "UNKNOWN status",
            "mean_self_confidence",
            "diagnostic_entropy",
            "modal_agreement",
            "LLM generation outputs",
        ],
        "allowed_split_inputs": [
            "scenario_id",
            "fault",
            "scenario_family",
            "instance",
            "scenario seed",
        ],
        "counts": {
            "total_scenarios": len(assignments),
            "development_scenarios": counts["DEVELOPMENT"],
            "holdout_scenarios": counts["HOLDOUT"],
        },
        "source": {
            "path": str(RAW_CASES),
            "sha256": sha256_file(RAW_CASES),
        },
        "assignment": {
            "path": str(ASSIGNMENT_CSV),
            "sha256": sha256_file(ASSIGNMENT_CSV),
        },
        "methodological_note": (
            "The 120-development / 30-holdout protocol was "
            "pre-specified before reliability modelling. The exact "
            "scenario-ID assignment had not been materialised in the "
            "repository. This artefact instantiates that protocol "
            "without reference to observed LLM outcomes."
        ),
    }

    with MANIFEST_JSON.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            manifest,
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")


def main() -> None:
    scenarios = load_unique_scenarios()
    assignments = create_partition(scenarios)

    validate_partition(assignments)
    write_assignment(assignments)
    write_manifest(assignments)

    counts = Counter(
        row["partition"]
        for row in assignments
    )

    print("A8.7 scenario partition created and validated.")
    print(f"Total scenarios:       {len(assignments)}")
    print(f"Development scenarios: {counts['DEVELOPMENT']}")
    print(f"Holdout scenarios:     {counts['HOLDOUT']}")
    print(f"Partition seed:         {PARTITION_SEED}")
    print(f"Assignment:             {ASSIGNMENT_CSV}")
    print(f"Manifest:               {MANIFEST_JSON}")


if __name__ == "__main__":
    main()