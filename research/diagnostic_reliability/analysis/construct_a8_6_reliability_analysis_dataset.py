"""A8.6 outcome and uncertainty construction.

Consumes the frozen A8.5 main-experiment corpus and creates derived
generation-level and evidence-case-level analytical records.

The raw experimental artefacts are never modified.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from statistics import mean

from research.diagnostic_reliability.scoring import score_diagnosis
from research.diagnostic_reliability.taxonomy import (
    DiagnosisLabel,
    EvidenceSufficiency,
    FaultLabel,
)
from research.diagnostic_reliability.uncertainty import (
    diagnosis_distribution,
    diagnostic_entropy,
    modal_fraction,
)


RUN = Path(
    "research/diagnostic_reliability/outputs/main_experiment"
)

OUT = Path(
    "research/diagnostic_reliability/analysis/outputs/a8_6_outcome_uncertainty_construction"
)


def load_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                )
                + "\n"
            )


def modal_diagnosis(
    labels: list[DiagnosisLabel],
) -> tuple[DiagnosisLabel, bool]:
    """Return frozen case-level modal diagnosis.

    If multiple labels share the maximum count, the case-level
    diagnosis becomes UNKNOWN.
    """

    counts = Counter(labels)
    maximum = max(counts.values())

    winners = [
        label
        for label, count in counts.items()
        if count == maximum
    ]

    if len(winners) != 1:
        return DiagnosisLabel.UNKNOWN, True

    return winners[0], False


def main() -> None:
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    cases = load_jsonl(
        RUN / "cases.jsonl"
    )

    generations = load_jsonl(
        RUN / "generations.jsonl"
    )

    generations_by_case = defaultdict(list)

    for row in generations:
        generations_by_case[
            row["case_id"]
        ].append(row)

    derived_generations = []
    analytical_cases = []

    for case in cases:
        case_id = case["case_id"]

        case_generations = sorted(
            generations_by_case[case_id],
            key=lambda row: row["generation_id"],
        )

        if len(case_generations) != 7:
            raise RuntimeError(
                f"{case_id} has "
                f"{len(case_generations)} generations"
            )

        truth = FaultLabel(
            case["private_research"]["fault"]
        )

        sufficiency = EvidenceSufficiency(
            case["expected_sufficiency"]
        )

        labels = [
            DiagnosisLabel(row["diagnosis"])
            for row in case_generations
        ]

        confidences = [
            float(row["confidence"])
            for row in case_generations
        ]

        # ----------------------------------------------------------
        # Generation-level derived outcomes
        # ----------------------------------------------------------

        for generation, diagnosis in zip(
            case_generations,
            labels,
        ):
            score = score_diagnosis(
                truth,
                diagnosis,
                sufficiency,
            )

            derived_generations.append(
                {
                    "case_id": case_id,
                    "scenario_id": (
                        case["private_research"][
                            "scenario_id"
                        ]
                    ),
                    "generation_id": (
                        generation["generation_id"]
                    ),
                    "ground_truth": truth.value,
                    "evidence_condition": (
                        case["evidence_condition"]
                    ),
                    "expected_sufficiency": (
                        sufficiency.value
                    ),
                    "diagnosis": diagnosis.value,
                    "self_confidence": (
                        generation["confidence"]
                    ),
                    "diagnostic_correctness": (
                        score.diagnostic_correctness
                    ),
                    "decision_appropriateness": (
                        score.decision_appropriateness
                    ),
                    "error_category": (
                        score.error_category.value
                    ),
                }
            )

        # ----------------------------------------------------------
        # Case-level repeated-generation measurements
        # ----------------------------------------------------------

        modal_label, modal_tie = modal_diagnosis(
            labels
        )

        case_score = score_diagnosis(
            truth,
            modal_label,
            sufficiency,
        )

        distribution = diagnosis_distribution(
            labels
        )

        validation = case["validation"]

        analytical_cases.append(
            {
                "case_id": case_id,
                "scenario_id": (
                    case["private_research"][
                        "scenario_id"
                    ]
                ),
                "scenario_family": (
                    case["private_research"][
                        "scenario_family"
                    ]
                ),
                "manifest_index": (
                    case["manifest_index"]
                ),

                # Controlled/private experimental variables
                "ground_truth": truth.value,
                "evidence_condition": (
                    case["evidence_condition"]
                ),
                "expected_sufficiency": (
                    sufficiency.value
                ),

                # Upstream evidence-readiness information
                "trust_score": (
                    validation["trust_score"]
                ),
                "trust_mode": (
                    validation["trust_mode"]
                ),
                "actionable": (
                    validation["actionable"]
                ),
                "failed_gate_id": (
                    validation["failed_gate_id"]
                ),

                # Downstream diagnosis and uncertainty
                "modal_diagnosis": (
                    modal_label.value
                ),
                "modal_tie": modal_tie,
                "mean_self_confidence": (
                    mean(confidences)
                ),
                "diagnostic_entropy": (
                    diagnostic_entropy(labels)
                ),
                "modal_agreement": (
                    modal_fraction(labels)
                ),
                "diagnosis_distribution": (
                    distribution
                ),

                # Frozen outcomes
                "diagnostic_correctness": (
                    case_score.diagnostic_correctness
                ),
                "primary_correctness_eligible": (
                    case_score.diagnostic_correctness
                    is not None
                ),
                "decision_appropriateness": (
                    case_score.decision_appropriateness
                ),
                "error_category": (
                    case_score.error_category.value
                ),
            }
        )

    # --------------------------------------------------------------
    # Integrity checks
    # --------------------------------------------------------------

    assert len(derived_generations) == 4200
    assert len(analytical_cases) == 600

    assert len({
        row["case_id"]
        for row in analytical_cases
    }) == 600

    # --------------------------------------------------------------
    # Persist derived artefacts
    # --------------------------------------------------------------

    write_jsonl(
        OUT / "a8_6_generation_level_diagnostic_outcomes.jsonl",
        derived_generations,
    )

    write_jsonl(
        OUT / "a8_6_evidence_case_reliability_dataset.jsonl",
        analytical_cases,
    )

    csv_fields = [
        key
        for key in analytical_cases[0]
        if key != "diagnosis_distribution"
    ]

    with (
        OUT / "a8_6_evidence_case_reliability_dataset.csv"
    ).open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=csv_fields,
        )

        writer.writeheader()

        for row in analytical_cases:
            writer.writerow({
                key: row[key]
                for key in csv_fields
            })

    # --------------------------------------------------------------
    # Readiness summary
    # --------------------------------------------------------------

    modal_counts = Counter(
        row["modal_diagnosis"]
        for row in analytical_cases
    )

    correctness_counts = Counter(
        row["diagnostic_correctness"]
        for row in analytical_cases
    )

    appropriateness_counts = Counter(
        row["decision_appropriateness"]
        for row in analytical_cases
    )

    error_counts = Counter(
        row["error_category"]
        for row in analytical_cases
    )

    ties = sum(
        row["modal_tie"]
        for row in analytical_cases
    )

    eligible = sum(
        row["primary_correctness_eligible"]
        for row in analytical_cases
    )

    print("=" * 72)
    print("A8.6 OUTCOME AND UNCERTAINTY CONSTRUCTION")
    print("=" * 72)

    print(
        f"Generation-level records: "
        f"{len(derived_generations)}"
    )

    print(
        f"Case-level records:       "
        f"{len(analytical_cases)}"
    )

    print(
        f"Primary Y_D eligible:     "
        f"{eligible}"
    )

    print(
        f"Modal ties:               "
        f"{ties}"
    )

    print("\nMODAL DIAGNOSIS")
    for label, count in modal_counts.most_common():
        print(
            f"{label:30} {count}"
        )

    print("\nDIAGNOSTIC CORRECTNESS Y_D")
    for outcome, count in sorted(
        correctness_counts.items(),
        key=lambda item: str(item[0]),
    ):
        print(
            f"{str(outcome):30} {count}"
        )

    print("\nDECISION APPROPRIATENESS Y_A")
    for outcome, count in sorted(
        appropriateness_counts.items()
    ):
        print(
            f"{str(outcome):30} {count}"
        )

    print("\nCASE-LEVEL ERROR CATEGORIES")
    for category, count in error_counts.most_common():
        print(
            f"{category:30} {count}"
        )

    print("\nOUTPUTS")
    print(
        OUT / "a8_6_generation_level_diagnostic_outcomes.jsonl"
    )
    print(
        OUT / "a8_6_evidence_case_reliability_dataset.jsonl"
    )
    print(
        OUT / "a8_6_evidence_case_reliability_dataset.csv"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()