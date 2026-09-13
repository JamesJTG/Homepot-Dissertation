"""Tests for the fixed diagnostic task specification."""

from research.diagnostic_reliability.prompt import (
    DIAGNOSTIC_CLASS_DEFINITIONS,
    DIAGNOSTIC_PROMPT,
    SYSTEM_PROMPT,
)
from research.diagnostic_reliability.taxonomy import (
    DiagnosisLabel,
)


def test_every_diagnosis_label_has_semantic_definition():
    assert set(
        DIAGNOSTIC_CLASS_DEFINITIONS
    ) == set(
        DiagnosisLabel
    )

    for label in DiagnosisLabel:
        definition = (
            DIAGNOSTIC_CLASS_DEFINITIONS[
                label
            ]
        )

        assert isinstance(
            definition,
            str,
        )

        assert definition.strip()


def test_diagnostic_prompt_includes_every_label():
    for label in DiagnosisLabel:
        assert (
            label.value
            in DIAGNOSTIC_PROMPT
        )


def test_normal_and_unknown_are_explicitly_distinguished():
    normal_definition = (
        DIAGNOSTIC_CLASS_DEFINITIONS[
            DiagnosisLabel.NORMAL
        ]
    )

    unknown_definition = (
        DIAGNOSTIC_CLASS_DEFINITIONS[
            DiagnosisLabel.UNKNOWN
        ]
    )

    assert "sufficient" in (
        normal_definition.lower()
    )

    assert "normal operation" in (
        normal_definition.lower()
    )

    assert "insufficient" in (
        unknown_definition.lower()
    )

    assert "not normal operation" in (
        unknown_definition.lower()
    )

    assert "NORMAL" in SYSTEM_PROMPT
    assert "UNKNOWN" in SYSTEM_PROMPT


def test_prompt_does_not_expose_scenario_family_codes():
    forbidden_family_codes = (
        "CS-1",
        "CS-2",
        "CS-3",
        "MP-1",
        "MP-2",
        "MP-3",
        "ND-1",
        "ND-2",
        "ND-3",
        "AF-1",
        "AF-2",
        "AF-3",
        "N-1",
        "N-2",
        "N-3",
    )

    combined_prompt = (
        SYSTEM_PROMPT
        + "\n"
        + DIAGNOSTIC_PROMPT
    )

    for code in forbidden_family_codes:
        assert code not in combined_prompt


def test_prompt_does_not_expose_private_experimental_fields():
    combined_prompt = (
        SYSTEM_PROMPT
        + "\n"
        + DIAGNOSTIC_PROMPT
    ).lower()

    forbidden = (
        "scenario_id",
        "scenario_family",
        "injection configuration",
        "expected diagnosis",
        "private ground truth",
        "seed=",
    )

    for phrase in forbidden:
        assert phrase not in combined_prompt