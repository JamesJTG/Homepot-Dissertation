"""Tests for fixed-support diagnostic uncertainty measures."""

import math

import pytest

from research.diagnostic_reliability.taxonomy import (
    DiagnosisLabel,
)
from research.diagnostic_reliability.uncertainty import (
    DIAGNOSIS_SUPPORT_SIZE,
    diagnosis_distribution,
    diagnostic_entropy,
    modal_fraction,
)


def test_support_size_matches_complete_diagnosis_taxonomy():
    assert (
        DIAGNOSIS_SUPPORT_SIZE
        == len(DiagnosisLabel)
    )

    assert DIAGNOSIS_SUPPORT_SIZE == 6


def test_complete_agreement_has_zero_entropy():
    labels = [
        DiagnosisLabel.CPU_SATURATION
    ] * 7

    assert (
        diagnostic_entropy(labels)
        == 0.0
    )

    assert modal_fraction(labels) == 1.0


def test_two_way_even_split_uses_fixed_support_normalisation():
    labels = [
        DiagnosisLabel.CPU_SATURATION,
        DiagnosisLabel.MEMORY_PRESSURE,
        DiagnosisLabel.CPU_SATURATION,
        DiagnosisLabel.MEMORY_PRESSURE,
    ]

    expected = (
        math.log(2)
        / math.log(
            DIAGNOSIS_SUPPORT_SIZE
        )
    )

    assert diagnostic_entropy(
        labels
    ) == pytest.approx(
        expected
    )

    assert modal_fraction(labels) == 0.5


def test_two_way_split_is_not_normalised_to_one():
    labels = [
        DiagnosisLabel.NORMAL,
        DiagnosisLabel.UNKNOWN,
    ]

    entropy = diagnostic_entropy(
        labels
    )

    assert 0.0 < entropy < 1.0


def test_uniform_complete_support_has_entropy_one():
    labels = list(
        DiagnosisLabel
    )

    assert diagnostic_entropy(
        labels
    ) == pytest.approx(
        1.0
    )


def test_raw_entropy_remains_available():
    labels = [
        DiagnosisLabel.NORMAL,
        DiagnosisLabel.UNKNOWN,
    ]

    assert diagnostic_entropy(
        labels,
        normalise=False,
    ) == pytest.approx(
        math.log(2)
    )


def test_distribution_contains_observed_labels_only():
    labels = [
        DiagnosisLabel.NORMAL,
        DiagnosisLabel.NORMAL,
        DiagnosisLabel.UNKNOWN,
    ]

    distribution = (
        diagnosis_distribution(
            labels
        )
    )

    assert distribution == {
        DiagnosisLabel.NORMAL.value: (
            2 / 3
        ),
        DiagnosisLabel.UNKNOWN.value: (
            1 / 3
        ),
    }


@pytest.mark.parametrize(
    "function",
    (
        diagnosis_distribution,
        diagnostic_entropy,
        modal_fraction,
    ),
)
def test_uncertainty_functions_reject_empty_input(
    function,
):
    with pytest.raises(
        ValueError,
        match=(
            "At least one diagnosis "
            "label is required"
        ),
    ):
        function([])