"""Tests protecting the frozen main-experiment configuration."""

from __future__ import annotations

from research.diagnostic_reliability.experiment_config import (
    MAIN_EXPERIMENT,
)


def test_frozen_model_and_generation_settings() -> None:
    assert MAIN_EXPERIMENT.model == "gemma3:12b"
    assert MAIN_EXPERIMENT.temperature == 0.4
    assert MAIN_EXPERIMENT.generations_per_evidence_case == 7


def test_frozen_scenario_structure() -> None:
    assert MAIN_EXPERIMENT.telemetry_points_per_scenario == 24
    assert MAIN_EXPERIMENT.families_per_fault_class == 3
    assert MAIN_EXPERIMENT.instances_per_family == 10


def test_derived_scenario_counts() -> None:
    assert MAIN_EXPERIMENT.fault_class_count == 5
    assert MAIN_EXPERIMENT.evidence_condition_count == 4

    assert MAIN_EXPERIMENT.total_scenario_families == 15
    assert MAIN_EXPERIMENT.total_scenario_instances == 150
    assert MAIN_EXPERIMENT.total_evidence_cases == 600


def test_planned_generation_count() -> None:
    assert MAIN_EXPERIMENT.total_llm_generations == 4200