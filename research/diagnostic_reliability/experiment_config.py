"""Frozen A8.4 configuration for the main diagnostic-reliability experiment.

These values represent decisions made after the A8.4 pilot and research
assurance work. They should not be changed during the main experiment merely
to improve diagnostic performance.

Any later change must be explicitly documented as a methodological change and
must not silently alter already collected experimental data.
"""

from __future__ import annotations

from dataclasses import dataclass

from .taxonomy import EvidenceCondition, FaultLabel


@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable configuration for the main experiment."""

    model: str
    temperature: float
    generations_per_evidence_case: int
    telemetry_points_per_scenario: int
    instances_per_family: int
    families_per_fault_class: int

    @property
    def fault_class_count(self) -> int:
        """Number of injected operational ground-truth classes."""

        return len(FaultLabel)

    @property
    def evidence_condition_count(self) -> int:
        """Number of controlled evidence-quality conditions."""

        return len(EvidenceCondition)

    @property
    def total_scenario_families(self) -> int:
        """Total number of scenario-pattern families."""

        return (
            self.fault_class_count
            * self.families_per_fault_class
        )

    @property
    def total_scenario_instances(self) -> int:
        """Total independently generated scenario instances."""

        return (
            self.total_scenario_families
            * self.instances_per_family
        )

    @property
    def total_evidence_cases(self) -> int:
        """Total scenario-by-evidence-condition cases."""

        return (
            self.total_scenario_instances
            * self.evidence_condition_count
        )

    @property
    def total_llm_generations(self) -> int:
        """Total LLM calls in the planned main experiment."""

        return (
            self.total_evidence_cases
            * self.generations_per_evidence_case
        )


MAIN_EXPERIMENT = ExperimentConfig(
    model="gemma3:12b",
    temperature=0.4,
    generations_per_evidence_case=7,
    telemetry_points_per_scenario=24,
    instances_per_family=10,
    families_per_fault_class=3,
)