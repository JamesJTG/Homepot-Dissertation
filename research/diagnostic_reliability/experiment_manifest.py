"""Deterministic manifest for the A8.5 main experiment.

The manifest defines the structural identity of every controlled scenario
instance before LLM execution.

It deliberately contains private research metadata such as ground-truth fault
labels and seeds. Manifest records must therefore never be passed directly to
the LLM. Model-visible context must continue to be constructed exclusively
through ``build_model_context``.

Scenario timestamps are intentionally not frozen in this manifest. HOMEPOT's
production Gate B evaluates evidence freshness relative to the time at which
validation is performed. Each evidence case therefore receives an execution-
time timestamp when it is first prepared. That timestamp, its validation
result, and its final model-visible context are then persisted by the main
experiment runner so that resumed generations continue to use the same frozen
case.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .experiment_config import MAIN_EXPERIMENT
from .scenarios import (
    AF1_FAMILY,
    AF2_FAMILY,
    AF3_FAMILY,
    CS1_FAMILY,
    CS2_FAMILY,
    CS3_FAMILY,
    MP1_FAMILY,
    MP2_FAMILY,
    MP3_FAMILY,
    N1_FAMILY,
    N2_FAMILY,
    N3_FAMILY,
    ND1_FAMILY,
    ND2_FAMILY,
    ND3_FAMILY,
)
from .taxonomy import FaultLabel


MANIFEST_VERSION = "A8.5-v2"


FAMILY_MAP: dict[FaultLabel, tuple[str, ...]] = {
    FaultLabel.NORMAL: (
        N1_FAMILY,
        N2_FAMILY,
        N3_FAMILY,
    ),
    FaultLabel.CPU_SATURATION: (
        CS1_FAMILY,
        CS2_FAMILY,
        CS3_FAMILY,
    ),
    FaultLabel.MEMORY_PRESSURE: (
        MP1_FAMILY,
        MP2_FAMILY,
        MP3_FAMILY,
    ),
    FaultLabel.NETWORK_DEGRADATION: (
        ND1_FAMILY,
        ND2_FAMILY,
        ND3_FAMILY,
    ),
    FaultLabel.APPLICATION_FAILURE: (
        AF1_FAMILY,
        AF2_FAMILY,
        AF3_FAMILY,
    ),
}


FAULT_ORDER = (
    FaultLabel.NORMAL,
    FaultLabel.CPU_SATURATION,
    FaultLabel.MEMORY_PRESSURE,
    FaultLabel.NETWORK_DEGRADATION,
    FaultLabel.APPLICATION_FAILURE,
)


@dataclass(frozen=True)
class ManifestEntry:
    """Private structural specification for one scenario instance."""

    manifest_index: int
    fault: str
    scenario_family: str
    instance: int
    seed: int


def _seed_for(
    *,
    fault_index: int,
    family_index: int,
    instance: int,
) -> int:
    """Return a deterministic unique seed for one scenario instance."""

    return (
        202609130000
        + (fault_index * 1000)
        + (family_index * 100)
        + instance
    )


def build_manifest() -> list[ManifestEntry]:
    """Build the complete deterministic main-experiment manifest."""

    entries: list[ManifestEntry] = []
    manifest_index = 1

    for fault_index, fault in enumerate(
        FAULT_ORDER,
        start=1,
    ):
        families = FAMILY_MAP[fault]

        if (
            len(families)
            != MAIN_EXPERIMENT.families_per_fault_class
        ):
            raise RuntimeError(
                f"{fault.value} has {len(families)} families; "
                f"expected "
                f"{MAIN_EXPERIMENT.families_per_fault_class}."
            )

        for family_index, family in enumerate(
            families,
            start=1,
        ):
            for instance in range(
                1,
                MAIN_EXPERIMENT.instances_per_family + 1,
            ):
                entries.append(
                    ManifestEntry(
                        manifest_index=manifest_index,
                        fault=fault.value,
                        scenario_family=family,
                        instance=instance,
                        seed=_seed_for(
                            fault_index=fault_index,
                            family_index=family_index,
                            instance=instance,
                        ),
                    )
                )

                manifest_index += 1

    if (
        len(entries)
        != MAIN_EXPERIMENT.total_scenario_instances
    ):
        raise RuntimeError(
            f"Manifest contains {len(entries)} scenarios; "
            f"expected "
            f"{MAIN_EXPERIMENT.total_scenario_instances}."
        )

    return entries


def manifest_document() -> dict:
    """Return the serialisable frozen manifest document."""

    entries = build_manifest()

    return {
        "manifest_version": MANIFEST_VERSION,
        "experiment": {
            "model": MAIN_EXPERIMENT.model,
            "temperature": MAIN_EXPERIMENT.temperature,
            "generations_per_evidence_case": (
                MAIN_EXPERIMENT.generations_per_evidence_case
            ),
            "telemetry_points_per_scenario": (
                MAIN_EXPERIMENT.telemetry_points_per_scenario
            ),
            "fault_class_count": (
                MAIN_EXPERIMENT.fault_class_count
            ),
            "families_per_fault_class": (
                MAIN_EXPERIMENT.families_per_fault_class
            ),
            "instances_per_family": (
                MAIN_EXPERIMENT.instances_per_family
            ),
            "evidence_condition_count": (
                MAIN_EXPERIMENT.evidence_condition_count
            ),
            "total_scenario_instances": (
                MAIN_EXPERIMENT.total_scenario_instances
            ),
            "total_evidence_cases": (
                MAIN_EXPERIMENT.total_evidence_cases
            ),
            "total_llm_generations": (
                MAIN_EXPERIMENT.total_llm_generations
            ),
        },
        "timestamp_policy": (
            "Assigned when an evidence case is first prepared, then persisted "
            "with the frozen case context."
        ),
        "scenarios": [
            asdict(entry)
            for entry in entries
        ],
    }


def write_manifest(path: str | Path) -> Path:
    """Write the deterministic manifest without silently overwriting it."""

    output_path = Path(path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    document = manifest_document()
    serialised = json.dumps(
        document,
        indent=2,
        sort_keys=True,
    )

    if output_path.exists():
        existing = output_path.read_text(
            encoding="utf-8"
        )

        if existing != serialised + "\n":
            raise RuntimeError(
                "Existing manifest differs from the frozen "
                f"manifest: {output_path}"
            )

        return output_path

    output_path.write_text(
        serialised + "\n",
        encoding="utf-8",
    )

    return output_path


if __name__ == "__main__":
    path = write_manifest(
        "research/diagnostic_reliability/"
        "outputs/main_experiment_manifest.json"
    )

    print(
        f"Wrote frozen experiment manifest: {path}"
    )