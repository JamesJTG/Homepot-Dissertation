"""Immutable snapshot schema for reproducible repeated LLM generation."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Any, Mapping

from .taxonomy import EvidenceCondition, EvidenceSufficiency, FaultLabel


@dataclass(frozen=True)
class ScenarioSnapshot:
    scenario_id: str
    device_id: str
    fault: FaultLabel
    evidence_condition: EvidenceCondition
    evidence_sufficiency: EvidenceSufficiency
    seed: int
    telemetry: tuple[dict[str, Any], ...]
    validation: Mapping[str, Any]
    model_context: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "device_id": self.device_id,
            "seed": self.seed,
            "telemetry": list(self.telemetry),
            "validation": dict(self.validation),
            "model_context": self.model_context,
            # Ground truth is stored as private experimental metadata and is
            # never serialized into model_context.
            "ground_truth": {
                "fault": self.fault.value,
                "evidence_condition": self.evidence_condition.value,
                "evidence_sufficiency": self.evidence_sufficiency.value,
            },
        }

    def write_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return destination
