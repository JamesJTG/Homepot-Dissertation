"""Shared data structures and utilities for controlled diagnostic scenarios."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime

from ..taxonomy import FaultLabel


@dataclass(frozen=True)
class TelemetryPoint:
    """One timestamped operational telemetry observation."""

    timestamp: datetime
    cpu_percent: float | None
    memory_percent: float | None
    disk_percent: float | None
    network_latency_ms: float | None
    error_rate: float | None
    active_connections: int | None = None
    queue_depth: int | None = None

    def to_dict(self) -> dict:
        """Return a serialisable representation of the telemetry point."""

        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


@dataclass(frozen=True)
class Scenario:
    """One controlled operational scenario with private ground truth."""

    scenario_id: str
    scenario_family: str
    device_id: str
    fault: FaultLabel
    seed: int
    telemetry: tuple[TelemetryPoint, ...]

    def to_dict(self) -> dict:
        """Return a serialisable representation of the scenario."""

        return {
            "scenario_id": self.scenario_id,
            "scenario_family": self.scenario_family,
            "device_id": self.device_id,
            "fault": self.fault.value,
            "seed": self.seed,
            "telemetry": [
                point.to_dict()
                for point in self.telemetry
            ],
        }


def clamp(value: float, low: float, high: float) -> float:
    """Clamp a numeric value to an inclusive range."""

    return max(low, min(high, value))