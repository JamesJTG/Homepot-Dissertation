"""Deterministic scenario generation for HOMEPOT diagnostic experiments.

The generator deliberately creates *operational evidence*, not answer-bearing
anomaly labels. Evidence-quality transformations are applied separately in
``evidence_conditions.py`` so the experimental factors F and Q remain distinct.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
import random
from typing import Dict, List

from .taxonomy import FaultLabel


@dataclass(frozen=True)
class TelemetryPoint:
    timestamp: datetime
    cpu_percent: float | None
    memory_percent: float | None
    disk_percent: float | None
    network_latency_ms: float | None
    error_rate: float | None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    device_id: str
    fault: FaultLabel
    seed: int
    telemetry: tuple[TelemetryPoint, ...]

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "device_id": self.device_id,
            "fault": self.fault.value,
            "seed": self.seed,
            "telemetry": [point.to_dict() for point in self.telemetry],
        }


# Values deliberately sit on clear sides of the current HOMEPOT anomaly
# thresholds (CPU/memory 90%, latency 200ms, error rate 0.05) while keeping
# non-target metrics plausible. Exact final ranges remain pilot-tunable.
_RANGES: Dict[FaultLabel, Dict[str, tuple[float, float]]] = {
    FaultLabel.NORMAL: {
        "cpu_percent": (30.0, 55.0),
        "memory_percent": (35.0, 60.0),
        "disk_percent": (25.0, 60.0),
        "network_latency_ms": (15.0, 80.0),
        "error_rate": (0.0, 0.015),
    },
    FaultLabel.CPU_SATURATION: {
        "cpu_percent": (92.0, 98.0),
        "memory_percent": (40.0, 65.0),
        "disk_percent": (25.0, 60.0),
        "network_latency_ms": (15.0, 80.0),
        "error_rate": (0.0, 0.015),
    },
    FaultLabel.MEMORY_PRESSURE: {
        "cpu_percent": (35.0, 65.0),
        "memory_percent": (92.0, 98.0),
        "disk_percent": (25.0, 60.0),
        "network_latency_ms": (15.0, 80.0),
        "error_rate": (0.0, 0.015),
    },
    FaultLabel.APPLICATION_ERROR: {
        "cpu_percent": (35.0, 65.0),
        "memory_percent": (40.0, 65.0),
        "disk_percent": (25.0, 60.0),
        "network_latency_ms": (15.0, 100.0),
        "error_rate": (0.08, 0.16),
    },
    FaultLabel.NETWORK_DEGRADATION: {
        "cpu_percent": (35.0, 65.0),
        "memory_percent": (40.0, 65.0),
        "disk_percent": (25.0, 60.0),
        "network_latency_ms": (300.0, 700.0),
        "error_rate": (0.0, 0.02),
    },
}


def generate_scenario(
    fault: FaultLabel,
    *,
    instance: int = 1,
    seed: int = 1,
    device_id: str = "EXP_DEVICE_01",
    n_points: int = 12,
    interval_seconds: int = 30,
    end_time: datetime | None = None,
) -> Scenario:
    """Create a reproducible telemetry scenario with one dominant condition."""
    if n_points < 2:
        raise ValueError("n_points must be at least 2")
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")

    rng = random.Random(seed)
    end = end_time or datetime.now(timezone.utc)
    ranges = _RANGES[fault]
    points: List[TelemetryPoint] = []

    for idx in range(n_points):
        # Oldest -> newest ordering makes context rendering and gap transforms clear.
        offset = (n_points - 1 - idx) * interval_seconds
        ts = end - timedelta(seconds=offset)
        values = {
            key: round(rng.uniform(low, high), 3)
            for key, (low, high) in ranges.items()
        }
        points.append(TelemetryPoint(timestamp=ts, **values))

    return Scenario(
        scenario_id=f"{fault.value}_{instance:03d}",
        device_id=device_id,
        fault=fault,
        seed=seed,
        telemetry=tuple(points),
    )
