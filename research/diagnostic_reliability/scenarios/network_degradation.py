"""Controlled network degradation scenario families.

The NETWORK_DEGRADATION fault class contains three families:

ND-1
    Persistent latency degradation.

ND-2
    Intermittent severe latency.

ND-3
    Network degradation with elevated request errors.

All three families are implemented.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import random

from .common import Scenario, TelemetryPoint, clamp
from ..taxonomy import FaultLabel


ND1_FAMILY = "ND-1_PERSISTENT_LATENCY_DEGRADATION"
ND2_FAMILY = "ND-2_INTERMITTENT_SEVERE_LATENCY"
ND3_FAMILY = "ND-3_DEGRADATION_WITH_REQUEST_ERRORS"


SUPPORTED_NETWORK_FAMILIES = {
    ND1_FAMILY,
    ND2_FAMILY,
    ND3_FAMILY,
}


def _generate_persistent_latency_degradation(
    *,
    rng: random.Random,
    end_time: datetime,
    n_points: int,
    interval_seconds: float,
) -> tuple[TelemetryPoint, ...]:
    """Generate ND-1: persistent network latency degradation."""

    points: list[TelemetryPoint] = []

    degradation_index = max(4, n_points // 3)

    for idx in range(n_points):
        if idx < degradation_index:
            pre_progress = idx / max(1, degradation_index - 1)

            latency_target = 55.0 + (35.0 * pre_progress)
            cpu_target = 48.0 + (3.0 * pre_progress)
            memory_target = 52.0 + (2.0 * pre_progress)
            connection_target = 22.0 + (2.0 * pre_progress)
            queue_target = 6.0 + (1.0 * pre_progress)
            error_target = 0.006

        else:
            degradation_progress = (
                (idx - degradation_index)
                / max(1, n_points - degradation_index - 1)
            )

            latency_target = 180.0 + (780.0 * degradation_progress)
            cpu_target = 51.0 + (9.0 * degradation_progress)
            memory_target = 54.0 + (5.0 * degradation_progress)
            connection_target = 24.0 + (6.0 * degradation_progress)
            queue_target = 7.0 + (5.0 * degradation_progress)
            error_target = 0.007 + (0.005 * degradation_progress)

        latency = clamp(
            latency_target + rng.uniform(-30.0, 30.0),
            20.0,
            1100.0,
        )

        cpu = clamp(
            cpu_target + rng.uniform(-2.0, 2.0),
            20.0,
            75.0,
        )

        memory = clamp(
            memory_target + rng.uniform(-1.5, 1.5),
            39.8,
            70.2,
        )

        active_connections = int(
            round(
                clamp(
                    connection_target + rng.uniform(-2.0, 2.0),
                    10.0,
                    40.0,
                )
            )
        )

        queue_depth = int(
            round(
                clamp(
                    queue_target + rng.uniform(-1.0, 1.0),
                    2.0,
                    15.0,
                )
            )
        )

        error_rate = clamp(
            error_target + rng.uniform(-0.002, 0.002),
            0.0,
            0.02,
        )

        disk = clamp(
            40.0 + rng.uniform(-5.0, 5.0),
            20.0,
            60.0,
        )

        offset = (n_points - 1 - idx) * interval_seconds
        timestamp = end_time - timedelta(seconds=offset)

        points.append(
            TelemetryPoint(
                timestamp=timestamp,
                cpu_percent=round(cpu, 3),
                memory_percent=round(memory, 3),
                disk_percent=round(disk, 3),
                network_latency_ms=round(latency, 3),
                error_rate=round(error_rate, 5),
                active_connections=active_connections,
                queue_depth=queue_depth,
            )
        )

    return tuple(points)


def _generate_intermittent_severe_latency(
    *,
    rng: random.Random,
    end_time: datetime,
    n_points: int,
    interval_seconds: float,
) -> tuple[TelemetryPoint, ...]:
    """Generate ND-2: intermittent severe network latency."""

    points: list[TelemetryPoint] = []

    spike_windows = (
        range(5, 8),
        range(11, 14),
        range(17, 20),
        range(22, 24),
    )

    spike_indices = {
        index
        for window in spike_windows
        for index in window
        if index < n_points
    }

    recovery_indices = {
        8,
        9,
        14,
        15,
        20,
        21,
    }

    for idx in range(n_points):
        progress = idx / max(1, n_points - 1)

        if idx in spike_indices:
            latency_target = 760.0 + (120.0 * progress)

            cpu_target = 52.0 + (4.0 * progress)
            memory_target = 53.0 + (3.0 * progress)
            connection_target = 24.0 + (3.0 * progress)
            queue_target = 7.0 + (3.0 * progress)
            error_target = 0.008

        elif idx in recovery_indices:
            latency_target = 130.0 + (40.0 * progress)

            cpu_target = 50.0 + (3.0 * progress)
            memory_target = 52.0 + (3.0 * progress)
            connection_target = 23.0 + (3.0 * progress)
            queue_target = 6.0 + (2.0 * progress)
            error_target = 0.007

        else:
            latency_target = 65.0 + (25.0 * progress)

            cpu_target = 48.0 + (4.0 * progress)
            memory_target = 51.0 + (3.0 * progress)
            connection_target = 22.0 + (3.0 * progress)
            queue_target = 6.0 + (2.0 * progress)
            error_target = 0.006

        latency = clamp(
            latency_target + rng.uniform(-35.0, 35.0),
            20.0,
            1100.0,
        )

        cpu = clamp(
            cpu_target + rng.uniform(-2.0, 2.0),
            20.0,
            75.0,
        )

        memory = clamp(
            memory_target + rng.uniform(-1.5, 1.5),
            39.8,
            70.2,
        )

        active_connections = int(
            round(
                clamp(
                    connection_target + rng.uniform(-2.0, 2.0),
                    10.0,
                    40.0,
                )
            )
        )

        queue_depth = int(
            round(
                clamp(
                    queue_target + rng.uniform(-1.0, 1.0),
                    2.0,
                    15.0,
                )
            )
        )

        error_rate = clamp(
            error_target + rng.uniform(-0.002, 0.002),
            0.0,
            0.02,
        )

        disk = clamp(
            40.0 + rng.uniform(-5.0, 5.0),
            20.0,
            60.0,
        )

        offset = (n_points - 1 - idx) * interval_seconds
        timestamp = end_time - timedelta(seconds=offset)

        points.append(
            TelemetryPoint(
                timestamp=timestamp,
                cpu_percent=round(cpu, 3),
                memory_percent=round(memory, 3),
                disk_percent=round(disk, 3),
                network_latency_ms=round(latency, 3),
                error_rate=round(error_rate, 5),
                active_connections=active_connections,
                queue_depth=queue_depth,
            )
        )

    return tuple(points)


def _generate_degradation_with_request_errors(
    *,
    rng: random.Random,
    end_time: datetime,
    n_points: int,
    interval_seconds: float,
) -> tuple[TelemetryPoint, ...]:
    """Generate ND-3: network degradation with request errors.

    Latency begins to deteriorate first. Request errors then rise as network
    conditions worsen, creating overlapping symptoms that could be mistaken
    for an application fault.

    CPU and memory remain controlled so the underlying pattern is still most
    consistent with network degradation.
    """

    points: list[TelemetryPoint] = []

    degradation_index = max(4, n_points // 4)
    error_index = max(degradation_index + 3, n_points // 2)

    for idx in range(n_points):
        progress = idx / max(1, n_points - 1)

        if idx < degradation_index:
            latency_target = 60.0 + (20.0 * progress)
            error_target = 0.006

            cpu_target = 48.0 + (2.0 * progress)
            memory_target = 52.0 + (2.0 * progress)
            connection_target = 22.0 + (2.0 * progress)
            queue_target = 6.0 + (1.0 * progress)

        elif idx < error_index:
            network_progress = (
                (idx - degradation_index)
                / max(1, error_index - degradation_index - 1)
            )

            latency_target = 180.0 + (420.0 * network_progress)
            error_target = 0.008 + (0.008 * network_progress)

            cpu_target = 50.0 + (4.0 * network_progress)
            memory_target = 53.0 + (3.0 * network_progress)
            connection_target = 23.0 + (4.0 * network_progress)
            queue_target = 7.0 + (3.0 * network_progress)

        else:
            failure_progress = (
                (idx - error_index)
                / max(1, n_points - error_index - 1)
            )

            latency_target = 620.0 + (300.0 * failure_progress)
            error_target = 0.035 + (0.045 * failure_progress)

            cpu_target = 54.0 + (5.0 * failure_progress)
            memory_target = 56.0 + (3.0 * failure_progress)
            connection_target = 27.0 + (4.0 * failure_progress)
            queue_target = 10.0 + (4.0 * failure_progress)

        latency = clamp(
            latency_target + rng.uniform(-30.0, 30.0),
            20.0,
            1100.0,
        )

        error_rate = clamp(
            error_target + rng.uniform(-0.004, 0.004),
            0.0,
            0.10,
        )

        cpu = clamp(
            cpu_target + rng.uniform(-2.0, 2.0),
            20.0,
            75.0,
        )

        memory = clamp(
            memory_target + rng.uniform(-1.5, 1.5),
            39.8,
            70.2,
        )

        active_connections = int(
            round(
                clamp(
                    connection_target + rng.uniform(-2.0, 2.0),
                    10.0,
                    40.0,
                )
            )
        )

        queue_depth = int(
            round(
                clamp(
                    queue_target + rng.uniform(-1.0, 1.0),
                    2.0,
                    16.0,
                )
            )
        )

        disk = clamp(
            40.0 + rng.uniform(-5.0, 5.0),
            20.0,
            60.0,
        )

        offset = (n_points - 1 - idx) * interval_seconds
        timestamp = end_time - timedelta(seconds=offset)

        points.append(
            TelemetryPoint(
                timestamp=timestamp,
                cpu_percent=round(cpu, 3),
                memory_percent=round(memory, 3),
                disk_percent=round(disk, 3),
                network_latency_ms=round(latency, 3),
                error_rate=round(error_rate, 5),
                active_connections=active_connections,
                queue_depth=queue_depth,
            )
        )

    return tuple(points)


def generate_network_degradation_scenario(
    *,
    scenario_family: str,
    instance: int,
    seed: int,
    device_id: str,
    n_points: int,
    interval_seconds: float,
    end_time: datetime,
) -> Scenario:
    """Generate one NETWORK_DEGRADATION scenario."""

    rng = random.Random(seed)

    if scenario_family == ND1_FAMILY:
        telemetry = _generate_persistent_latency_degradation(
            rng=rng,
            end_time=end_time,
            n_points=n_points,
            interval_seconds=interval_seconds,
        )

    elif scenario_family == ND2_FAMILY:
        telemetry = _generate_intermittent_severe_latency(
            rng=rng,
            end_time=end_time,
            n_points=n_points,
            interval_seconds=interval_seconds,
        )

    elif scenario_family == ND3_FAMILY:
        telemetry = _generate_degradation_with_request_errors(
            rng=rng,
            end_time=end_time,
            n_points=n_points,
            interval_seconds=interval_seconds,
        )

    else:
        raise ValueError(
            "Unsupported NETWORK_DEGRADATION scenario family: "
            f"{scenario_family}"
        )

    return Scenario(
        scenario_id=f"{FaultLabel.NETWORK_DEGRADATION.value}_{instance:03d}",
        scenario_family=scenario_family,
        device_id=device_id,
        fault=FaultLabel.NETWORK_DEGRADATION,
        seed=seed,
        telemetry=telemetry,
    )