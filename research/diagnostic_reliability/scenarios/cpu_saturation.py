"""Controlled CPU saturation scenario families.

The CPU_SATURATION fault class contains three distinct operational families:

CS-1
    Gradual processing overload.

CS-2
    Sudden sustained workload spike.

CS-3
    High-concurrency CPU pressure.

These families are intended to represent meaningfully different temporal
manifestations of the same private ground-truth fault.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import random

from .common import Scenario, TelemetryPoint, build_scenario_id, clamp
from ..taxonomy import FaultLabel


CS1_FAMILY = "CS-1_GRADUAL_PROCESSING_OVERLOAD"
CS2_FAMILY = "CS-2_SUDDEN_SUSTAINED_SPIKE"
CS3_FAMILY = "CS-3_HIGH_CONCURRENCY_PRESSURE"


SUPPORTED_CPU_FAMILIES = {
    CS1_FAMILY,
    CS2_FAMILY,
    CS3_FAMILY,
}


def _generate_cpu_gradual_overload(
    *,
    rng: random.Random,
    end_time: datetime,
    n_points: int,
    interval_seconds: float,
) -> tuple[TelemetryPoint, ...]:
    """Generate CS-1: gradual processing overload."""

    points: list[TelemetryPoint] = []

    for idx in range(n_points):
        progress = idx / (n_points - 1)

        cpu_target = 55.0 + (42.0 * progress)
        cpu = clamp(
            cpu_target + rng.uniform(-2.0, 2.0),
            20.0,
            99.0,
        )

        memory_target = 52.0 + (12.0 * progress)
        memory = clamp(
            memory_target + rng.uniform(-1.5, 1.5),
            39.8,
            70.2,
        )

        queue_target = 6.0 + (14.0 * progress)
        queue_depth = int(
            round(
                clamp(
                    queue_target + rng.uniform(-1.0, 1.0),
                    2.0,
                    20.0,
                )
            )
        )

        connection_target = 20.0 + (28.0 * progress)
        active_connections = int(
            round(
                clamp(
                    connection_target + rng.uniform(-2.0, 2.0),
                    10.0,
                    50.0,
                )
            )
        )

        latency_target = 55.0 + (75.0 * progress)
        latency = clamp(
            latency_target + rng.uniform(-8.0, 8.0),
            20.0,
            180.0,
        )

        error_rate = clamp(
            0.006
            + (0.008 * progress)
            + rng.uniform(-0.002, 0.002),
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


def _generate_cpu_sudden_spike(
    *,
    rng: random.Random,
    end_time: datetime,
    n_points: int,
    interval_seconds: float,
) -> tuple[TelemetryPoint, ...]:
    """Generate CS-2: sudden sustained workload spike."""

    points: list[TelemetryPoint] = []

    spike_index = max(3, n_points // 3)

    for idx in range(n_points):
        if idx < spike_index:
            cpu_target = 52.0
            memory_target = 53.0
            queue_target = 7.0
            connection_target = 22.0
            latency_target = 58.0
            error_target = 0.007

        else:
            post_progress = (
                (idx - spike_index)
                / max(1, n_points - spike_index - 1)
            )

            cpu_target = 94.0 + (2.0 * post_progress)
            memory_target = 56.0 + (6.0 * post_progress)
            queue_target = 13.0 + (7.0 * post_progress)
            connection_target = 34.0 + (14.0 * post_progress)
            latency_target = 82.0 + (42.0 * post_progress)
            error_target = 0.009 + (0.004 * post_progress)

        cpu = clamp(
            cpu_target + rng.uniform(-2.0, 2.0),
            20.0,
            99.0,
        )

        memory = clamp(
            memory_target + rng.uniform(-1.5, 1.5),
            39.8,
            70.2,
        )

        queue_depth = int(
            round(
                clamp(
                    queue_target + rng.uniform(-1.0, 1.0),
                    2.0,
                    20.0,
                )
            )
        )

        active_connections = int(
            round(
                clamp(
                    connection_target + rng.uniform(-2.0, 2.0),
                    10.0,
                    50.0,
                )
            )
        )

        latency = clamp(
            latency_target + rng.uniform(-7.0, 7.0),
            20.0,
            180.0,
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


def _generate_cpu_high_concurrency(
    *,
    rng: random.Random,
    end_time: datetime,
    n_points: int,
    interval_seconds: float,
) -> tuple[TelemetryPoint, ...]:
    """Generate CS-3: high-concurrency CPU pressure."""

    points: list[TelemetryPoint] = []

    pressure_index = max(4, n_points // 3)

    for idx in range(n_points):
        progress = idx / (n_points - 1)

        connection_target = 18.0 + (31.0 * progress)
        active_connections = int(
            round(
                clamp(
                    connection_target + rng.uniform(-1.5, 1.5),
                    10.0,
                    50.0,
                )
            )
        )

        queue_target = 5.0 + (15.0 * progress)
        queue_depth = int(
            round(
                clamp(
                    queue_target + rng.uniform(-1.0, 1.0),
                    2.0,
                    20.0,
                )
            )
        )

        if idx < pressure_index:
            pre_progress = idx / max(1, pressure_index - 1)
            cpu_target = 48.0 + (18.0 * pre_progress)

        else:
            pressure_progress = (
                (idx - pressure_index)
                / max(1, n_points - pressure_index - 1)
            )
            cpu_target = 68.0 + (29.0 * pressure_progress)

        cpu = clamp(
            cpu_target + rng.uniform(-2.0, 2.0),
            20.0,
            99.0,
        )

        memory_target = 51.0 + (13.0 * progress)
        memory = clamp(
            memory_target + rng.uniform(-1.5, 1.5),
            39.8,
            70.2,
        )

        latency_target = 52.0 + (80.0 * progress)
        latency = clamp(
            latency_target + rng.uniform(-7.0, 7.0),
            20.0,
            180.0,
        )

        error_rate = clamp(
            0.006
            + (0.007 * progress)
            + rng.uniform(-0.002, 0.002),
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


def generate_cpu_saturation_scenario(
    *,
    scenario_family: str,
    instance: int,
    seed: int,
    device_id: str,
    n_points: int,
    interval_seconds: float,
    end_time: datetime,
) -> Scenario:
    """Generate one CPU_SATURATION scenario."""

    rng = random.Random(seed)

    if scenario_family == CS1_FAMILY:
        telemetry = _generate_cpu_gradual_overload(
            rng=rng,
            end_time=end_time,
            n_points=n_points,
            interval_seconds=interval_seconds,
        )

    elif scenario_family == CS2_FAMILY:
        telemetry = _generate_cpu_sudden_spike(
            rng=rng,
            end_time=end_time,
            n_points=n_points,
            interval_seconds=interval_seconds,
        )

    elif scenario_family == CS3_FAMILY:
        telemetry = _generate_cpu_high_concurrency(
            rng=rng,
            end_time=end_time,
            n_points=n_points,
            interval_seconds=interval_seconds,
        )

    else:
        raise ValueError(
            "Unsupported CPU_SATURATION scenario family: "
            f"{scenario_family}"
        )

    return Scenario(
        scenario_id=build_scenario_id(
    fault=FaultLabel.CPU_SATURATION,
    scenario_family=scenario_family,
    instance=instance,
),
        scenario_family=scenario_family,
        device_id=device_id,
        fault=FaultLabel.CPU_SATURATION,
        seed=seed,
        telemetry=telemetry,
    )