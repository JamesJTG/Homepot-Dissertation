"""Controlled memory pressure scenario families.

The MEMORY_PRESSURE fault class contains three families:

MP-1
    Gradual memory accumulation.

MP-2
    Sustained high memory pressure.

MP-3
    Memory pressure with queue growth.

Historical HOMEPOT telemetry did not contain a true memory-pressure regime.
These scenarios are therefore controlled causal extensions that preserve the
existing telemetry structure while introducing known memory-related faults.

The families deliberately represent different temporal manifestations of
memory pressure. Absolute memory utilisation is therefore informative, but a
single fixed threshold is not intended to define every family.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import random

from .common import Scenario, TelemetryPoint, build_scenario_id, clamp
from ..taxonomy import FaultLabel


MP1_FAMILY = "MP-1_GRADUAL_MEMORY_ACCUMULATION"
MP2_FAMILY = "MP-2_SUSTAINED_HIGH_MEMORY_PRESSURE"
MP3_FAMILY = "MP-3_MEMORY_PRESSURE_WITH_QUEUE_GROWTH"


SUPPORTED_MEMORY_FAMILIES = {
    MP1_FAMILY,
    MP2_FAMILY,
    MP3_FAMILY,
}


def _generate_gradual_memory_accumulation(
    *,
    rng: random.Random,
    end_time: datetime,
    n_points: int,
    interval_seconds: float,
) -> tuple[TelemetryPoint, ...]:
    """Generate MP-1: gradual memory accumulation.

    Memory usage rises progressively from a normal operating range into a
    sustained elevated range. The diagnostic pattern is the persistent upward
    trajectory rather than crossing one severe absolute threshold.

    CPU, network latency and error rate rise only modestly so that memory is
    the dominant diagnostic signal.
    """

    points: list[TelemetryPoint] = []

    for idx in range(n_points):
        progress = idx / max(1, n_points - 1)

        memory_target = 58.0 + (30.0 * progress)

        cpu_target = 49.0 + (13.0 * progress)
        latency_target = 60.0 + (65.0 * progress)
        connection_target = 22.0 + (8.0 * progress)
        queue_target = 6.0 + (9.0 * progress)
        error_target = 0.006 + (0.008 * progress)

        memory = clamp(
            memory_target + rng.uniform(-1.8, 1.8),
            45.0,
            92.0,
        )

        cpu = clamp(
            cpu_target + rng.uniform(-2.0, 2.0),
            20.0,
            75.0,
        )

        latency = clamp(
            latency_target + rng.uniform(-20.0, 20.0),
            20.0,
            250.0,
        )

        active_connections = int(
            round(
                clamp(
                    connection_target + rng.uniform(-2.0, 2.0),
                    10.0,
                    45.0,
                )
            )
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

        error_rate = clamp(
            error_target + rng.uniform(-0.002, 0.002),
            0.0,
            0.025,
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


def _generate_sustained_high_memory_pressure(
    *,
    rng: random.Random,
    end_time: datetime,
    n_points: int,
    interval_seconds: float,
) -> tuple[TelemetryPoint, ...]:
    """Generate MP-2: sustained high memory pressure.

    Memory begins in an already elevated state and remains persistently high
    throughout the observation window.

    This differs from MP-1 because the diagnosis depends on persistence rather
    than a progressive accumulation trend.
    """

    points: list[TelemetryPoint] = []

    for idx in range(n_points):
        progress = idx / max(1, n_points - 1)

        memory_target = 89.0 + (3.0 * progress)

        cpu_target = 53.0 + (7.0 * progress)
        latency_target = 75.0 + (55.0 * progress)
        connection_target = 24.0 + (5.0 * progress)
        queue_target = 8.0 + (5.0 * progress)
        error_target = 0.007 + (0.006 * progress)

        memory = clamp(
            memory_target + rng.uniform(-2.5, 2.5),
            82.0,
            97.0,
        )

        cpu = clamp(
            cpu_target + rng.uniform(-2.0, 2.0),
            20.0,
            75.0,
        )

        latency = clamp(
            latency_target + rng.uniform(-20.0, 20.0),
            20.0,
            250.0,
        )

        active_connections = int(
            round(
                clamp(
                    connection_target + rng.uniform(-2.0, 2.0),
                    10.0,
                    45.0,
                )
            )
        )

        queue_depth = int(
            round(
                clamp(
                    queue_target + rng.uniform(-1.0, 1.0),
                    2.0,
                    18.0,
                )
            )
        )

        error_rate = clamp(
            error_target + rng.uniform(-0.002, 0.002),
            0.0,
            0.025,
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


def _generate_memory_pressure_with_queue_growth(
    *,
    rng: random.Random,
    end_time: datetime,
    n_points: int,
    interval_seconds: float,
) -> tuple[TelemetryPoint, ...]:
    """Generate MP-3: memory pressure with queue growth.

    Increasing memory utilisation develops alongside substantial queue
    accumulation. The diagnostic pattern therefore depends on the joint
    behaviour of memory and queue depth rather than an extreme memory
    threshold alone.

    The queue signal deliberately introduces overlap with other operational
    failure modes while memory remains the primary underlying fault.
    """

    points: list[TelemetryPoint] = []

    pressure_index = max(4, n_points // 4)

    for idx in range(n_points):
        if idx < pressure_index:
            progress = idx / max(1, pressure_index - 1)

            memory_target = 57.0 + (6.0 * progress)
            queue_target = 6.0 + (3.0 * progress)

            cpu_target = 49.0 + (3.0 * progress)
            latency_target = 60.0 + (25.0 * progress)
            connection_target = 22.0 + (3.0 * progress)
            error_target = 0.006

        else:
            progress = (
                (idx - pressure_index)
                / max(1, n_points - pressure_index - 1)
            )

            memory_target = 66.0 + (18.0 * progress)
            queue_target = 10.0 + (20.0 * progress)

            cpu_target = 52.0 + (13.0 * progress)
            latency_target = 85.0 + (90.0 * progress)
            connection_target = 25.0 + (10.0 * progress)
            error_target = 0.007 + (0.012 * progress)

        memory = clamp(
            memory_target + rng.uniform(-1.8, 1.8),
            45.0,
            90.0,
        )

        queue_depth = int(
            round(
                clamp(
                    queue_target + rng.uniform(-1.5, 1.5),
                    2.0,
                    32.0,
                )
            )
        )

        cpu = clamp(
            cpu_target + rng.uniform(-2.0, 2.0),
            20.0,
            78.0,
        )

        latency = clamp(
            latency_target + rng.uniform(-25.0, 25.0),
            20.0,
            300.0,
        )

        active_connections = int(
            round(
                clamp(
                    connection_target + rng.uniform(-2.0, 2.0),
                    10.0,
                    45.0,
                )
            )
        )

        error_rate = clamp(
            error_target + rng.uniform(-0.0025, 0.0025),
            0.0,
            0.03,
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


def generate_memory_pressure_scenario(
    *,
    scenario_family: str,
    instance: int,
    seed: int,
    device_id: str,
    n_points: int,
    interval_seconds: float,
    end_time: datetime,
) -> Scenario:
    """Generate one MEMORY_PRESSURE scenario."""

    rng = random.Random(seed)

    if scenario_family == MP1_FAMILY:
        telemetry = _generate_gradual_memory_accumulation(
            rng=rng,
            end_time=end_time,
            n_points=n_points,
            interval_seconds=interval_seconds,
        )

    elif scenario_family == MP2_FAMILY:
        telemetry = _generate_sustained_high_memory_pressure(
            rng=rng,
            end_time=end_time,
            n_points=n_points,
            interval_seconds=interval_seconds,
        )

    elif scenario_family == MP3_FAMILY:
        telemetry = _generate_memory_pressure_with_queue_growth(
            rng=rng,
            end_time=end_time,
            n_points=n_points,
            interval_seconds=interval_seconds,
        )

    else:
        raise ValueError(
            "Unsupported MEMORY_PRESSURE scenario family: "
            f"{scenario_family}"
        )

    return Scenario(
        scenario_id=build_scenario_id(
    fault=FaultLabel.MEMORY_PRESSURE,
    scenario_family=scenario_family,
    instance=instance,
),
        scenario_family=scenario_family,
        device_id=device_id,
        fault=FaultLabel.MEMORY_PRESSURE,
        seed=seed,
        telemetry=telemetry,
    )