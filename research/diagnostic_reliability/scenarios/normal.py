"""Controlled normal-operation scenario families.

The NORMAL class contains three families:

N-1
    Stable normal operation.

N-2
    High-but-valid workload.

N-3
    Brief benign transient.

The normal families are deliberately not all trivial. N-2 and N-3 introduce
challenging but non-fault conditions so that the diagnostic benchmark tests
whether unusual telemetry is incorrectly interpreted as evidence of a
persistent operational fault.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import random

from .common import Scenario, TelemetryPoint, clamp
from ..taxonomy import FaultLabel


N1_FAMILY = "N-1_STABLE_NORMAL_OPERATION"
N2_FAMILY = "N-2_HIGH_BUT_VALID_WORKLOAD"
N3_FAMILY = "N-3_BRIEF_BENIGN_TRANSIENT"


SUPPORTED_NORMAL_FAMILIES = {
    N1_FAMILY,
    N2_FAMILY,
    N3_FAMILY,
}


def _generate_stable_normal_operation(
    *,
    rng: random.Random,
    end_time: datetime,
    n_points: int,
    interval_seconds: float,
) -> tuple[TelemetryPoint, ...]:
    """Generate N-1: stable normal operation.

    All telemetry remains within an ordinary operating range with only small
    natural variation.

    This family acts as the clean negative-control condition.
    """

    points: list[TelemetryPoint] = []

    for idx in range(n_points):
        progress = idx / max(1, n_points - 1)

        cpu_target = 44.0 + (3.0 * progress)
        memory_target = 52.0 + (2.0 * progress)
        latency_target = 60.0 + (8.0 * progress)
        error_target = 0.006
        connection_target = 22.0 + (2.0 * progress)
        queue_target = 5.0 + (1.0 * progress)

        cpu = clamp(
            cpu_target + rng.uniform(-3.0, 3.0),
            25.0,
            65.0,
        )

        memory = clamp(
            memory_target + rng.uniform(-2.0, 2.0),
            35.0,
            65.0,
        )

        latency = clamp(
            latency_target + rng.uniform(-18.0, 18.0),
            20.0,
            150.0,
        )

        error_rate = clamp(
            error_target + rng.uniform(-0.003, 0.003),
            0.0,
            0.02,
        )

        active_connections = int(
            round(
                clamp(
                    connection_target + rng.uniform(-2.0, 2.0),
                    10.0,
                    35.0,
                )
            )
        )

        queue_depth = int(
            round(
                clamp(
                    queue_target + rng.uniform(-1.0, 1.0),
                    1.0,
                    10.0,
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


def _generate_high_but_valid_workload(
    *,
    rng: random.Random,
    end_time: datetime,
    n_points: int,
    interval_seconds: float,
) -> tuple[TelemetryPoint, ...]:
    """Generate N-2: sustained high-but-valid workload.

    CPU utilisation, memory use, connection count, queue depth and latency are
    higher than in N-1, representing a busy device or service.

    The workload remains stable and does not develop into one of the injected
    fault regimes. Error rate remains low.

    This family tests whether elevated resource utilisation alone causes a
    false-positive fault diagnosis.
    """

    points: list[TelemetryPoint] = []

    for idx in range(n_points):
        progress = idx / max(1, n_points - 1)

        cpu_target = 68.0 + (6.0 * progress)
        memory_target = 62.0 + (4.0 * progress)
        latency_target = 100.0 + (45.0 * progress)
        error_target = 0.009 + (0.003 * progress)
        connection_target = 34.0 + (8.0 * progress)
        queue_target = 9.0 + (4.0 * progress)

        cpu = clamp(
            cpu_target + rng.uniform(-3.0, 3.0),
            50.0,
            79.0,
        )

        memory = clamp(
            memory_target + rng.uniform(-2.0, 2.0),
            45.0,
            70.0,
        )

        latency = clamp(
            latency_target + rng.uniform(-25.0, 25.0),
            40.0,
            220.0,
        )

        error_rate = clamp(
            error_target + rng.uniform(-0.003, 0.003),
            0.0,
            0.025,
        )

        active_connections = int(
            round(
                clamp(
                    connection_target + rng.uniform(-3.0, 3.0),
                    20.0,
                    50.0,
                )
            )
        )

        queue_depth = int(
            round(
                clamp(
                    queue_target + rng.uniform(-1.5, 1.5),
                    4.0,
                    18.0,
                )
            )
        )

        disk = clamp(
            42.0 + rng.uniform(-5.0, 5.0),
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


def _generate_brief_benign_transient(
    *,
    rng: random.Random,
    end_time: datetime,
    n_points: int,
    interval_seconds: float,
) -> tuple[TelemetryPoint, ...]:
    """Generate N-3: brief benign transient followed by recovery.

    Most of the window represents normal operation.

    A short-lived workload disturbance temporarily elevates CPU, latency,
    queue depth and error rate. The disturbance then resolves without becoming
    a persistent operational fault.

    This family intentionally provides abnormal-looking individual
    observations while preserving NORMAL as the private ground-truth label.
    """

    points: list[TelemetryPoint] = []

    transient_start = max(5, n_points // 2 - 2)
    transient_end = min(transient_start + 3, n_points)

    for idx in range(n_points):
        if transient_start <= idx < transient_end:
            transient_progress = (
                idx - transient_start
            ) / max(1, transient_end - transient_start - 1)

            cpu_target = 84.0 + (5.0 * transient_progress)
            memory_target = 61.0 + (3.0 * transient_progress)
            latency_target = 280.0 + (90.0 * transient_progress)
            error_target = 0.028 + (0.012 * transient_progress)
            connection_target = 35.0 + (4.0 * transient_progress)
            queue_target = 14.0 + (4.0 * transient_progress)

        else:
            distance_after = idx - transient_end

            if 0 <= distance_after < 3:
                recovery_progress = (
                    distance_after / 2.0
                )

                cpu_target = 70.0 - (18.0 * recovery_progress)
                memory_target = 60.0 - (5.0 * recovery_progress)
                latency_target = 190.0 - (110.0 * recovery_progress)
                error_target = 0.018 - (0.010 * recovery_progress)
                connection_target = 30.0 - (5.0 * recovery_progress)
                queue_target = 10.0 - (4.0 * recovery_progress)

            else:
                cpu_target = 48.0
                memory_target = 54.0
                latency_target = 70.0
                error_target = 0.006
                connection_target = 24.0
                queue_target = 6.0

        cpu = clamp(
            cpu_target + rng.uniform(-2.5, 2.5),
            25.0,
            94.0,
        )

        memory = clamp(
            memory_target + rng.uniform(-1.8, 1.8),
            35.0,
            72.0,
        )

        latency = clamp(
            latency_target + rng.uniform(-25.0, 25.0),
            20.0,
            450.0,
        )

        error_rate = clamp(
            error_target + rng.uniform(-0.004, 0.004),
            0.0,
            0.05,
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


def generate_normal_scenario(
    *,
    scenario_family: str,
    instance: int,
    seed: int,
    device_id: str,
    n_points: int,
    interval_seconds: float,
    end_time: datetime,
) -> Scenario:
    """Generate one NORMAL scenario."""

    rng = random.Random(seed)

    if scenario_family == N1_FAMILY:
        telemetry = _generate_stable_normal_operation(
            rng=rng,
            end_time=end_time,
            n_points=n_points,
            interval_seconds=interval_seconds,
        )

    elif scenario_family == N2_FAMILY:
        telemetry = _generate_high_but_valid_workload(
            rng=rng,
            end_time=end_time,
            n_points=n_points,
            interval_seconds=interval_seconds,
        )

    elif scenario_family == N3_FAMILY:
        telemetry = _generate_brief_benign_transient(
            rng=rng,
            end_time=end_time,
            n_points=n_points,
            interval_seconds=interval_seconds,
        )

    else:
        raise ValueError(
            "Unsupported NORMAL scenario family: "
            f"{scenario_family}"
        )

    return Scenario(
        scenario_id=f"{FaultLabel.NORMAL.value}_{instance:03d}",
        scenario_family=scenario_family,
        device_id=device_id,
        fault=FaultLabel.NORMAL,
        seed=seed,
        telemetry=telemetry,
    )