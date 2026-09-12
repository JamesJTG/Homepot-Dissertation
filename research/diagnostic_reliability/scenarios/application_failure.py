"""Controlled application failure scenario families.

The application-failure fault class contains three families:

AF-1
    Service process failure.

AF-2
    Dependency or downstream service failure.

AF-3
    Intermittent application error burst.

The current diagnostic taxonomy still uses FaultLabel.APPLICATION_ERROR.
The scenario-family terminology uses APPLICATION_FAILURE because that is the
research-level fault class adopted for the dissertation.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import random

from .common import Scenario, TelemetryPoint, clamp
from ..taxonomy import FaultLabel


AF1_FAMILY = "AF-1_SERVICE_PROCESS_FAILURE"
AF2_FAMILY = "AF-2_DEPENDENCY_SERVICE_FAILURE"
AF3_FAMILY = "AF-3_APPLICATION_ERROR_BURST"


SUPPORTED_APPLICATION_FAMILIES = {
    AF1_FAMILY,
    AF2_FAMILY,
    AF3_FAMILY,
}


def _generate_service_process_failure(
    *,
    rng: random.Random,
    end_time: datetime,
    n_points: int,
    interval_seconds: float,
) -> tuple[TelemetryPoint, ...]:
    """Generate AF-1: service process failure.

    The service begins normally before an abrupt operational failure causes
    request errors to rise sharply.

    Queue depth increases and successful active connections fall after the
    failure point. CPU and memory remain controlled, and network latency stays
    far below the severe network-degradation regime.

    The dominant evidence is therefore an abrupt application/service failure
    rather than CPU, memory, or network saturation.
    """

    points: list[TelemetryPoint] = []

    failure_index = max(5, n_points // 3)

    for idx in range(n_points):
        if idx < failure_index:
            progress = idx / max(1, failure_index - 1)

            cpu_target = 49.0 + (3.0 * progress)
            memory_target = 53.0 + (2.0 * progress)
            latency_target = 60.0 + (20.0 * progress)
            error_target = 0.006
            connection_target = 25.0 + (3.0 * progress)
            queue_target = 6.0 + (2.0 * progress)

        else:
            failure_progress = (
                (idx - failure_index)
                / max(1, n_points - failure_index - 1)
            )

            cpu_target = 50.0 - (5.0 * failure_progress)
            memory_target = 55.0 + (3.0 * failure_progress)
            latency_target = 95.0 + (65.0 * failure_progress)

            error_target = 0.075 + (0.055 * failure_progress)

            connection_target = 24.0 - (10.0 * failure_progress)
            queue_target = 10.0 + (12.0 * failure_progress)

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

        latency = clamp(
            latency_target + rng.uniform(-20.0, 20.0),
            20.0,
            250.0,
        )

        error_rate = clamp(
            error_target + rng.uniform(-0.008, 0.008),
            0.0,
            0.16,
        )

        active_connections = int(
            round(
                clamp(
                    connection_target + rng.uniform(-2.0, 2.0),
                    8.0,
                    40.0,
                )
            )
        )

        queue_depth = int(
            round(
                clamp(
                    queue_target + rng.uniform(-1.5, 1.5),
                    2.0,
                    25.0,
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


def _generate_dependency_service_failure(
    *,
    rng: random.Random,
    end_time: datetime,
    n_points: int,
    interval_seconds: float,
) -> tuple[TelemetryPoint, ...]:
    """Generate AF-2: dependency or downstream service failure.

    The application itself initially appears healthy. A downstream dependency
    then becomes unreliable, producing sustained request failures and moderate
    response-time degradation.

    Latency is affected but deliberately remains below the severe values used
    by the NETWORK_DEGRADATION families. This allows the combination of error
    rate, queue accumulation and moderate latency to represent a dependency
    failure rather than a primary network fault.
    """

    points: list[TelemetryPoint] = []

    degradation_index = max(5, n_points // 4)

    for idx in range(n_points):
        if idx < degradation_index:
            progress = idx / max(1, degradation_index - 1)

            cpu_target = 48.0 + (3.0 * progress)
            memory_target = 52.0 + (2.0 * progress)
            latency_target = 65.0 + (25.0 * progress)
            error_target = 0.006
            connection_target = 23.0 + (3.0 * progress)
            queue_target = 6.0 + (2.0 * progress)

        else:
            dependency_progress = (
                (idx - degradation_index)
                / max(1, n_points - degradation_index - 1)
            )

            cpu_target = 51.0 + (7.0 * dependency_progress)
            memory_target = 54.0 + (5.0 * dependency_progress)

            latency_target = 125.0 + (125.0 * dependency_progress)

            error_target = 0.045 + (0.075 * dependency_progress)

            connection_target = 25.0 + (3.0 * dependency_progress)
            queue_target = 9.0 + (12.0 * dependency_progress)

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

        latency = clamp(
            latency_target + rng.uniform(-25.0, 25.0),
            20.0,
            320.0,
        )

        error_rate = clamp(
            error_target + rng.uniform(-0.007, 0.007),
            0.0,
            0.15,
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
                    queue_target + rng.uniform(-1.5, 1.5),
                    2.0,
                    25.0,
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


def _generate_application_error_burst(
    *,
    rng: random.Random,
    end_time: datetime,
    n_points: int,
    interval_seconds: float,
) -> tuple[TelemetryPoint, ...]:
    """Generate AF-3: intermittent application error bursts.

    Short episodes of severe request failure alternate with healthy or nearly
    healthy periods.

    Network latency remains broadly normal to moderately elevated, making the
    recurring error-rate pattern the dominant evidence.

    This family introduces temporal intermittency without relying on sustained
    resource saturation.
    """

    points: list[TelemetryPoint] = []

    burst_indices: set[int] = set()

    for start in (5, 11, 17):
        for idx in range(start, min(start + 3, n_points)):
            burst_indices.add(idx)

    recovery_indices = {
        8,
        9,
        14,
        15,
        20,
        21,
        22,
        23,
    }

    for idx in range(n_points):
        progress = idx / max(1, n_points - 1)

        if idx in burst_indices:
            cpu_target = 53.0 + (3.0 * progress)
            memory_target = 54.0 + (2.0 * progress)
            latency_target = 95.0 + (35.0 * progress)

            error_target = 0.115 + (0.020 * progress)

            connection_target = 23.0
            queue_target = 10.0 + (3.0 * progress)

        elif idx in recovery_indices:
            cpu_target = 50.0 + (2.0 * progress)
            memory_target = 53.0 + (2.0 * progress)
            latency_target = 75.0 + (20.0 * progress)

            error_target = 0.012

            connection_target = 24.0
            queue_target = 7.0

        else:
            cpu_target = 48.0 + (3.0 * progress)
            memory_target = 52.0 + (2.0 * progress)
            latency_target = 60.0 + (20.0 * progress)

            error_target = 0.006

            connection_target = 24.0
            queue_target = 6.0

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

        latency = clamp(
            latency_target + rng.uniform(-20.0, 20.0),
            20.0,
            220.0,
        )

        error_rate = clamp(
            error_target + rng.uniform(-0.008, 0.008),
            0.0,
            0.16,
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
                    18.0,
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


def generate_application_failure_scenario(
    *,
    scenario_family: str,
    instance: int,
    seed: int,
    device_id: str,
    n_points: int,
    interval_seconds: float,
    end_time: datetime,
) -> Scenario:
    """Generate one application-failure scenario."""

    rng = random.Random(seed)

    if scenario_family == AF1_FAMILY:
        telemetry = _generate_service_process_failure(
            rng=rng,
            end_time=end_time,
            n_points=n_points,
            interval_seconds=interval_seconds,
        )

    elif scenario_family == AF2_FAMILY:
        telemetry = _generate_dependency_service_failure(
            rng=rng,
            end_time=end_time,
            n_points=n_points,
            interval_seconds=interval_seconds,
        )

    elif scenario_family == AF3_FAMILY:
        telemetry = _generate_application_error_burst(
            rng=rng,
            end_time=end_time,
            n_points=n_points,
            interval_seconds=interval_seconds,
        )

    else:
        raise ValueError(
            "Unsupported application-failure scenario family: "
            f"{scenario_family}"
        )

    return Scenario(
        scenario_id=f"{FaultLabel.APPLICATION_ERROR.value}_{instance:03d}",
        scenario_family=scenario_family,
        device_id=device_id,
        fault=FaultLabel.APPLICATION_ERROR,
        seed=seed,
        telemetry=telemetry,
    )