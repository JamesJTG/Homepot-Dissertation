"""Controlled diagnostic scenario generation."""

from __future__ import annotations

from datetime import datetime, timezone

from ..taxonomy import FaultLabel
from .common import Scenario, TelemetryPoint

from .cpu_saturation import (
    CS1_FAMILY,
    CS2_FAMILY,
    CS3_FAMILY,
    generate_cpu_saturation_scenario,
)

from .network_degradation import (
    ND1_FAMILY,
    ND2_FAMILY,
    ND3_FAMILY,
    generate_network_degradation_scenario,
)

from .memory_pressure import (
    MP1_FAMILY,
    MP2_FAMILY,
    MP3_FAMILY,
    generate_memory_pressure_scenario,
)

from .application_failure import (
    AF1_FAMILY,
    AF2_FAMILY,
    AF3_FAMILY,
    generate_application_failure_scenario,
)

from .normal import (
    N1_FAMILY,
    N2_FAMILY,
    N3_FAMILY,
    generate_normal_scenario,
)


DEFAULT_N_POINTS = 24
DEFAULT_INTERVAL_SECONDS = 2.3


def generate_scenario(
    fault: FaultLabel,
    *,
    instance: int = 1,
    seed: int = 42,
    device_id: str = "controlled-device-001",
    scenario_family: str | None = None,
    n_points: int = DEFAULT_N_POINTS,
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
    end_time: datetime | None = None,
) -> Scenario:
    """Generate one controlled diagnostic scenario.

    The fault may be passed positionally or as a keyword for backwards
    compatibility with the original M6 vertical-slice API.
    """

    if n_points < 2:
        raise ValueError("n_points must be at least 2.")

    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be greater than zero.")

    if end_time is None:
        end_time = datetime.now(timezone.utc)

    if fault == FaultLabel.CPU_SATURATION:
        family = scenario_family or CS1_FAMILY

        return generate_cpu_saturation_scenario(
            scenario_family=family,
            instance=instance,
            seed=seed,
            device_id=device_id,
            n_points=n_points,
            interval_seconds=interval_seconds,
            end_time=end_time,
        )

    if fault == FaultLabel.NETWORK_DEGRADATION:
        family = scenario_family or ND1_FAMILY

        return generate_network_degradation_scenario(
            scenario_family=family,
            instance=instance,
            seed=seed,
            device_id=device_id,
            n_points=n_points,
            interval_seconds=interval_seconds,
            end_time=end_time,
        )

    if fault == FaultLabel.MEMORY_PRESSURE:
        family = scenario_family or MP1_FAMILY

        return generate_memory_pressure_scenario(
            scenario_family=family,
            instance=instance,
            seed=seed,
            device_id=device_id,
            n_points=n_points,
            interval_seconds=interval_seconds,
            end_time=end_time,
        )

    if fault == FaultLabel.APPLICATION_FAILURE:
        family = scenario_family or AF1_FAMILY

        return generate_application_failure_scenario(
            scenario_family=family,
            instance=instance,
            seed=seed,
            device_id=device_id,
            n_points=n_points,
            interval_seconds=interval_seconds,
            end_time=end_time,
        )

    if fault == FaultLabel.NORMAL:
        family = scenario_family or N1_FAMILY

        return generate_normal_scenario(
            scenario_family=family,
            instance=instance,
            seed=seed,
            device_id=device_id,
            n_points=n_points,
            interval_seconds=interval_seconds,
            end_time=end_time,
        )

    raise NotImplementedError(
        f"Controlled scenario generation is not yet implemented for {fault.value}."
    )


__all__ = [
    "Scenario",
    "TelemetryPoint",
    "generate_scenario",
    "CS1_FAMILY",
    "CS2_FAMILY",
    "CS3_FAMILY",
    "ND1_FAMILY",
    "ND2_FAMILY",
    "ND3_FAMILY",
    "MP1_FAMILY",
    "MP2_FAMILY",
    "MP3_FAMILY",
    "AF1_FAMILY",
    "AF2_FAMILY",
    "AF3_FAMILY",
    "N1_FAMILY",
    "N2_FAMILY",
    "N3_FAMILY",
]