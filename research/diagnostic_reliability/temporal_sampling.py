"""Deterministic temporal sampling for representation experiments.

The controlled scenario generators remain scientifically frozen. This module
creates research-only copies of existing scenarios with fewer model-visible
telemetry observations.

Sampling is deterministic and spans the complete original temporal window.
The first and final telemetry observations are always preserved.

No telemetry values are modified, interpolated, averaged, or synthesised.
Only a subset of existing observations is selected.
"""

from __future__ import annotations

from dataclasses import replace

from .scenarios.common import Scenario


def evenly_spaced_indices(
    n_observations: int,
    n_selected: int,
) -> tuple[int, ...]:
    """Return deterministic approximately evenly spaced observation indices.

    The first and final observations are always included.

    Example:
        24 observations reduced to 8 observations.

    No sampling randomness is used.
    """

    if n_observations < 1:
        raise ValueError(
            "n_observations must be at least 1."
        )

    if n_selected < 1:
        raise ValueError(
            "n_selected must be at least 1."
        )

    if n_selected > n_observations:
        raise ValueError(
            "n_selected cannot exceed n_observations."
        )

    if n_selected == n_observations:
        return tuple(
            range(n_observations)
        )

    if n_selected == 1:
        return (0,)

    indices = tuple(
        round(
            position
            * (n_observations - 1)
            / (n_selected - 1)
        )
        for position in range(
            n_selected
        )
    )

    if indices[0] != 0:
        raise RuntimeError(
            "Temporal sampling failed to preserve "
            "the first observation."
        )

    if indices[-1] != (
        n_observations - 1
    ):
        raise RuntimeError(
            "Temporal sampling failed to preserve "
            "the final observation."
        )

    if len(set(indices)) != len(indices):
        raise RuntimeError(
            "Temporal sampling produced duplicate indices."
        )

    return indices


def downsample_scenario(
    scenario: Scenario,
    *,
    n_selected: int,
) -> Scenario:
    """Return a scenario copy containing selected existing observations.

    Scenario identity, family, device, private ground truth and seed remain
    unchanged. Only the telemetry tuple is replaced with a deterministic
    subset of the original telemetry observations.

    The source scenario itself is never mutated.
    """

    indices = evenly_spaced_indices(
        len(scenario.telemetry),
        n_selected,
    )

    selected_telemetry = tuple(
        scenario.telemetry[index]
        for index in indices
    )

    return replace(
        scenario,
        telemetry=selected_telemetry,
    )