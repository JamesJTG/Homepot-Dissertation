"""Controlled evidence-quality transformations for dissertation scenarios."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

from ai.gates.config import gate_b_kwargs

from .scenarios import Scenario, TelemetryPoint
from .taxonomy import EvidenceCondition, EvidenceSufficiency


def expected_sufficiency(condition: EvidenceCondition) -> EvidenceSufficiency:
    """Return the expected inference sufficiency under the primary protocol."""
    if condition is EvidenceCondition.COMPLETE_FRESH:
        return EvidenceSufficiency.SUFFICIENT
    return EvidenceSufficiency.INSUFFICIENT


def apply_evidence_condition(
    scenario: Scenario,
    condition: EvidenceCondition,
) -> Scenario:
    """Return a transformed copy while preserving the underlying ground truth.

    Transformations reference the current Gate B configuration so the
    dissertation harness cannot silently drift from HOMEPOT's deployed
    validation thresholds.
    """
    if condition is EvidenceCondition.COMPLETE_FRESH:
        return scenario

    config = gate_b_kwargs()
    points = list(scenario.telemetry)

    if condition is EvidenceCondition.STALE:
        max_age = int(config["freshness_max_age_seconds"])
        # Shift all observations far enough into the past to fail freshness,
        # retaining within-series continuity and the underlying fault pattern.
        shift = timedelta(seconds=max_age + 60)
        points = [replace(point, timestamp=point.timestamp - shift) for point in points]

    elif condition is EvidenceCondition.INCOMPLETE:
        # Null one required metric in enough rows to exceed Gate B's configured
        # null-ratio tolerance. The fault label remains unchanged in metadata.
        tolerance = float(config["completeness_max_null_ratio"])
        n = len(points)
        # Gate B denominator is total rows * 4 required fields. Null at least
        # floor(tolerance * denominator) + 1 values to force a fail.
        minimum_nulls = int(tolerance * n * 4) + 1
        for idx in range(min(minimum_nulls, n)):
            points[idx] = replace(points[idx], disk_percent=None)

    elif condition is EvidenceCondition.DISCONTINUOUS:
        max_gap = int(config["continuity_gap_seconds"])
        # Create one deliberate timestamp gap by moving the older half back.
        pivot = max(1, len(points) // 2)
        shift = timedelta(seconds=max_gap + 60)
        points = [
            replace(point, timestamp=point.timestamp - shift) if idx < pivot else point
            for idx, point in enumerate(points)
        ]

    else:  # defensive for future enum expansion
        raise ValueError(f"Unsupported evidence condition: {condition}")

    return replace(
        scenario,
        scenario_id=f"{scenario.scenario_id}_{condition.value}",
        telemetry=tuple(points),
    )
