"""Experimental diagnosis taxonomy for the dissertation reliability study."""

from enum import Enum


class FaultLabel(str, Enum):
    """Ground-truth operational conditions used by the controlled experiment."""

    NORMAL = "NORMAL"
    CPU_SATURATION = "CPU_SATURATION"
    MEMORY_PRESSURE = "MEMORY_PRESSURE"
    APPLICATION_FAILURE = "APPLICATION_FAILURE"
    NETWORK_DEGRADATION = "NETWORK_DEGRADATION"


class DiagnosisLabel(str, Enum):
    """Labels the LLM is permitted to return."""

    NORMAL = FaultLabel.NORMAL.value
    CPU_SATURATION = FaultLabel.CPU_SATURATION.value
    MEMORY_PRESSURE = FaultLabel.MEMORY_PRESSURE.value
    APPLICATION_FAILURE = FaultLabel.APPLICATION_FAILURE.value
    NETWORK_DEGRADATION = FaultLabel.NETWORK_DEGRADATION.value
    UNKNOWN = "UNKNOWN"


class EvidenceCondition(str, Enum):
    """Controlled quality state applied independently of the fault condition."""

    COMPLETE_FRESH = "COMPLETE_FRESH"
    STALE = "STALE"
    INCOMPLETE = "INCOMPLETE"
    DISCONTINUOUS = "DISCONTINUOUS"


class EvidenceSufficiency(str, Enum):
    """Expected decision authority for a scenario."""

    SUFFICIENT = "SUFFICIENT"
    INSUFFICIENT = "INSUFFICIENT"
