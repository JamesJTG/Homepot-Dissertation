"""Frozen experimental prompts and diagnostic task specification."""

from __future__ import annotations

from .taxonomy import DiagnosisLabel


_ALLOWED = ", ".join(
    label.value
    for label in DiagnosisLabel
)


DIAGNOSTIC_CLASS_DEFINITIONS = {
    DiagnosisLabel.NORMAL: (
        "The available evidence is sufficient and supports normal operation. "
        "A temporary unusual condition may still be NORMAL when the evidence "
        "shows recovery rather than persistent operational degradation."
    ),
    DiagnosisLabel.CPU_SATURATION: (
        "A sustained or dominant processing-resource overload in which CPU "
        "pressure is the principal operational problem."
    ),
    DiagnosisLabel.MEMORY_PRESSURE: (
        "Persistent or progressive memory-resource pressure in which memory "
        "utilisation is the principal resource-pressure pattern."
    ),
    DiagnosisLabel.NETWORK_DEGRADATION: (
        "Persistent or intermittent degradation primarily associated with "
        "network behaviour, particularly abnormal latency and potentially "
        "associated request errors."
    ),
    DiagnosisLabel.APPLICATION_FAILURE: (
        "Application- or service-level failure behaviour, such as abnormal "
        "request errors, failed service behaviour, or queue accumulation, "
        "where the primary problem is not CPU, memory, or network-resource "
        "saturation."
    ),
    DiagnosisLabel.UNKNOWN: (
        "The available evidence is insufficient or too ambiguous to support "
        "a reliable diagnosis. UNKNOWN represents insufficient diagnostic "
        "certainty, not normal operation."
    ),
}


def _format_class_definitions() -> str:
    """Return the fixed semantic specification for the diagnostic taxonomy."""

    return "\n".join(
        (
            f"- {label.value}: "
            f"{DIAGNOSTIC_CLASS_DEFINITIONS[label]}"
        )
        for label in DiagnosisLabel
    )


VALIDATION_SEMANTICS = """
The validation status describes the readiness and integrity of the supplied
evidence. It does not describe the operational health of the device.

A grounded validation mode, trust_score of 1.0, actionable status, or absence
of a failed validation gate means that the evidence passed the configured
readiness checks. These values do not imply a NORMAL operational condition and
do not indicate that a diagnosis is correct.

Use validation information only to judge whether the supplied evidence is
suitable for diagnostic reasoning. Determine the operational condition from
the supplied operational evidence.
""".strip()


SYSTEM_PROMPT = f"""You are the HOMEPOT experimental operational-diagnostics assistant.

Use only the supplied operational evidence. Do not invent missing measurements
or causes.

{VALIDATION_SEMANTICS}

Distinguish normal operation from insufficient evidence: use NORMAL only when
the supplied evidence is sufficient and supports normal operation, and use
UNKNOWN when the evidence is insufficient or too ambiguous to support a
reliable diagnosis.

Return valid JSON only, with exactly the keys diagnosis, confidence, and
explanation. Confidence must be a number from 0 to 1.
"""


DIAGNOSTIC_PROMPT = f"""Identify the current operational condition of the specified device.

Select exactly one diagnosis from: {_ALLOWED}.

Use the following fixed semantic definitions:

{_format_class_definitions()}

Base the diagnosis only on the supplied operational evidence and validation
status. Do not assume missing measurements. Provide a concise evidence-based
explanation.
"""