"""Fixed experimental prompts and output schema."""

from .taxonomy import DiagnosisLabel

_ALLOWED = ", ".join(label.value for label in DiagnosisLabel)

SYSTEM_PROMPT = """You are the HOMEPOT experimental operational-diagnostics assistant.
Use only the supplied operational evidence. Do not invent missing measurements or
causes. Respect the validation status. If evidence is insufficient to support a
reliable diagnosis, select UNKNOWN. Return valid JSON only, with exactly the keys
diagnosis, confidence, and explanation. Confidence must be a number from 0 to 1.
"""

DIAGNOSTIC_PROMPT = f"""Identify the current operational condition of the specified device.
Select exactly one diagnosis from: {_ALLOWED}.
Use UNKNOWN when the supplied evidence is insufficient to support a reliable diagnosis.
Provide a concise evidence-based explanation.
"""
