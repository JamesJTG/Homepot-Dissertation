"""Controlled Ollama adapter for dissertation diagnostic experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import ollama
import yaml

from .taxonomy import DiagnosisLabel


DEFAULT_CONFIG_PATH = Path("ai/config.yaml")


def _load_llm_config(config_path: Path) -> dict[str, Any]:
    """Load the HOMEPOT LLM configuration."""
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    try:
        return config["llm"]
    except (TypeError, KeyError) as exc:
        raise ValueError("Missing 'llm' configuration section") from exc


def _diagnostic_schema() -> dict[str, Any]:
    """Return the exact structured-output schema used by the experiment."""
    return {
        "type": "object",
        "properties": {
            "diagnosis": {
                "type": "string",
                "enum": [label.value for label in DiagnosisLabel],
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "explanation": {
                "type": "string",
                "minLength": 1,
            },
        },
        "required": [
            "diagnosis",
            "confidence",
            "explanation",
        ],
        "additionalProperties": False,
    }


class ExperimentalOllamaAdapter:
    """Controlled Ollama interface for diagnostic reliability experiments.

    This adapter deliberately differs from the production HOMEPOT LLM wrapper:

    - generation settings are explicit;
    - structured JSON output is requested through Ollama;
    - failures are raised rather than converted into conversational text;
    - the interface remains compatible with ``run_repeated_generations``.
    """

    def __init__(
        self,
        *,
        temperature: float,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ) -> None:
        if temperature < 0.0:
            raise ValueError("temperature must be non-negative")

        config_path = Path(config_path)
        llm_config = _load_llm_config(config_path)

        self.model = str(llm_config["model"])
        self.base_url = str(llm_config["base_url"])
        self.context_window = int(llm_config.get("context_window", 4096))
        self.temperature = float(temperature)

        self.client = ollama.Client(host=self.base_url)

    def check_health(self) -> bool:
        """Return True when the configured Ollama service is reachable."""
        try:
            self.client.list()
            return True
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        context: str,
        system_prompt: str,
    ) -> str:
        """Generate one structured diagnostic response.

        The method signature intentionally matches the injectable callable
        expected by ``run_repeated_generations``.
        """
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {prompt}",
            },
        ]

        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                format=_diagnostic_schema(),
                options={
                    "temperature": self.temperature,
                    "num_ctx": self.context_window,
                },
            )
        except Exception as exc:
            raise RuntimeError(
                f"Ollama generation failed for model {self.model!r}"
            ) from exc

        content = str(response["message"]["content"]).strip()

        if not content:
            raise RuntimeError("Ollama returned an empty diagnostic response")

        return content