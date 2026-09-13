"""Tests for the controlled dissertation Ollama adapter."""

from __future__ import annotations

import json

import pytest

from research.diagnostic_reliability.ollama_adapter import (
    ExperimentalOllamaAdapter,
    _diagnostic_schema,
)
from research.diagnostic_reliability.taxonomy import DiagnosisLabel


def test_schema_contains_all_diagnosis_labels() -> None:
    schema = _diagnostic_schema()

    assert schema["properties"]["diagnosis"]["enum"] == [
        label.value for label in DiagnosisLabel
    ]


def test_schema_rejects_additional_properties() -> None:
    schema = _diagnostic_schema()

    assert schema["additionalProperties"] is False


def test_negative_temperature_is_rejected() -> None:
    with pytest.raises(ValueError):
        ExperimentalOllamaAdapter(
            temperature=-0.1
        )


def test_generate_uses_controlled_settings(
    monkeypatch,
    tmp_path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
llm:
  model: "test-model"
  base_url: "http://localhost:11434"
  context_window: 2048
""".strip(),
        encoding="utf-8",
    )

    captured = {}

    class FakeClient:
        def __init__(self, host: str):
            captured["host"] = host

        def chat(self, **kwargs):
            captured["chat"] = kwargs

            return {
                "message": {
                    "content": json.dumps(
                        {
                            "diagnosis": "NORMAL",
                            "confidence": 0.8,
                            "explanation": "Normal telemetry.",
                        }
                    )
                }
            }

    monkeypatch.setattr(
        (
            "research.diagnostic_reliability."
            "ollama_adapter.ollama.Client"
        ),
        FakeClient,
    )

    adapter = ExperimentalOllamaAdapter(
        temperature=0.6,
        config_path=config_path,
    )

    result = adapter.generate(
        "diagnose",
        "telemetry evidence",
        "system instructions",
    )

    assert captured["host"] == (
        "http://localhost:11434"
    )

    call = captured["chat"]

    # No research override was supplied, so the HOMEPOT-configured
    # model must be used.
    assert adapter.model == "test-model"
    assert call["model"] == "test-model"

    assert (
        call["options"]["temperature"]
        == 0.6
    )

    assert (
        call["options"]["num_ctx"]
        == 2048
    )

    assert (
        call["format"]
        == _diagnostic_schema()
    )

    assert call["messages"][0] == {
        "role": "system",
        "content": "system instructions",
    }

    assert (
        "telemetry evidence"
        in call["messages"][1]["content"]
    )

    assert (
        "diagnose"
        in call["messages"][1]["content"]
    )

    assert (
        '"diagnosis": "NORMAL"'
        in result
    )


def test_explicit_model_override_is_research_local(
    monkeypatch,
    tmp_path,
) -> None:
    config_path = tmp_path / "config.yaml"

    config_path.write_text(
        """
llm:
  model: "configured-homepot-model"
  base_url: "http://localhost:11434"
  context_window: 2048
""".strip(),
        encoding="utf-8",
    )

    captured = {}

    class FakeClient:
        def __init__(self, host: str):
            captured["host"] = host

        def chat(self, **kwargs):
            captured["chat"] = kwargs

            return {
                "message": {
                    "content": json.dumps(
                        {
                            "diagnosis": "NORMAL",
                            "confidence": 0.8,
                            "explanation": "Normal telemetry.",
                        }
                    )
                }
            }

    monkeypatch.setattr(
        (
            "research.diagnostic_reliability."
            "ollama_adapter.ollama.Client"
        ),
        FakeClient,
    )

    adapter = ExperimentalOllamaAdapter(
        temperature=0.4,
        model="research-model",
        config_path=config_path,
    )

    adapter.generate(
        "diagnose",
        "telemetry evidence",
        "system instructions",
    )

    assert adapter.model == "research-model"

    assert (
        captured["chat"]["model"]
        == "research-model"
    )

    # The configuration file itself remains unchanged.
    assert (
        'model: "configured-homepot-model"'
        in config_path.read_text(
            encoding="utf-8"
        )
    )


def test_empty_model_override_is_rejected(
    tmp_path,
) -> None:
    config_path = tmp_path / "config.yaml"

    config_path.write_text(
        """
llm:
  model: "configured-homepot-model"
  base_url: "http://localhost:11434"
  context_window: 2048
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="model must not be empty",
    ):
        ExperimentalOllamaAdapter(
            temperature=0.4,
            model="",
            config_path=config_path,
        )


def test_generation_failure_is_raised(
    monkeypatch,
    tmp_path,
) -> None:
    config_path = tmp_path / "config.yaml"

    config_path.write_text(
        """
llm:
  model: "test-model"
  base_url: "http://localhost:11434"
  context_window: 2048
""".strip(),
        encoding="utf-8",
    )

    class FakeClient:
        def __init__(self, host: str):
            pass

        def chat(self, **kwargs):
            raise RuntimeError(
                "connection failed"
            )

    monkeypatch.setattr(
        (
            "research.diagnostic_reliability."
            "ollama_adapter.ollama.Client"
        ),
        FakeClient,
    )

    adapter = ExperimentalOllamaAdapter(
        temperature=0.6,
        config_path=config_path,
    )

    with pytest.raises(
        RuntimeError,
        match="Ollama generation failed",
    ):
        adapter.generate(
            "diagnose",
            "telemetry evidence",
            "system instructions",
        )


def test_empty_generation_is_rejected(
    monkeypatch,
    tmp_path,
) -> None:
    config_path = tmp_path / "config.yaml"

    config_path.write_text(
        """
llm:
  model: "test-model"
  base_url: "http://localhost:11434"
  context_window: 2048
""".strip(),
        encoding="utf-8",
    )

    class FakeClient:
        def __init__(self, host: str):
            pass

        def chat(self, **kwargs):
            return {
                "message": {
                    "content": "   "
                }
            }

    monkeypatch.setattr(
        (
            "research.diagnostic_reliability."
            "ollama_adapter.ollama.Client"
        ),
        FakeClient,
    )

    adapter = ExperimentalOllamaAdapter(
        temperature=0.6,
        config_path=config_path,
    )

    with pytest.raises(
        RuntimeError,
        match="empty diagnostic response",
    ):
        adapter.generate(
            "diagnose",
            "telemetry evidence",
            "system instructions",
        )