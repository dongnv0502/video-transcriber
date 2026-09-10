"""Unit tests for config.py: precedence (CLI > Env > YAML > Default) and validation."""

import os
from pathlib import Path

import pytest
import yaml

from transcriber.config import ConfigError, Settings


def test_default_settings() -> None:
    settings = Settings()
    assert settings.language == "vi"
    assert settings.vad_enabled is True
    assert settings.llm_batch_size == 6


def test_yaml_config_loading(tmp_path: Path) -> None:
    cfg_file = tmp_path / "custom_config.yml"
    cfg_data = {
        "language": "en",
        "vad_min_silence_ms": 900,
        "llm_batch_size": 10,
    }
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_data, f)

    settings = Settings.load(cfg_file)
    assert settings.language == "en"
    assert settings.vad_min_silence_ms == 900
    assert settings.llm_batch_size == 10


def test_env_precedence_over_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_file = tmp_path / "custom_config.yml"
    cfg_data = {"language": "en", "vad_min_silence_ms": 900}
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_data, f)

    # Set environment variable
    monkeypatch.setenv("TRANSCRIBER_LANGUAGE", "fr")

    settings = Settings.load(cfg_file)
    assert settings.language == "fr"  # Env overrides YAML
    assert settings.vad_min_silence_ms == 900  # YAML applies where env is absent


def test_cli_precedence_over_env_and_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_file = tmp_path / "custom_config.yml"
    cfg_data = {"language": "en"}
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_data, f)

    monkeypatch.setenv("TRANSCRIBER_LANGUAGE", "fr")

    settings = Settings.load(cfg_file, language="vi")
    assert settings.language == "vi"  # CLI override wins over both env and YAML


def test_validate_llm_raises_when_missing_config() -> None:
    # llm_enabled is True by default, but base_url and model are None
    settings = Settings(llm_base_url=None, llm_model=None)
    with pytest.raises(ConfigError) as exc_info:
        settings.validate_llm()
    assert "TRANSCRIBER_LLM_BASE_URL" in str(exc_info.value)
    assert "TRANSCRIBER_LLM_MODEL" in str(exc_info.value)

    # Valid settings
    valid_settings = Settings(
        llm_base_url="https://api.openai.com/v1",
        llm_model="gpt-4o-mini",
    )
    valid_settings.validate_llm()  # Should not raise
