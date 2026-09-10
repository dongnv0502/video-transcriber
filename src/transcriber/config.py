"""Configuration management with multi-tier precedence (CLI > Env > YAML > Default)."""

import os
from pathlib import Path
from typing import Any
import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigError(Exception):
    """Raised when configuration validation fails."""
    pass


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TRANSCRIBER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Directories
    video_dir: Path = Path("./videos")
    output_dir: Path = Path("./output")
    work_dir: Path = Path("./work")
    db_path: Path = Path("./state.db")
    log_dir: Path = Path("./logs")

    # ASR Model settings
    model_name: str = "mlx-community/whisper-large-v3-mlx"
    model_backend: str = "mlx"  # mlx | faster-whisper
    model_name_fw: str | None = None
    language: str = "vi"
    initial_prompt: str = (
        "Bài giảng AWS tiếng Việt. Thuật ngữ: IAM, EC2, S3, VPC, CloudFormation, "
        "Lambda, ECS, EKS, CloudWatch."
    )
    word_timestamps: bool = False

    # VAD settings
    vad_enabled: bool = True
    vad_threshold: float = 0.5
    vad_min_silence_ms: int = 700
    vad_speech_pad_ms: int = 300
    vad_min_speech_ms: int = 250
    vad_merge_gap_s: float = 1.0
    vad_max_window_s: float = 300.0
    keep_audio: bool = True

    # LLM Correction settings (OpenAI-compatible HTTP endpoint)
    llm_enabled: bool = True
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None
    llm_temperature: float = 0.0
    llm_batch_size: int = 6
    llm_response_format: str = "json_schema"  # json_schema | json_object
    llm_timeout_s: float = 120.0
    llm_max_retries: int = 5

    # Workers
    asr_workers: int = 1
    llm_workers: int = 4

    def validate_llm(self) -> None:
        """Validate LLM settings if LLM correction is enabled."""
        if self.llm_enabled:
            missing: list[str] = []
            if not self.llm_base_url:
                missing.append("TRANSCRIBER_LLM_BASE_URL")
            if not self.llm_model:
                missing.append("TRANSCRIBER_LLM_MODEL")
            if missing:
                raise ConfigError(
                    f"Set {' and '.join(missing)}, or run with --no-llm."
                )

    @classmethod
    def load(cls, config_path: Path | str | None = None, **cli_overrides: Any) -> "Settings":
        """Load settings implementing CLI > Environment > config.yml > Default precedence.

        1. Read YAML dict if config file exists.
        2. Filter out YAML keys present in os.environ (prefixed with TRANSCRIBER_).
        3. Pass remaining YAML keys as base init kwargs.
        4. Apply explicit non-None cli_overrides on top.
        """
        yaml_data: dict[str, Any] = {}

        # Locate config file
        resolved_config_path: Path | None = None
        if config_path:
            p = Path(config_path)
            if p.exists():
                resolved_config_path = p
        else:
            for default_name in ("config.yml", "config.yaml"):
                p = Path(default_name)
                if p.exists():
                    resolved_config_path = p
                    break

        if resolved_config_path and resolved_config_path.exists():
            with open(resolved_config_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    yaml_data = loaded

        # Drop YAML keys overridden by TRANSCRIBER_<KEY> in environment
        filtered_yaml: dict[str, Any] = {}
        for k, v in yaml_data.items():
            env_key = f"TRANSCRIBER_{k.upper()}"
            if env_key not in os.environ:
                filtered_yaml[k] = v

        # Instantiate settings with filtered YAML as initial values
        # Pydantic-settings will still pull in env vars for unspecified fields
        settings = cls(**filtered_yaml)

        # Apply CLI overrides (non-None values override everything)
        for k, v in cli_overrides.items():
            if v is not None and hasattr(settings, k):
                setattr(settings, k, v)

        return settings
