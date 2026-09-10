"""Pydantic v2 data models for raw and corrected transcripts."""

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class RawSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    start: float
    end: float
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None
    temperature: float | None = None
    confidence: float | None = None
    window_index: int = 0


class TranscriptionMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    backend: str
    language: str
    word_timestamps: bool
    vad: dict[str, Any]
    created_at: str
    tool_version: str
    processing_seconds: float
    real_time_factor: float | None


class VideoMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str
    relative_path: str
    duration_seconds: float | None
    size_bytes: int | None


class RawTranscript(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    video: VideoMeta
    transcription: TranscriptionMeta
    segments: list[RawSegment]


class CorrectionInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["glossary_case", "rule", "llm"]
    reason: str
    confidence: float | None = None
    model: str | None = None
    rejected_reason: str | None = None


class CorrectedSegment(RawSegment):
    model_config = ConfigDict(extra="forbid")

    raw_text: str
    corrected: bool = False
    needs_review: bool = False
    flags: list[str] = Field(default_factory=list)
    correction: CorrectionInfo | None = None


class CorrectionMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    llm_model: str | None
    llm_endpoint_host: str | None
    glossary_version: int
    corrections_version: int
    created_at: str
    stats: dict[str, Any]


class CorrectedTranscript(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    video: VideoMeta
    transcription: TranscriptionMeta
    correction: CorrectionMeta
    segments: list[CorrectedSegment]


class CorrectionDiffItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    start: float
    end: float
    raw: str
    corrected: str
    source: str
    reason: str | None = None
    confidence: float | None = None
    accepted: bool
    rejected_reason: str | None = None
    flags: list[str] = Field(default_factory=list)


class CorrectionsFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    video: str
    stats: dict[str, Any]
    changes: list[CorrectionDiffItem]
