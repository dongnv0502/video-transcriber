"""Unit tests for export.py: SRT format, HMS format, and Markdown shape."""

import sqlite3
from pathlib import Path

from transcriber.config import Settings
from transcriber.models import (
    CorrectedSegment,
    CorrectedTranscript,
    CorrectionMeta,
    RawSegment,
    RawTranscript,
    TranscriptionMeta,
    VideoMeta,
)
from transcriber.stages.export import export_video, format_hms, format_srt_time, slug_to_title


def test_srt_time_formatting() -> None:
    # 0 seconds
    assert format_srt_time(0.0) == "00:00:00,000"
    # Millisecond rounding
    assert format_srt_time(1.2351) == "00:00:01,235"
    # > 1 hour
    assert format_srt_time(3665.5) == "01:01:05,500"


def test_hms_formatting() -> None:
    assert format_hms(0) == "00:00:00"
    assert format_hms(134) == "00:02:14"
    assert format_hms(3665) == "01:01:05"


def test_slug_to_title() -> None:
    assert slug_to_title("001-introduction-to-iam-and-s3") == "001 Introduction To IAM And S3"
    assert slug_to_title("002-ec2-vpc-networking") == "002 EC2 VPC Networking"


def test_markdown_and_srt_export_shape(tmp_path: Path) -> None:
    settings = Settings(output_dir=tmp_path / "output")
    slug = "001-intro"
    row = {"slug": slug, "id": 1}

    # Build sample transcript with one needs_review segment
    video_meta = VideoMeta(
        filename="001-intro.mp4",
        relative_path="videos/001-intro.mp4",
        duration_seconds=120.0,
        size_bytes=1000,
    )
    tx_meta = TranscriptionMeta(
        model="mlx-community/whisper-large-v3-mlx",
        backend="mlx",
        language="vi",
        word_timestamps=False,
        vad={},
        created_at="2026-09-11T00:00:00Z",
        tool_version="0.1.0",
        processing_seconds=5.0,
        real_time_factor=0.04,
    )
    corr_meta = CorrectionMeta(
        enabled=True,
        llm_model="test-model",
        llm_endpoint_host="localhost",
        glossary_version=1,
        corrections_version=1,
        created_at="2026-09-11T00:00:00Z",
        stats={},
    )
    segments = [
        CorrectedSegment(
            id=0,
            start=0.0,
            end=4.0,
            text="chào mừng các bạn",
            raw_text="chào mừng các bạn",
            needs_review=False,
        ),
        CorrectedSegment(
            id=1,
            start=4.5,
            end=9.0,
            text="cấu hình EC2",
            raw_text="cấu hình ec hai",
            needs_review=True,
        ),
    ]

    transcript = CorrectedTranscript(
        video=video_meta,
        transcription=tx_meta,
        correction=corr_meta,
        segments=segments,
    )

    out_p = export_video(settings, row, transcript=transcript)
    assert out_p.exists()

    md_file = out_p / "transcript.md"
    srt_file = out_p / "transcript.srt"
    txt_file = out_p / "transcript.txt"

    assert md_file.exists()
    assert srt_file.exists()
    assert txt_file.exists()

    md_text = md_file.read_text(encoding="utf-8")
    assert "# 001 Intro" in md_text
    assert "- segments: 2" in md_text
    assert "- needs_review: 1" in md_text
    assert "[00:00:00 - 00:00:04]" in md_text
    assert "[00:00:04 - 00:00:09] <!-- needs_review -->" in md_text

    srt_text = srt_file.read_text(encoding="utf-8")
    assert "1\n00:00:00,000 --> 00:00:04,000\nchào mừng các bạn" in srt_text
    assert "2\n00:00:04,500 --> 00:00:09,000\ncấu hình EC2" in srt_text
