"""Integration test for crash recovery and resume behavior."""

import json
import sqlite3
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from transcriber.config import Settings
from transcriber.db import (
    COMPLETED,
    FAILED,
    NEEDS_REVIEW,
    PENDING,
    TRANSCRIBED,
    init_db,
    set_status,
)
from transcriber.models import RawTranscript
from transcriber.paths import atomic_write_json
from transcriber.stages.correct import correct_video
from transcriber.stages.export import export_video
from transcriber.stages.transcribe import transcribe_video


@pytest.mark.integration
def test_resume_preserves_raw_json_and_advances_failed_video(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(
        video_dir=tmp_path / "videos",
        output_dir=tmp_path / "output",
        work_dir=tmp_path / "work",
        db_path=tmp_path / "state.db",
        log_dir=tmp_path / "logs",
        llm_enabled=False,
    )
    conn = init_db(settings.db_path)

    # Load fixture raw transcript
    with open("tests/fixtures/raw_sample.json", "r", encoding="utf-8") as f:
        fixture_data = json.load(f)

    # 1. Seed video 1: previously transcribed, but FAILED at correct stage
    slug1 = "video-one"
    out_dir1 = settings.output_dir / slug1
    raw_path1 = out_dir1 / "raw.json"
    atomic_write_json(raw_path1, fixture_data)
    mtime_before = raw_path1.stat().st_mtime

    cur1 = conn.execute(
        """
        INSERT INTO videos (path, filename, slug, status, failed_stage, duration_seconds, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
        """,
        (f"/videos/{slug1}.mp4", f"{slug1}.mp4", slug1, FAILED, "correct", 134.0),
    )
    vid1 = cur1.lastrowid

    # 2. Seed video 2: PENDING
    slug2 = "video-two"
    cur2 = conn.execute(
        """
        INSERT INTO videos (path, filename, slug, status, duration_seconds, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, datetime('now'), datetime('now'))
        """,
        (f"/videos/{slug2}.mp4", f"{slug2}.mp4", slug2, PENDING, 60.0),
    )
    vid2 = cur2.lastrowid

    # Mock transcribe_video backend to return fixture segments without running Whisper
    def mock_transcribe(*args, **kwargs):
        row = args[2]
        out_p = settings.output_dir / row["slug"] / "raw.json"
        atomic_write_json(out_p, fixture_data)
        set_status(conn, row["id"], TRANSCRIBED, stage="transcribe")
        return RawTranscript.model_validate(fixture_data)

    monkeypatch.setattr("transcriber.stages.transcribe.transcribe_video", mock_transcribe)

    # --- Run resume step 1: transcribe ---
    # Query transcribe candidates: video1 was FAILED at 'correct' so it is NOT selected for transcription!
    from transcriber.db import get_videos_for_correct, get_videos_for_transcribe

    transcribe_candidates = get_videos_for_transcribe(conn)
    transcribe_slugs = [r["slug"] for r in transcribe_candidates]
    assert slug1 not in transcribe_slugs  # raw.json of video 1 will NOT be touched
    assert slug2 in transcribe_slugs

    for r in transcribe_candidates:
        mock_transcribe(conn, settings, r)

    # Check raw_path1 mtime unchanged
    mtime_after = raw_path1.stat().st_mtime
    assert mtime_before == mtime_after

    # --- Run resume step 2: correct ---
    correct_candidates = get_videos_for_correct(conn)
    correct_slugs = [r["slug"] for r in correct_candidates]
    assert slug1 in correct_slugs  # video 1 picked up for correction!

    for r in correct_candidates:
        correct_video(conn, settings, r)

    # --- Run resume step 3: export ---
    for r in [conn.execute("SELECT * FROM videos WHERE id = ?", (vid1,)).fetchone(),
             conn.execute("SELECT * FROM videos WHERE id = ?", (vid2,)).fetchone()]:
        export_video(settings, r, conn=conn)

    # Assert both videos reached terminal status (COMPLETED or NEEDS_REVIEW)
    row1 = conn.execute("SELECT status FROM videos WHERE id = ?", (vid1,)).fetchone()
    row2 = conn.execute("SELECT status FROM videos WHERE id = ?", (vid2,)).fetchone()
    assert row1["status"] in (COMPLETED, NEEDS_REVIEW)
    assert row2["status"] in (COMPLETED, NEEDS_REVIEW)
