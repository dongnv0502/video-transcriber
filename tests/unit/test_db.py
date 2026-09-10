"""Unit tests for db.py: status transitions, history logging, and resume queries."""

import sqlite3

from transcriber.db import (
    COMPLETED,
    EXTRACTING,
    FAILED,
    NEEDS_REVIEW,
    PENDING,
    TRANSCRIBED,
    TRANSCRIBING,
    Status,
    get_status_counts,
    get_videos_for_correct,
    get_videos_for_transcribe,
    set_status,
)


def seed_video(conn: sqlite3.Connection, slug: str, status: str, failed_stage: str | None = None) -> int:
    now = "2026-09-11T00:00:00Z"
    cur = conn.execute(
        """
        INSERT INTO videos (path, filename, slug, status, failed_stage, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (f"/videos/{slug}.mp4", f"{slug}.mp4", slug, status, failed_stage, now, now),
    )
    return cur.lastrowid


def test_status_transitions_and_history(test_db: sqlite3.Connection) -> None:
    vid = seed_video(test_db, "video1", PENDING)

    # Transition to EXTRACTING
    set_status(test_db, vid, EXTRACTING, stage="extract")
    row = test_db.execute("SELECT status, attempts FROM videos WHERE id = ?", (vid,)).fetchone()
    assert row["status"] == EXTRACTING

    # Transition to FAILED
    set_status(test_db, vid, FAILED, stage="transcribe", failed_stage="transcribe", error="OOM")
    row = test_db.execute("SELECT status, failed_stage, error, attempts FROM videos WHERE id = ?", (vid,)).fetchone()
    assert row["status"] == FAILED
    assert row["error"] == "OOM"
    assert row["attempts"] == 1
    # Check stage history
    hist = test_db.execute("SELECT stage, status, error FROM stage_history WHERE video_id = ? ORDER BY id ASC", (vid,)).fetchall()
    assert len(hist) == 2
    assert hist[0]["stage"] == "extract"
    assert hist[1]["stage"] == "transcribe"
    assert hist[1]["error"] == "OOM"


def test_work_selection_queries(test_db: sqlite3.Connection) -> None:
    seed_video(test_db, "v_pending", PENDING)
    seed_video(test_db, "v_extracting", EXTRACTING)
    seed_video(test_db, "v_transcribing", TRANSCRIBING)
    seed_video(test_db, "v_failed_extract", FAILED, failed_stage="extract")
    seed_video(test_db, "v_failed_transcribe", FAILED, failed_stage="transcribe")
    seed_video(test_db, "v_failed_correct", FAILED, failed_stage="correct")
    seed_video(test_db, "v_transcribed", TRANSCRIBED)
    seed_video(test_db, "v_completed", COMPLETED)
    seed_video(test_db, "v_needs_review", NEEDS_REVIEW)

    # 1. Transcribe selection
    to_transcribe = [r["slug"] for r in get_videos_for_transcribe(test_db)]
    expected_transcribe = {"v_pending", "v_extracting", "v_transcribing", "v_failed_extract", "v_failed_transcribe"}
    assert set(to_transcribe) == expected_transcribe

    # 2. Correct selection
    to_correct = [r["slug"] for r in get_videos_for_correct(test_db)]
    expected_correct = {"v_transcribed", "v_failed_correct"}
    assert set(to_correct) == expected_correct

    # Status counts
    counts = get_status_counts(test_db)
    assert counts[PENDING] == 1
    assert counts[FAILED] == 3
    assert counts[COMPLETED] == 1
