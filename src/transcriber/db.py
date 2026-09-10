"""SQLite database schema, status transitions, and queries."""

import sqlite3
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any


class Status(StrEnum):
    PENDING = "PENDING"
    EXTRACTING = "EXTRACTING"
    TRANSCRIBING = "TRANSCRIBING"
    TRANSCRIBED = "TRANSCRIBED"
    CORRECTING = "CORRECTING"
    COMPLETED = "COMPLETED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    FAILED = "FAILED"


# String constants matching enum values
PENDING = Status.PENDING.value
EXTRACTING = Status.EXTRACTING.value
TRANSCRIBING = Status.TRANSCRIBING.value
TRANSCRIBED = Status.TRANSCRIBED.value
CORRECTING = Status.CORRECTING.value
COMPLETED = Status.COMPLETED.value
NEEDS_REVIEW = Status.NEEDS_REVIEW.value
FAILED = Status.FAILED.value

SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = FULL;
PRAGMA foreign_keys = ON;
PRAGMA user_version = 1;

CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    filename TEXT NOT NULL,
    slug TEXT NOT NULL UNIQUE,
    size_bytes INTEGER,
    mtime REAL,
    duration_seconds REAL,
    status TEXT NOT NULL,
    failed_stage TEXT,
    model TEXT,
    backend TEXT,
    segment_count INTEGER,
    suspicious_count INTEGER,
    needs_review_count INTEGER,
    corrected_count INTEGER,
    llm_tokens_in INTEGER NOT NULL DEFAULT 0,
    llm_tokens_out INTEGER NOT NULL DEFAULT 0,
    attempts INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    started_at TEXT,
    completed_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS stage_history (
    id INTEGER PRIMARY KEY,
    video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    status TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    error TEXT
);

CREATE TABLE IF NOT EXISTS llm_cache (
    video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    segment_id INTEGER NOT NULL,
    request_hash TEXT NOT NULL,
    response_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (video_id, segment_id, request_hash)
);

CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(status);
"""


def utc_now() -> str:
    """Return ISO-formatted UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def get_connection(db_path: Path | str) -> sqlite3.Connection:
    """Open SQLite connection with row factory, foreign keys, and WAL mode."""
    path = Path(db_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = FULL;")
    return conn


def init_db(db_path: Path | str) -> sqlite3.Connection:
    """Initialize database schema idempotently."""
    conn = get_connection(db_path)
    with conn:
        conn.executescript(SCHEMA_SQL)
    return conn


def set_status(
    conn: sqlite3.Connection,
    video_id: int,
    status: str | Status,
    stage: str | None = None,
    failed_stage: str | None = None,
    error: str | None = None,
    attempt: int | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
    **fields: Any,
) -> None:
    """Update video status, record stage history, and commit."""
    status_str = status.value if isinstance(status, Status) else status
    now = utc_now()

    # Determine attempt counter
    if attempt is None:
        row = conn.execute("SELECT attempts FROM videos WHERE id = ?", (video_id,)).fetchone()
        current_attempts = row["attempts"] if row else 0
        if status_str == FAILED:
            attempt = current_attempts + 1
            fields["attempts"] = attempt
        else:
            attempt = max(current_attempts, 1)

    fields["status"] = status_str
    fields["updated_at"] = now
    if failed_stage is not None:
        fields["failed_stage"] = failed_stage
    if error is not None:
        fields["error"] = error
    if status_str in (COMPLETED, NEEDS_REVIEW):
        fields["completed_at"] = now

    set_clauses = [f"{k} = ?" for k in fields.keys()]
    values = list(fields.values()) + [video_id]

    with conn:
        conn.execute(
            f"UPDATE videos SET {', '.join(set_clauses)} WHERE id = ?",
            values,
        )
        if stage:
            conn.execute(
                """
                INSERT INTO stage_history (video_id, stage, status, attempt, started_at, finished_at, error)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    video_id,
                    stage,
                    status_str,
                    attempt,
                    started_at or now,
                    finished_at or (now if status_str in (COMPLETED, NEEDS_REVIEW, FAILED, TRANSCRIBED) else None),
                    error,
                ),
            )


def get_videos_for_transcribe(conn: sqlite3.Connection, force: bool = False, limit: int | None = None) -> list[sqlite3.Row]:
    """Select videos pending audio extraction or transcription."""
    query = """
        SELECT * FROM videos
        WHERE (
            status IN ('PENDING', 'EXTRACTING', 'TRANSCRIBING')
            OR (status = 'FAILED' AND failed_stage IN ('extract', 'transcribe'))
            OR (? = 1)
        )
        ORDER BY path ASC
    """
    params: list[Any] = [1 if force else 0]
    if limit is not None and limit > 0:
        query += " LIMIT ?"
        params.append(limit)
    return conn.execute(query, params).fetchall()


def get_videos_for_correct(conn: sqlite3.Connection, force: bool = False, limit: int | None = None) -> list[sqlite3.Row]:
    """Select videos ready for correction."""
    query = """
        SELECT * FROM videos
        WHERE (
            status = 'TRANSCRIBED'
            OR (status = 'FAILED' AND failed_stage = 'correct')
            OR (? = 1 AND status IN ('TRANSCRIBED', 'COMPLETED', 'NEEDS_REVIEW', 'FAILED'))
        )
        ORDER BY path ASC
    """
    params: list[Any] = [1 if force else 0]
    if limit is not None and limit > 0:
        query += " LIMIT ?"
        params.append(limit)
    return conn.execute(query, params).fetchall()


def get_status_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Return count of videos grouped by status."""
    rows = conn.execute("SELECT status, COUNT(*) as count FROM videos GROUP BY status").fetchall()
    return {row["status"]: row["count"] for row in rows}
