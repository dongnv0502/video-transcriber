"""Stage: Scan video directory, probe duration via ffprobe, and persist to SQLite."""

import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from sqlite3 import Connection
from typing import Any

from transcriber.config import Settings
from transcriber.db import PENDING, utc_now
from transcriber.paths import slugify

logger = logging.getLogger("transcriber.scan")

VIDEO_EXTENSIONS = {".mp4", ".m4v", ".mov"}


@dataclass
class ScanReport:
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    total: int = 0
    items: list[dict[str, Any]] = field(default_factory=list)


def probe_duration(path: Path) -> float | None:
    """Probe audio/video duration in seconds via ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nk=1:nw=1",
        "--",
        str(path),
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        out = res.stdout.strip()
        if out:
            return float(out)
    except Exception as e:
        logger.warning("Failed to probe duration for %s: %s", path.name, e)
    return None


def get_unique_slug(conn: Connection, base_slug: str, exclude_path: str | None = None) -> str:
    """Ensure slug is unique in DB by appending -2, -3, ... if colliding with a different path."""
    slug = base_slug
    counter = 1
    while True:
        row = conn.execute("SELECT path FROM videos WHERE slug = ?", (slug,)).fetchone()
        if not row:
            return slug
        if exclude_path and row["path"] == exclude_path:
            return slug
        counter += 1
        slug = f"{base_slug}-{counter}"


def scan(conn: Connection, settings: Settings, rescan: bool = False) -> ScanReport:
    """Scan video directory for MP4/M4V/MOV files and update SQLite database."""
    video_dir = settings.video_dir.resolve()
    report = ScanReport()

    if not video_dir.exists():
        logger.warning("Video directory %s does not exist", video_dir)
        return report

    all_files = sorted(video_dir.rglob("*"))
    video_paths: list[Path] = []
    for p in all_files:
        if not p.is_file():
            continue
        # Skip dotfiles and AppleDouble files
        if p.name.startswith(".") or p.name.startswith("._"):
            continue
        if p.suffix.lower() in VIDEO_EXTENSIONS:
            video_paths.append(p)

    report.total = len(video_paths)
    now = utc_now()

    for p in video_paths:
        abs_path = p.resolve().as_posix()
        stat = p.stat()
        size_bytes = stat.st_size
        mtime = stat.st_mtime
        filename = p.name

        existing = conn.execute("SELECT * FROM videos WHERE path = ?", (abs_path,)).fetchone()

        if existing is None:
            # Generate unique slug
            base_slug = slugify(p.stem)
            slug = get_unique_slug(conn, base_slug)
            duration = probe_duration(p)

            with conn:
                conn.execute(
                    """
                    INSERT INTO videos (
                        path, filename, slug, size_bytes, mtime, duration_seconds,
                        status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        abs_path,
                        filename,
                        slug,
                        size_bytes,
                        mtime,
                        duration,
                        PENDING,
                        now,
                        now,
                    ),
                )
            report.added += 1
            report.items.append({"path": abs_path, "slug": slug, "action": "added"})
            logger.debug("Added new video: %s (slug=%s)", filename, slug)

        else:
            changed = existing["size_bytes"] != size_bytes or abs(existing["mtime"] - mtime) > 1e-4

            if changed:
                if rescan:
                    duration = probe_duration(p)
                    with conn:
                        # Clear cache and reset to PENDING
                        conn.execute("DELETE FROM llm_cache WHERE video_id = ?", (existing["id"],))
                        conn.execute(
                            """
                            UPDATE videos SET
                                size_bytes = ?, mtime = ?, duration_seconds = ?,
                                status = ?, failed_stage = NULL, error = NULL,
                                updated_at = ?
                            WHERE id = ?
                            """,
                            (size_bytes, mtime, duration, PENDING, now, existing["id"]),
                        )
                    report.updated += 1
                    report.items.append({"path": abs_path, "slug": existing["slug"], "action": "updated"})
                    logger.info("Rescanned and reset video: %s", filename)
                else:
                    report.unchanged += 1
                    report.items.append({"path": abs_path, "slug": existing["slug"], "action": "unchanged"})
                    logger.warning(
                        "Video file changed on disk but --rescan not set: %s", filename
                    )
            else:
                report.unchanged += 1
                report.items.append({"path": abs_path, "slug": existing["slug"], "action": "unchanged"})

    logger.info(
        "Scan completed: %d total, %d added, %d updated, %d unchanged",
        report.total,
        report.added,
        report.updated,
        report.unchanged,
    )
    return report
