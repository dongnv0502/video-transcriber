"""Stage: Export transcripts to SRT, TXT, and Markdown formats."""

import json
import logging
import sqlite3
from pathlib import Path

from transcriber.config import Settings
from transcriber.db import Status, set_status
from transcriber.models import CorrectedTranscript, RawTranscript
from transcriber.paths import atomic_write_text

logger = logging.getLogger("transcriber.export")


def format_srt_time(t: float) -> str:
    """Format seconds into SRT timestamp HH:MM:SS,mmm."""
    total_ms = max(0, int(round(t * 1000)))
    total_s, ms = divmod(total_ms, 1000)
    m, s = divmod(total_s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def format_hms(t: float) -> str:
    """Format seconds into HH:MM:SS."""
    total_s = max(0, int(round(t)))
    m, s = divmod(total_s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


COMMON_ACRONYMS = {
    "iam", "ec2", "s3", "vpc", "ebs", "efs", "alb", "nlb", "rds",
    "sns", "sqs", "kms", "sts", "cli", "sdk", "api", "aws", "ecs", "eks",
}


def slug_to_title(slug: str) -> str:
    """Convert slug into title, capitalizing only the first character of each token.

    Restores common AWS acronyms (e.g. IAM, EC2, S3, VPC) to full uppercase.
    """
    words = slug.replace("_", "-").split("-")
    capitalized = []
    for w in words:
        if not w:
            continue
        if w.lower() in COMMON_ACRONYMS:
            capitalized.append(w.upper())
        else:
            capitalized.append(w[0].upper() + w[1:])
    return " ".join(capitalized).strip()

def export_video(
    settings: Settings,
    row: sqlite3.Row,
    transcript: RawTranscript | CorrectedTranscript | None = None,
    force: bool = False,
    conn: sqlite3.Connection | None = None,
) -> Path:
    """Export transcripts to SRT, TXT, and MD in output/<slug>/."""
    slug = row["slug"]
    out_dir = settings.output_dir.resolve() / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    srt_path = out_dir / "transcript.srt"
    txt_path = out_dir / "transcript.txt"
    md_path = out_dir / "transcript.md"

    # Source transcript selection
    if transcript is None:
        corrected_json_path = out_dir / "corrected.json"
        raw_json_path = out_dir / "raw.json"

        if corrected_json_path.exists():
            with open(corrected_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            transcript = CorrectedTranscript.model_validate(data)
        elif raw_json_path.exists():
            with open(raw_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            transcript = RawTranscript.model_validate(data)
        else:
            raise FileNotFoundError(f"Neither corrected.json nor raw.json found in {out_dir}")

    # Check if files already exist and can be skipped
    source_json = out_dir / ("corrected.json" if isinstance(transcript, CorrectedTranscript) else "raw.json")
    if not force and srt_path.exists() and txt_path.exists() and md_path.exists() and source_json.exists():
        src_mtime = source_json.stat().st_mtime
        if all(p.stat().st_mtime >= src_mtime for p in (srt_path, txt_path, md_path)):
            logger.info("Exports are up to date for %s", slug)
            return out_dir

    # 1. Generate SRT
    srt_cues: list[str] = []
    for idx, seg in enumerate(transcript.segments, start=1):
        clean_text = " ".join(seg.text.splitlines()).strip()
        t_start = format_srt_time(seg.start)
        t_end = format_srt_time(seg.end)
        srt_cues.append(f"{idx}\n{t_start} --> {t_end}\n{clean_text}\n")
    srt_content = "\n".join(srt_cues)
    atomic_write_text(srt_path, srt_content)

    # 2. Generate TXT
    txt_content = "\n".join(" ".join(seg.text.splitlines()).strip() for seg in transcript.segments) + "\n"
    atomic_write_text(txt_path, txt_content)

    # 3. Generate Markdown
    needs_review_count = sum(1 for seg in transcript.segments if getattr(seg, "needs_review", False))
    duration_s = transcript.video.duration_seconds or (transcript.segments[-1].end if transcript.segments else 0.0)

    title = slug_to_title(slug)
    md_lines: list[str] = [
        f"# {title}",
        "",
        f"- source: `{transcript.video.filename}`",
        f"- duration: {format_hms(duration_s)}",
        f"- model: {transcript.transcription.model}",
        f"- language: {transcript.transcription.language}",
        f"- segments: {len(transcript.segments)}",
        f"- needs_review: {needs_review_count}",
        "",
        "## Transcript",
        "",
    ]

    for seg in transcript.segments:
        start_hms = format_hms(seg.start)
        end_hms = format_hms(seg.end)
        cue_header = f"[{start_hms} - {end_hms}]"
        if getattr(seg, "needs_review", False):
            cue_header += " <!-- needs_review -->"

        clean_text = " ".join(seg.text.splitlines()).strip()
        md_lines.append(cue_header)
        md_lines.append(clean_text)
        md_lines.append("")

    atomic_write_text(md_path, "\n".join(md_lines).rstrip() + "\n")

    # Update DB status if connection is available
    if conn is not None:
        final_status = Status.NEEDS_REVIEW if needs_review_count > 0 else Status.COMPLETED
        set_status(
            conn,
            row["id"],
            final_status,
            stage="export",
            needs_review_count=needs_review_count,
        )

    logger.debug("Successfully exported SRT, TXT, MD to %s", out_dir)
    return out_dir
