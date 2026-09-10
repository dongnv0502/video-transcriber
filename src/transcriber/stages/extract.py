"""Stage: Extract 16kHz mono PCM WAV from video via FFmpeg."""

import logging
import os
import subprocess
from pathlib import Path

from transcriber.stages import StageError

logger = logging.getLogger("transcriber.extract")


def extract_audio(video_path: Path | str, out_wav: Path | str, force: bool = False) -> Path:
    """Extract mono 16kHz 16-bit PCM WAV from video file using ffmpeg.

    Skips extraction if out_wav exists and is non-empty, unless force=True.
    """
    video = Path(video_path).resolve()
    target = Path(out_wav).resolve()

    if not video.exists():
        raise StageError("extract", f"Video file not found: {video}")

    if target.exists() and target.stat().st_size > 0 and not force:
        logger.debug("Reusing existing extracted audio: %s", target)
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + ".tmp")

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-i",
        str(video),
        "-vn",
        "-sn",
        "-dn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        str(tmp_path),
    ]

    try:
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        err_msg = e.stderr.decode("utf-8", errors="replace")[-2000:]
        logger.error("FFmpeg extraction failed for %s: %s", video.name, err_msg)
        raise StageError("extract", f"FFmpeg failed: {err_msg}") from e

    os.replace(tmp_path, target)

    # Sync parent directory
    dir_fd = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)

    logger.debug("Successfully extracted audio to %s", target)
    return target
