"""Path utilities, slugification, and atomic filesystem operations."""

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any


def fold(text: str) -> str:
    """Lowercase and strip diacritics via NFD for normalization and matching."""
    nfd = unicodedata.normalize("NFD", text)
    no_marks = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    return no_marks.lower().replace("đ", "d").replace("Đ", "d")


def slugify(stem: str) -> str:
    """Generate clean slug from a file stem.

    NFC-normalize, lowercase, replace any run of non [a-z0-9] with '-',
    strip leading/trailing '-', collapse repeats, truncate to 80 chars;
    empty -> 'video'.
    """
    s = unicodedata.normalize("NFC", stem).lower()
    # Normalize Vietnamese accented characters to latin equivalents for clean URL/slug strings
    s = fold(s)
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    if len(s) > 80:
        s = s[:80].rstrip("-")
    return s if s else "video"


def atomic_write_text(path: Path | str, text: str, encoding: str = "utf-8") -> None:
    """Write text to path atomically using a temporary file and os.replace."""
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + ".tmp")

    try:
        with open(tmp_path, "w", encoding=encoding) as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, target)

        # Fsync parent directory to ensure directory entry durability on POSIX
        dir_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def atomic_write_json(path: Path | str, obj: Any, indent: int = 2, encoding: str = "utf-8") -> None:
    """Write object as JSON to path atomically.

    Leaves no .tmp file and never creates a partial file if serialization fails.
    """
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + ".tmp")

    # Serialize to memory first to ensure no partial file on serialization error
    payload = json.dumps(obj, indent=indent, ensure_ascii=False)

    try:
        with open(tmp_path, "w", encoding=encoding) as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, target)

        dir_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
