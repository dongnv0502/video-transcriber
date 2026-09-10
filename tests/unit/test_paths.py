"""Unit tests for paths.py: slugify and atomic file writes."""

import json
from pathlib import Path

import pytest

from transcriber.paths import atomic_write_json, atomic_write_text, fold, slugify


def test_slugify_normalization() -> None:
    assert slugify("001_Introduction! to AWS") == "001-introduction-to-aws"
    assert slugify("   ---hello---world---   ") == "hello-world"
    assert slugify("Bài Giảng AWS: EC2 & S3") == "bai-giang-aws-ec2-s3"
    assert slugify("---___---") == "video"
    assert slugify("") == "video"


def test_slugify_truncation() -> None:
    very_long = "a" * 100
    slug = slugify(very_long)
    assert len(slug) == 80
    assert slug == "a" * 80


def test_fold_diacritics() -> None:
    assert fold("Bài Giảng Tiếng Việt") == "bai giang tieng viet"
    assert fold("Đường Dẫn") == "duong dan"


def test_atomic_write_text(tmp_path: Path) -> None:
    target = tmp_path / "test.txt"
    atomic_write_text(target, "hello world")
    assert target.exists()
    assert target.read_text(encoding="utf-8") == "hello world"

    # Overwrite
    atomic_write_text(target, "second write")
    assert target.read_text(encoding="utf-8") == "second write"


def test_atomic_write_json_success(tmp_path: Path) -> None:
    target = tmp_path / "test.json"
    data = {"key": "value", "numbers": [1, 2, 3]}
    atomic_write_json(target, data)
    assert target.exists()
    with open(target, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == data


def test_atomic_write_json_cleanup_on_error(tmp_path: Path) -> None:
    target = tmp_path / "failed.json"
    tmp_file = target.with_suffix(".json.tmp")

    # Pass non-serializable object
    with pytest.raises(TypeError):
        atomic_write_json(target, {"invalid": object()})

    # Assert neither target nor tmp exists
    assert not target.exists()
    assert not tmp_file.exists()
