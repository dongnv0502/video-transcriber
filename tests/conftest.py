"""Pytest test configuration and shared fixtures."""

import os
import sqlite3
from pathlib import Path

import pytest

from transcriber.config import Settings
from transcriber.db import init_db


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
    """Create isolated directory structure for tests."""
    (tmp_path / "videos").mkdir()
    (tmp_path / "output").mkdir()
    (tmp_path / "work").mkdir()
    (tmp_path / "logs").mkdir()
    return tmp_path


@pytest.fixture
def test_settings(tmp_workspace: Path) -> Settings:
    """Return Settings pointing to isolated temporary directories."""
    return Settings(
        video_dir=tmp_workspace / "videos",
        output_dir=tmp_workspace / "output",
        work_dir=tmp_workspace / "work",
        db_path=tmp_workspace / "state.db",
        log_dir=tmp_workspace / "logs",
        llm_enabled=False,
    )


@pytest.fixture
def test_db(test_settings: Settings) -> sqlite3.Connection:
    """Initialize and return a SQLite connection in isolated workspace."""
    conn = init_db(test_settings.db_path)
    yield conn
    conn.close()
