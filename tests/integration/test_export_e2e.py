"""Integration test for export stage byte-level parity against fixtures."""

import json
from pathlib import Path

import pytest

from transcriber.config import Settings
from transcriber.models import RawTranscript
from transcriber.stages.export import export_video


@pytest.mark.integration
def test_export_parity_with_fixtures(tmp_path: Path) -> None:
    fixture_dir = Path("tests/fixtures")
    raw_fixture = fixture_dir / "raw_sample.json"
    with open(raw_fixture, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw_transcript = RawTranscript.model_validate(data)

    settings = Settings(output_dir=tmp_path / "output")
    row = {"slug": "001-introduction", "id": 1}

    out_dir = export_video(settings, row, transcript=raw_transcript, force=True)
    assert out_dir.exists()

    srt_out = (out_dir / "transcript.srt").read_text(encoding="utf-8")
    txt_out = (out_dir / "transcript.txt").read_text(encoding="utf-8")
    md_out = (out_dir / "transcript.md").read_text(encoding="utf-8")

    expected_srt = (fixture_dir / "expected_transcript.srt").read_text(encoding="utf-8")
    expected_txt = (fixture_dir / "expected_transcript.txt").read_text(encoding="utf-8")
    expected_md = (fixture_dir / "expected_transcript.md").read_text(encoding="utf-8")

    assert srt_out.strip() == expected_srt.strip()
    assert txt_out.strip() == expected_txt.strip()
    assert md_out.strip() == expected_md.strip()
