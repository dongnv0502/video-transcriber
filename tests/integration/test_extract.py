"""Integration test for audio extraction from video using ffmpeg."""

import subprocess
import wave
from pathlib import Path

import pytest

from transcriber.stages.extract import extract_audio


@pytest.fixture(scope="module")
def sample_mp4(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate a valid 3-second test MP4 file with audio using ffmpeg lavfi."""
    tmp_dir = tmp_path_factory.mktemp("mp4_media")
    mp4_path = tmp_dir / "test_synth.mp4"
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=3",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=3:size=320x240:rate=10",
        "-c:a",
        "aac",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-shortest",
        str(mp4_path),
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return mp4_path


@pytest.mark.integration
def test_extract_audio_format(sample_mp4: Path, tmp_path: Path) -> None:
    out_wav = tmp_path / "extracted.wav"
    result_path = extract_audio(sample_mp4, out_wav)

    assert result_path.exists()
    assert result_path == out_wav

    # Validate audio format via stdlib wave
    with wave.open(str(out_wav), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 16000
        assert wf.getsampwidth() == 2
        duration = wf.getnframes() / wf.getframerate()
        assert 2.8 <= duration <= 3.2
