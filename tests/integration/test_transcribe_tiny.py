"""Integration test running Whisper on a short audio clip."""

import subprocess
from pathlib import Path

import pytest

from transcriber.config import Settings
from transcriber.stages.audio import load_wav_mono16k
from transcriber.stages.transcribe import MlxBackend


@pytest.mark.integration
@pytest.mark.requires_model
def test_transcribe_tiny_model(tmp_path: Path) -> None:
    # Synthesize a short 3-second speech-like wav
    wav_path = tmp_path / "test_tone.wav"
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=300:duration=2",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    settings = Settings(
        model_name="mlx-community/whisper-tiny",
        language="vi",
    )

    audio = load_wav_mono16k(wav_path)
    backend = MlxBackend(settings)
    segments = backend.transcribe_window(audio, settings)

    assert isinstance(segments, list)
