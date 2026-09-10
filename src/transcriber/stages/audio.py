"""Audio loading and validation for 16kHz mono PCM WAV."""

import logging
import wave
from pathlib import Path

import numpy as np

from transcriber.stages import StageError

logger = logging.getLogger("transcriber.audio")


def load_wav_mono16k(path: Path | str) -> np.ndarray:
    """Load mono 16kHz 16-bit PCM WAV into a 1D float32 numpy array normalized to [-1.0, 1.0]."""
    p = Path(path).resolve()
    if not p.exists():
        raise StageError("audio", f"WAV file not found: {p}")

    try:
        with wave.open(str(p), "rb") as wf:
            channels = wf.getnchannels()
            framerate = wf.getframerate()
            sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()

            if channels != 1:
                raise StageError("audio", f"Expected mono audio, got {channels} channels in {p.name}")
            if framerate != 16000:
                raise StageError("audio", f"Expected 16000 Hz, got {framerate} Hz in {p.name}")
            if sampwidth != 2:
                raise StageError("audio", f"Expected 16-bit PCM (2 bytes), got {sampwidth} bytes in {p.name}")

            frames = wf.readframes(n_frames)
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            return audio
    except StageError:
        raise
    except Exception as e:
        logger.error("Failed to read WAV file %s: %s", p.name, e)
        raise StageError("audio", f"Failed to read WAV file {p.name}: {e}") from e
