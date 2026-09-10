"""Stage: Local Whisper ASR transcription and raw transcript generation."""

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from transcriber.config import Settings
from transcriber.db import Status, set_status, utc_now
from transcriber.models import RawSegment, RawTranscript, TranscriptionMeta, VideoMeta
from transcriber.paths import atomic_write_json
from transcriber.stages import StageError
from transcriber.stages.audio import load_wav_mono16k
from transcriber.stages.extract import extract_audio
from transcriber.stages.vad import speech_windows

logger = logging.getLogger("transcriber.transcribe")


class Backend(Protocol):
    """Protocol for Whisper transcription backends."""

    name: str

    def transcribe_window(self, audio: np.ndarray, settings: Settings) -> list[dict[str, Any]]:
        ...


class MlxBackend:
    """Apple Silicon MLX Whisper backend."""

    name: str = "mlx"

    def __init__(self, settings: Settings):
        self.model_name = settings.model_name
        self._check_available()

    def _check_available(self) -> None:
        try:
            import mlx_whisper  # noqa: F401
        except ImportError as e:
            raise StageError("transcribe", "mlx-whisper is not installed or available") from e

    def transcribe_window(self, audio: np.ndarray, settings: Settings) -> list[dict[str, Any]]:
        import mlx_whisper

        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.model_name,
            language=settings.language,
            initial_prompt=settings.initial_prompt,
            word_timestamps=settings.word_timestamps,
            condition_on_previous_text=False,
            verbose=None,
        )
        return result.get("segments", [])


class FasterWhisperBackend:
    """CPU/int8 faster-whisper backend for comparative benchmarking."""

    name: str = "faster-whisper"

    def __init__(self, settings: Settings):
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise StageError("transcribe", "faster-whisper is not available") from e

        model_name = settings.model_name_fw or "large-v3"
        self.model = WhisperModel(model_name, device="cpu", compute_type="int8")

    def transcribe_window(self, audio: np.ndarray, settings: Settings) -> list[dict[str, Any]]:
        segments, _ = self.model.transcribe(
            audio,
            language=settings.language,
            initial_prompt=settings.initial_prompt,
            vad_filter=False,
            word_timestamps=settings.word_timestamps,
            condition_on_previous_text=False,
        )
        out: list[dict[str, Any]] = []
        for s in segments:
            item: dict[str, Any] = {
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "avg_logprob": s.avg_logprob,
                "no_speech_prob": s.no_speech_prob,
                "compression_ratio": s.compression_ratio,
                "temperature": s.temperature,
            }
            if settings.word_timestamps and s.words:
                item["words"] = [
                    {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
                    for w in s.words
                ]
            out.append(item)
        return out


def get_backend(name: str, settings: Settings) -> Backend:
    """Instantiate requested transcription backend."""
    if name == "mlx":
        return MlxBackend(settings)
    elif name == "faster-whisper":
        return FasterWhisperBackend(settings)
    else:
        raise ValueError(f"Unknown backend: {name}. Supported: 'mlx', 'faster-whisper'")


def transcribe_video(
    conn: sqlite3.Connection,
    settings: Settings,
    row: sqlite3.Row,
    force: bool = False,
    on_progress: Callable[[float, int], None] | None = None,
) -> RawTranscript:
    """Execute audio extraction, VAD, and transcription for a video."""
    video_id = row["id"]
    slug = row["slug"]
    video_path = Path(row["path"])

    out_video_dir = settings.output_dir.resolve() / slug
    out_video_dir.mkdir(parents=True, exist_ok=True)
    raw_json_path = out_video_dir / "raw.json"

    # Check if raw.json exists and is valid (skip if not force)
    if raw_json_path.exists() and not force:
        try:
            with open(raw_json_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            parsed = RawTranscript.model_validate(raw_data)
            logger.info("Reusing existing valid raw.json for %s", slug)
            set_status(
                conn,
                video_id,
                Status.TRANSCRIBED,
                segment_count=len(parsed.segments),
                model=parsed.transcription.model,
                backend=parsed.transcription.backend,
            )
            return parsed
        except Exception as e:
            logger.warning("Existing raw.json for %s is invalid (%s); re-transcribing", slug, e)

    # Stage 1: Audio Extraction
    set_status(conn, video_id, Status.EXTRACTING, stage="extract")
    work_audio_dir = settings.work_dir.resolve() / "audio"
    work_audio_dir.mkdir(parents=True, exist_ok=True)
    wav_path = work_audio_dir / f"{slug}.wav"

    try:
        extract_audio(video_path, wav_path, force=force)
    except Exception as e:
        set_status(conn, video_id, Status.FAILED, failed_stage="extract", error=str(e), stage="extract")
        raise

    # Stage 2: Transcription
    t0 = time.perf_counter()
    set_status(conn, video_id, Status.TRANSCRIBING, stage="transcribe")

    try:
        audio = load_wav_mono16k(wav_path)
        windows = speech_windows(audio, settings)
        total_window_s = sum(w.end_s - w.start_s for w in windows)
        processed_window_s = 0.0

        backend = get_backend(settings.model_backend, settings)
        segments: list[RawSegment] = []
        seg_id = 0

        for w_idx, win in enumerate(windows):
            win_audio = audio[win.start_sample:win.end_sample]
            if win_audio.size == 0:
                continue

            raw_segs = backend.transcribe_window(win_audio, settings)

            for item in raw_segs:
                seg_text = str(item.get("text", "")).strip()
                if not seg_text:
                    continue

                seg_start = win.start_s + float(item.get("start", 0.0))
                seg_end = min(win.start_s + float(item.get("end", 0.0)), win.end_s)
                if seg_end <= seg_start:
                    seg_end = seg_start + 0.1

                confidence: float | None = None
                if settings.word_timestamps and "words" in item and item["words"]:
                    probs = [float(w["probability"]) for w in item["words"] if "probability" in w]
                    if probs:
                        confidence = float(np.mean(probs))

                seg = RawSegment(
                    id=seg_id,
                    start=round(seg_start, 3),
                    end=round(seg_end, 3),
                    text=seg_text,
                    avg_logprob=item.get("avg_logprob"),
                    no_speech_prob=item.get("no_speech_prob"),
                    compression_ratio=item.get("compression_ratio"),
                    temperature=item.get("temperature"),
                    confidence=confidence,
                    window_index=w_idx,
                )
                segments.append(seg)
                seg_id += 1

            processed_window_s += (win.end_s - win.start_s)
            if on_progress:
                progress_pct = processed_window_s / max(total_window_s, 1e-4)
                on_progress(min(progress_pct, 1.0), len(segments))

        t_elapsed = time.perf_counter() - t0
        duration = row["duration_seconds"] or (segments[-1].end if segments else 0.0)
        rtf = round(t_elapsed / max(duration, 1e-4), 4)

        video_meta = VideoMeta(
            filename=row["filename"],
            relative_path=row["path"],
            duration_seconds=duration,
            size_bytes=row["size_bytes"],
        )
        transcription_meta = TranscriptionMeta(
            model=settings.model_name,
            backend=settings.model_backend,
            language=settings.language,
            word_timestamps=settings.word_timestamps,
            vad={
                "enabled": settings.vad_enabled,
                "threshold": settings.vad_threshold,
                "min_silence_ms": settings.vad_min_silence_ms,
                "speech_pad_ms": settings.vad_speech_pad_ms,
            },
            created_at=utc_now(),
            tool_version="0.1.0",
            processing_seconds=round(t_elapsed, 2),
            real_time_factor=rtf,
        )

        raw_transcript = RawTranscript(
            video=video_meta,
            transcription=transcription_meta,
            segments=segments,
        )

        # Write raw.json atomically
        atomic_write_json(raw_json_path, raw_transcript.model_dump())

        # Cleanup WAV if keep_audio is false
        if not settings.keep_audio and wav_path.exists():
            try:
                wav_path.unlink()
            except OSError:
                pass

        set_status(
            conn,
            video_id,
            Status.TRANSCRIBED,
            stage="transcribe",
            segment_count=len(segments),
            model=settings.model_name,
            backend=settings.model_backend,
            duration_seconds=duration,
        )

        return raw_transcript

    except Exception as e:
        logger.error("Transcription failed for %s: %s", slug, e, exc_info=True)
        set_status(conn, video_id, Status.FAILED, failed_stage="transcribe", error=str(e), stage="transcribe")
        raise
