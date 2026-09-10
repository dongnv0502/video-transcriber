"""Voice Activity Detection (VAD) and speech window chunking."""

import logging
from typing import NamedTuple

import numpy as np

from transcriber.config import Settings

logger = logging.getLogger("transcriber.vad")


class Window(NamedTuple):
    start_s: float
    end_s: float
    start_sample: int
    end_sample: int


def merge_speech_chunks(
    chunks: list[dict[str, int]],
    total_samples: int,
    sample_rate: int = 16000,
    merge_gap_s: float = 1.0,
    max_window_s: float = 300.0,
    min_window_s: float = 0.30,
) -> list[Window]:
    """Merge consecutive speech chunks based on gap and maximum window duration.

    Chunks longer than max_window_s remain intact as single windows.
    Windows shorter than min_window_s are dropped.
    If the merged list is empty, returns a single whole-file window.
    """
    if not chunks or total_samples <= 0:
        total_s = total_samples / max(sample_rate, 1)
        return [Window(0.0, total_s, 0, total_samples)]

    merged: list[tuple[int, int]] = []
    curr_start = chunks[0]["start"]
    curr_end = chunks[0]["end"]

    for next_chunk in chunks[1:]:
        n_start = next_chunk["start"]
        n_end = next_chunk["end"]

        gap_s = (n_start - curr_end) / sample_rate
        merged_duration_s = (n_end - curr_start) / sample_rate

        if gap_s < merge_gap_s and merged_duration_s <= max_window_s:
            curr_end = n_end
        else:
            merged.append((curr_start, curr_end))
            curr_start = n_start
            curr_end = n_end

    merged.append((curr_start, curr_end))

    # Convert to Windows and filter out short windows
    windows: list[Window] = []
    for s_sample, e_sample in merged:
        # Clamp samples to valid range
        s_sample = max(0, min(s_sample, total_samples))
        e_sample = max(s_sample, min(e_sample, total_samples))
        start_s = s_sample / sample_rate
        end_s = e_sample / sample_rate

        if (end_s - start_s) >= min_window_s:
            windows.append(Window(start_s, end_s, s_sample, e_sample))

    if not windows:
        logger.warning("All VAD speech windows were shorter than %s s; falling back to whole-file window", min_window_s)
        total_s = total_samples / sample_rate
        return [Window(0.0, total_s, 0, total_samples)]

    return windows


def speech_windows(audio: np.ndarray, settings: Settings, sample_rate: int = 16000) -> list[Window]:
    """Segment audio into contiguous speech windows using Silero VAD from faster-whisper.

    Contiguous slices of original audio are preserved so timestamps only need + window.start_s.
    """
    total_samples = len(audio)
    if not settings.vad_enabled or total_samples == 0:
        total_s = total_samples / max(sample_rate, 1)
        return [Window(0.0, total_s, 0, total_samples)]

    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        opts = VadOptions(
            threshold=settings.vad_threshold,
            min_speech_duration_ms=settings.vad_min_speech_ms,
            min_silence_duration_ms=settings.vad_min_silence_ms,
            speech_pad_ms=settings.vad_speech_pad_ms,
        )

        raw_chunks = get_speech_timestamps(
            audio,
            vad_options=opts,
            sampling_rate=sample_rate,
        )

        return merge_speech_chunks(
            raw_chunks,
            total_samples=total_samples,
            sample_rate=sample_rate,
            merge_gap_s=settings.vad_merge_gap_s,
            max_window_s=settings.vad_max_window_s,
            min_window_s=0.30,
        )

    except Exception as e:
        logger.warning("VAD detection failed (%s); falling back to whole-file window", e)
        total_s = total_samples / max(sample_rate, 1)
        return [Window(0.0, total_s, 0, total_samples)]
