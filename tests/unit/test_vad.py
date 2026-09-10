"""Unit tests for vad.py: speech chunk merging and window duration capping."""

from transcriber.stages.vad import merge_speech_chunks


def test_vad_gap_merge() -> None:
    # 16kHz audio: 16000 samples = 1.0s
    chunks = [
        {"start": 0, "end": 16000},       # 0.0 - 1.0s
        {"start": 24000, "end": 48000},   # 1.5 - 3.0s (gap = 0.5s < 1.0s merge_gap_s -> merge)
        {"start": 80000, "end": 96000},   # 5.0 - 6.0s (gap = 2.0s > 1.0s -> separate)
    ]
    windows = merge_speech_chunks(
        chunks,
        total_samples=160000,
        sample_rate=16000,
        merge_gap_s=1.0,
        max_window_s=10.0,
    )
    assert len(windows) == 2
    assert windows[0].start_s == 0.0 and windows[0].end_s == 3.0
    assert windows[1].start_s == 5.0 and windows[1].end_s == 6.0


def test_vad_max_window_cap() -> None:
    # Multiple chunks with small gaps that would exceed max_window_s (e.g. 4.0s)
    chunks = [
        {"start": 0, "end": 32000},      # 0.0 - 2.0s
        {"start": 35200, "end": 67200},  # 2.2 - 4.2s (total 4.2s > 4.0s -> separate)
    ]
    windows = merge_speech_chunks(
        chunks,
        total_samples=80000,
        sample_rate=16000,
        merge_gap_s=1.0,
        max_window_s=4.0,
    )
    assert len(windows) == 2
    assert windows[0].end_s == 2.0
    assert windows[1].start_s == 2.2


def test_vad_oversized_single_chunk() -> None:
    # Single chunk of 10s when max_window_s is 5s remains intact as its own window
    chunks = [{"start": 0, "end": 160000}]  # 10s
    windows = merge_speech_chunks(
        chunks,
        total_samples=160000,
        sample_rate=16000,
        max_window_s=5.0,
    )
    assert len(windows) == 1
    assert windows[0].start_s == 0.0
    assert windows[0].end_s == 10.0


def test_vad_empty_input_fallback() -> None:
    windows = merge_speech_chunks(
        [],
        total_samples=48000,
        sample_rate=16000,
    )
    assert len(windows) == 1
    assert windows[0].start_s == 0.0
    assert windows[0].end_s == 3.0
