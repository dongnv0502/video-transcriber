"""Quality and performance benchmark harness for ASR backends and models."""

import json
import logging
import resource
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from rapidfuzz.distance import Levenshtein

from transcriber.config import Settings
from transcriber.models import RawSegment, RawTranscript
from transcriber.paths import atomic_write_json, atomic_write_text, slugify
from transcriber.stages.audio import load_wav_mono16k
from transcriber.stages.extract import extract_audio
from transcriber.stages.rules import load_corrections, load_glossary
from transcriber.stages.scan import probe_duration
from transcriber.stages.suspicious import flag_segment
from transcriber.stages.transcribe import get_backend
from transcriber.stages.vad import speech_windows

logger = logging.getLogger("transcriber.benchmark")


@dataclass
class BenchmarkRunResult:
    video: str
    slug: str
    backend: str
    model: str
    duration_seconds: float
    processing_seconds: float
    real_time_factor: float
    peak_rss_mb: float
    segment_count: int
    mean_avg_logprob: float | None
    flags_breakdown: dict[str, int]
    term_recall: float | None = None
    must_not_contain_hits: int | None = None
    mean_sample_wer: float | None = None
    correction_stats: dict[str, Any] | None = None


def get_peak_rss_mb() -> float:
    """Get current peak RSS in megabytes."""
    # On macOS, ru_maxrss is in bytes
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    return rusage.ru_maxrss / (1024.0 * 1024.0)


def load_ground_truth(slug: str, base_dir: Path | str = "benchmark/ground_truth") -> dict[str, Any] | None:
    """Load ground truth YAML for video slug if present."""
    p = Path(base_dir) / f"{slug}.yml"
    if not p.exists():
        p = Path(base_dir) / f"{slug}.yaml"
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning("Failed reading ground truth %s: %s", p, e)
    return None


def calculate_ground_truth_metrics(
    transcript_text: str,
    segments: list[RawSegment],
    gt: dict[str, Any],
) -> tuple[float | None, int | None, float | None]:
    """Calculate term recall, false positives, and sample segment WER."""
    term_recall: float | None = None
    must_not_hits: int | None = None
    sample_wer: float | None = None

    # 1. Must contain recall
    must_contain = gt.get("must_contain", [])
    if must_contain:
        hits = sum(1 for term in must_contain if term in transcript_text)
        term_recall = round((hits / len(must_contain)) * 100.0, 2)

    # 2. Must not contain violations
    must_not = gt.get("must_not_contain", [])
    if must_not:
        must_not_hits = sum(1 for term in must_not if term in transcript_text)

    # 3. Sample segments WER
    sample_segments = gt.get("sample_segments", [])
    if sample_segments and segments:
        wers: list[float] = []
        for sample in sample_segments:
            s_start = float(sample.get("start", 0.0))
            s_text = str(sample.get("text", "")).strip()
            if not s_text:
                continue

            # Find nearest segment by start timestamp
            nearest = min(segments, key=lambda s: abs(s.start - s_start))
            ref_tokens = s_text.split()
            hyp_tokens = nearest.text.split()

            dist = Levenshtein.distance(ref_tokens, hyp_tokens)
            wer = dist / max(len(ref_tokens), 1)
            wers.append(wer)

        if wers:
            sample_wer = round(float(np.mean(wers)) * 100.0, 2)

    return term_recall, must_not_hits, sample_wer


def run_benchmark(
    video_paths: list[Path],
    backends: list[str],
    models: list[str],
    out_dir: Path | str,
    settings: Settings,
    with_correction: bool = False,
) -> tuple[list[BenchmarkRunResult], Path]:
    """Run benchmark grid across videos x backends x models."""
    out_root = Path(out_dir).resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load glossary & corrections for flag detection
    glossary_terms: list[str] = []
    g_path = Path("glossary/aws.yml")
    if g_path.exists():
        glossary_terms = load_glossary(g_path).terms

    correction_rules: list[Any] = []
    c_path = Path("glossary/corrections.yml")
    if c_path.exists():
        correction_rules = load_corrections(c_path).corrections

    results: list[BenchmarkRunResult] = []

    for v_path in video_paths:
        v_path = v_path.resolve()
        if not v_path.exists():
            logger.warning("Video file not found for benchmark: %s", v_path)
            continue

        slug = slugify(v_path.stem)
        duration = probe_duration(v_path) or 0.0

        # Extract audio to temp directory once per video
        temp_wav = run_dir / "audio" / f"{slug}.wav"
        temp_wav.parent.mkdir(parents=True, exist_ok=True)
        extract_audio(v_path, temp_wav)
        audio = load_wav_mono16k(temp_wav)
        windows = speech_windows(audio, settings)

        gt = load_ground_truth(slug)

        for backend_name in backends:
            for model_name in models:
                logger.info(
                    "Benchmarking: video=%s, backend=%s, model=%s",
                    slug,
                    backend_name,
                    model_name,
                )

                # Clone settings for this run
                bench_settings = settings.model_copy()
                bench_settings.model_backend = backend_name
                bench_settings.model_name = model_name

                rss_before = get_peak_rss_mb()
                t0 = time.perf_counter()

                backend = get_backend(backend_name, bench_settings)
                segments: list[RawSegment] = []
                seg_id = 0

                for w_idx, win in enumerate(windows):
                    win_audio = audio[win.start_sample:win.end_sample]
                    if win_audio.size == 0:
                        continue

                    raw_segs = backend.transcribe_window(win_audio, bench_settings)
                    for item in raw_segs:
                        seg_text = str(item.get("text", "")).strip()
                        if not seg_text:
                            continue
                        s_start = win.start_s + float(item.get("start", 0.0))
                        s_end = min(win.start_s + float(item.get("end", 0.0)), win.end_s)
                        segments.append(
                            RawSegment(
                                id=seg_id,
                                start=round(s_start, 3),
                                end=round(s_end, 3),
                                text=seg_text,
                                avg_logprob=item.get("avg_logprob"),
                                no_speech_prob=item.get("no_speech_prob"),
                                compression_ratio=item.get("compression_ratio"),
                                temperature=item.get("temperature"),
                                window_index=w_idx,
                            )
                        )
                        seg_id += 1

                t_elapsed = time.perf_counter() - t0
                rss_after = get_peak_rss_mb()

                effective_dur = duration if duration > 0 else (segments[-1].end if segments else 0.0)
                rtf = round(t_elapsed / max(effective_dur, 1e-4), 4)

                # Acoustic / metric calculations
                logprobs = [s.avg_logprob for s in segments if s.avg_logprob is not None]
                mean_logprob = float(np.mean(logprobs)) if logprobs else None

                # Flags breakdown
                flags_count: dict[str, int] = {}
                n_segs = len(segments)
                for i, s in enumerate(segments):
                    p_text = segments[i - 1].text if i > 0 else ""
                    n_text = segments[i + 1].text if i < n_segs - 1 else ""
                    flags = flag_segment(s, p_text, n_text, glossary_terms, correction_rules, bench_settings)
                    for f in flags:
                        flags_count[f] = flags_count.get(f, 0) + 1

                full_text = " ".join(s.text for s in segments)
                term_recall, must_not_hits, sample_wer = (None, None, None)
                if gt:
                    term_recall, must_not_hits, sample_wer = calculate_ground_truth_metrics(
                        full_text,
                        segments,
                        gt,
                    )

                result = BenchmarkRunResult(
                    video=v_path.name,
                    slug=slug,
                    backend=backend_name,
                    model=model_name,
                    duration_seconds=round(effective_dur, 2),
                    processing_seconds=round(t_elapsed, 2),
                    real_time_factor=rtf,
                    peak_rss_mb=round(rss_after, 1),
                    segment_count=len(segments),
                    mean_avg_logprob=round(mean_logprob, 3) if mean_logprob is not None else None,
                    flags_breakdown=flags_count,
                    term_recall=term_recall,
                    must_not_contain_hits=must_not_hits,
                    mean_sample_wer=sample_wer,
                )
                results.append(result)

    # Save results.json
    results_json_path = run_dir / "results.json"
    atomic_write_json(
        results_json_path,
        {"timestamp": timestamp, "runs": [asdict(r) for r in results]},
    )

    # Generate report.md
    report_md_path = run_dir / "report.md"
    generate_markdown_report(results, report_md_path, timestamp)

    return results, report_md_path


def generate_markdown_report(
    results: list[BenchmarkRunResult],
    report_path: Path,
    timestamp: str,
) -> None:
    """Generate markdown benchmark report per specification §17."""
    lines: list[str] = [
        f"# Benchmark Report — {timestamp}",
        "",
        "- Hardware: Apple Silicon (Metal acceleration)",
        "- GPU/Metal utilization: not measured",
        "",
        "## 1. Performance & Throughput",
        "",
        "| Video | Backend | Model | Duration | Processing | RTF | Peak RSS | Segments |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for r in results:
        lines.append(
            f"| `{r.slug}` | `{r.backend}` | `{r.model}` | "
            f"{r.duration_seconds:.1f}s | {r.processing_seconds:.1f}s | "
            f"**{r.real_time_factor:.3f}** | {r.peak_rss_mb:.1f} MB | {r.segment_count} |"
        )

    lines.extend([
        "",
        "## 2. Acoustic Quality & Suspicious Flags",
        "",
        "| Video | Backend | Model | Mean Logprob | Flags Total | Top Flags |",
        "|---|---|---|---|---|---|",
    ])

    for r in results:
        flags_total = sum(r.flags_breakdown.values())
        top_flags = ", ".join(f"{k}:{v}" for k, v in sorted(r.flags_breakdown.items(), key=lambda x: -x[1])[:3])
        logprob_str = f"{r.mean_avg_logprob:.3f}" if r.mean_avg_logprob is not None else "N/A"
        lines.append(
            f"| `{r.slug}` | `{r.backend}` | `{r.model}` | "
            f"{logprob_str} | {flags_total} | {top_flags or 'none'} |"
        )

    # Check if ground truth metrics are present in any run
    has_gt = any(r.term_recall is not None for r in results)
    if has_gt:
        lines.extend([
            "",
            "## 3. Ground Truth Accuracy",
            "",
            "| Video | Backend | Model | Term Recall | False Positives | Sample WER |",
            "|---|---|---|---|---|---|",
        ])
        for r in results:
            recall_str = f"{r.term_recall:.1f}%" if r.term_recall is not None else "N/A"
            fp_str = str(r.must_not_contain_hits) if r.must_not_contain_hits is not None else "N/A"
            wer_str = f"{r.mean_sample_wer:.1f}%" if r.mean_sample_wer is not None else "N/A"
            lines.append(
                f"| `{r.slug}` | `{r.backend}` | `{r.model}` | "
                f"{recall_str} | {fp_str} | {wer_str} |"
            )

    atomic_write_text(report_path, "\n".join(lines) + "\n")
