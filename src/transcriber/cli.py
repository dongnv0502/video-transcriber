"""Command-line interface (CLI) for the AWS Video Transcriber pipeline."""

import json
import logging
import os
import platform
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Optional

import httpx
import typer
from rich.console import Console
from rich.table import Table

from transcriber import __version__
from transcriber.config import ConfigError, Settings
from transcriber.db import (
    COMPLETED,
    FAILED,
    NEEDS_REVIEW,
    PENDING,
    TRANSCRIBED,
    Status,
    get_connection,
    get_status_counts,
    get_videos_for_correct,
    get_videos_for_transcribe,
    init_db,
    set_status,
)
from transcriber.logging_setup import setup_logging, stage_logger
from transcriber.progress import print_progress_completed, print_progress_inflight
from transcriber.stages.export import export_video
from transcriber.stages.scan import scan
from transcriber.stages.transcribe import transcribe_video

app = typer.Typer(
    help="AWS Video Transcriber: Local ASR pipeline for Vietnamese AWS courses.",
    no_args_is_help=True,
)
console = Console()


def get_configured_settings(
    config: Path | None = None,
    verbose: bool = False,
    no_llm: bool = False,
    video_dir: Path | None = None,
    output_dir: Path | None = None,
    work_dir: Path | None = None,
    db_path: Path | None = None,
    log_dir: Path | None = None,
    model_name: str | None = None,
) -> Settings:
    """Load settings with CLI overrides and initialize logging and database."""
    overrides: dict[str, Any] = {}
    if no_llm:
        overrides["llm_enabled"] = False
    if video_dir is not None:
        overrides["video_dir"] = video_dir
    if output_dir is not None:
        overrides["output_dir"] = output_dir
    if work_dir is not None:
        overrides["work_dir"] = work_dir
    if db_path is not None:
        overrides["db_path"] = db_path
    if log_dir is not None:
        overrides["log_dir"] = log_dir
    if model_name is not None:
        overrides["model_name"] = model_name

    settings = Settings.load(config, **overrides)
    setup_logging(settings.log_dir, verbose=verbose)
    init_db(settings.db_path)
    return settings


@app.command()
def doctor(
    config: Annotated[Optional[Path], typer.Option("--config", "-c", help="Config file")] = None,
    no_llm: Annotated[bool, typer.Option("--no-llm", help="Bypass LLM checks")] = False,
) -> None:
    """Diagnose system readiness, hardware, dependencies, and LLM endpoint."""
    console.print(f"[bold cyan]AWS Video Transcriber v{__version__} — Doctor[/bold cyan]\n")
    settings = Settings.load(config, llm_enabled=not no_llm if no_llm else None)

    all_passed = True

    # 1. Architecture & macOS
    machine = platform.machine()
    is_arm64 = machine == "arm64"
    mac_ver, _, _ = platform.mac_ver()
    if is_arm64:
        console.print(f"  [green]✓[/green] Architecture: {machine} (macOS {mac_ver})")
    else:
        console.print(f"  [red]✗[/red] Architecture: {machine} (requires Apple Silicon arm64)")
        all_passed = False

    # 2. Python version
    py_ver = sys.version.split()[0]
    is_py312 = sys.version_info.major == 3 and sys.version_info.minor == 12
    if is_py312:
        console.print(f"  [green]✓[/green] Python: {py_ver}")
    else:
        console.print(f"  [yellow]![/yellow] Python: {py_ver} (recommended CPython 3.12.x)")

    # 3. Memory
    try:
        mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        mem_gib = mem_bytes / (1024**3)
        console.print(f"  [green]✓[/green] System RAM: {mem_gib:.1f} GiB")
    except Exception:
        console.print("  [green]✓[/green] System RAM: available")

    # 4. FFmpeg and FFprobe
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")
    if ffmpeg_path and ffprobe_path:
        console.print(f"  [green]✓[/green] FFmpeg: {ffmpeg_path}")
        console.print(f"  [green]✓[/green] FFprobe: {ffprobe_path}")
    else:
        console.print("  [red]✗[/red] FFmpeg or FFprobe missing from PATH")
        all_passed = False

    # 5. MLX Core & Metal
    try:
        import mlx.core as mx

        metal_avail = getattr(mx.metal, "is_available", lambda: True)()
        if metal_avail:
            console.print("  [green]✓[/green] MLX Core: installed (Metal acceleration available)")
        else:
            console.print("  [yellow]![/yellow] MLX Core: installed (Metal acceleration not reported)")
    except Exception as e:
        console.print(f"  [red]✗[/red] MLX Core unavailable: {e}")
        all_passed = False

    # 6. Disk space for output
    out_dir = settings.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(out_dir)
    free_gib = usage.free / (1024**3)
    console.print(f"  [green]✓[/green] Disk space ({out_dir.name}/): {free_gib:.1f} GiB free")

    # 7. Model cache check
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(settings.model_name, "config.json")
        if cached and isinstance(cached, str):
            console.print(f"  [green]✓[/green] Model cache ({settings.model_name}): cached locally")
        else:
            console.print(f"  [yellow]![/yellow] Model cache ({settings.model_name}): will download on first run")
    except Exception:
        console.print(f"  [green]✓[/green] Model ({settings.model_name}): ready to load")

    # 8. LLM Endpoint check
    if settings.llm_enabled:
        if not settings.llm_base_url or not settings.llm_model:
            console.print(
                "  [yellow]![/yellow] LLM Correction: enabled but TRANSCRIBER_LLM_BASE_URL "
                "or TRANSCRIBER_LLM_MODEL is not set"
            )
        else:
            try:
                from urllib.parse import urlparse

                parsed = urlparse(settings.llm_base_url)
                host = parsed.netloc or parsed.path
                url = f"{settings.llm_base_url.rstrip('/')}/models"
                headers = {}
                if settings.llm_api_key:
                    headers["Authorization"] = f"Bearer {settings.llm_api_key}"

                with httpx.Client(timeout=10.0) as client:
                    resp = client.get(url, headers=headers)
                    console.print(f"  [green]✓[/green] LLM Endpoint: {host} (HTTP {resp.status_code})")
            except Exception as e:
                console.print(f"  [yellow]![/yellow] LLM Endpoint ({host}): unreachable ({e})")
    else:
        console.print("  [blue]i[/blue] LLM Correction: disabled (--no-llm)")

    console.print()
    if all_passed:
        console.print("[bold green]System is ready for transcription.[/bold green]")
    else:
        console.print("[bold red]Please resolve missing dependencies before continuing.[/bold red]")
        sys.exit(1)


@app.command(name="scan")
def scan_command(
    video_dir: Annotated[Optional[Path], typer.Argument(help="Video directory to scan")] = None,
    config: Annotated[Optional[Path], typer.Option("--config", "-c", help="Config file")] = None,
    rescan: Annotated[bool, typer.Option("--rescan", help="Reset videos with changed size/mtime")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose output")] = False,
) -> None:
    """Scan video directory for MP4/M4V/MOV files and update state database."""
    settings = get_configured_settings(config=config, verbose=verbose, video_dir=video_dir)
    conn = get_connection(settings.db_path)

    report = scan(conn, settings, rescan=rescan)
    console.print(
        f"Scan complete: [bold]{report.total}[/bold] total videos "
        f"([green]+{report.added} added[/green], "
        f"[yellow]~{report.updated} updated[/yellow], "
        f"[blue]{report.unchanged} unchanged[/blue])"
    )

@app.command()
def status(
    config: Annotated[Optional[Path], typer.Option("--config", "-c", help="Config file")] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Output machine-readable JSON")] = False,
) -> None:
    """Report processing status counts across all discovered videos."""
    settings = get_configured_settings(config=config)
    conn = get_connection(settings.db_path)

    counts = get_status_counts(conn)
    total = sum(counts.values())

    if as_json:
        payload = {"total": total, "status_counts": counts}
        print(json.dumps(payload, indent=2))
        return

    table = Table(title="AWS Video Transcriber — Batch Status")
    table.add_column("Status", style="bold")
    table.add_column("Count", justify="right")
    table.add_column("Percentage", justify="right")

    for st in (
        Status.COMPLETED.value,
        Status.NEEDS_REVIEW.value,
        Status.TRANSCRIBED.value,
        Status.TRANSCRIBING.value,
        Status.EXTRACTING.value,
        Status.PENDING.value,
        Status.FAILED.value,
    ):
        cnt = counts.get(st, 0)
        pct = (cnt / max(total, 1)) * 100
        table.add_row(st, str(cnt), f"{pct:.1f}%")

    console.print(table)
    completed_cnt = counts.get(COMPLETED, 0)
    failed_cnt = counts.get(FAILED, 0)
    pending_cnt = counts.get(PENDING, 0)
    console.print(
        f"\n[bold]{total} total[/bold] / [green]{completed_cnt} completed[/green] / "
        f"[red]{failed_cnt} failed[/red] / [yellow]{pending_cnt} pending[/yellow]"
    )


@app.command()
def transcribe(
    config: Annotated[Optional[Path], typer.Option("--config", "-c", help="Config file")] = None,
    video: Annotated[Optional[Path], typer.Option("--video", help="Filter to specific video path")] = None,
    limit: Annotated[Optional[int], typer.Option("--limit", "-n", help="Max videos to process")] = None,
    force: Annotated[bool, typer.Option("--force", "-f", help="Force re-transcription")] = False,
    no_llm: Annotated[bool, typer.Option("--no-llm", help="Disable LLM")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose output")] = False,
) -> None:
    """Run audio extraction and ASR transcription on pending videos."""
    settings = get_configured_settings(config=config, verbose=verbose, no_llm=no_llm)
    conn = get_connection(settings.db_path)

    if video:
        p_abs = video.resolve().as_posix()
        rows = conn.execute("SELECT * FROM videos WHERE path = ?", (p_abs,)).fetchall()
    else:
        rows = get_videos_for_transcribe(conn, force=force, limit=limit)

    if not rows:
        console.print("No videos pending transcription.")
        return

    total_count = len(rows)
    console.print(f"Starting transcription of [bold]{total_count}[/bold] video(s)...\n")

    for idx, row in enumerate(rows, start=1):
        slug = row["slug"]
        video_id = row["id"]

        def on_prog(pct: float, segs: int) -> None:
            print_progress_inflight(
                index=idx,
                total=total_count,
                slug=slug,
                progress_pct=pct,
                segments=segs,
                suspicious=0,
                status="transcribing",
            )

        try:
            raw_t = transcribe_video(conn, settings, row, force=force, on_progress=on_prog)
            out_dir = settings.output_dir.resolve() / slug
            print_progress_completed(
                slug=slug,
                output_path=f"output/{slug}/",
                duration_s=raw_t.video.duration_seconds or 0.0,
                processing_s=raw_t.transcription.processing_seconds,
                segments=len(raw_t.segments),
                needs_review=0,
            )
            console.print()
        except KeyboardInterrupt:
            console.print("\n[bold red]Interrupted by user. Resetting status...[/bold red]")
            set_status(conn, video_id, Status.PENDING)
            sys.exit(130)
        except Exception as e:
            console.print(f"[bold red]Failed transcribing {slug}: {e}[/bold red]")
            continue


@app.command()
def correct(
    config: Annotated[Optional[Path], typer.Option("--config", "-c", help="Config file")] = None,
    video: Annotated[Optional[Path], typer.Option("--video", help="Filter to specific video path")] = None,
    limit: Annotated[Optional[int], typer.Option("--limit", "-n", help="Max videos to process")] = None,
    force: Annotated[bool, typer.Option("--force", "-f", help="Force re-correction")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Simulate correction without saving")] = False,
    no_llm: Annotated[bool, typer.Option("--no-llm", help="Disable LLM correction")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose output")] = False,
) -> None:
    """Run deterministic and LLM-assisted correction on transcribed videos."""
    from transcriber.stages.correct import correct_video

    settings = get_configured_settings(config=config, verbose=verbose, no_llm=no_llm)
    conn = get_connection(settings.db_path)

    if video:
        p_abs = video.resolve().as_posix()
        rows = conn.execute("SELECT * FROM videos WHERE path = ?", (p_abs,)).fetchall()
    else:
        rows = get_videos_for_correct(conn, force=force, limit=limit)

    if not rows:
        console.print("No videos ready for correction.")
        return

    console.print(f"Starting correction for [bold]{len(rows)}[/bold] video(s)...\n")

    for row in rows:
        slug = row["slug"]
        video_id = row["id"]
        try:
            res = correct_video(conn, settings, row, dry_run=dry_run, force=force)
            stats = res.correction.stats
            console.print(
                f"[bold green]Corrected {slug}:[/bold green] "
                f"flagged={stats.get('flagged', 0)}, "
                f"applied_llm={stats.get('applied_llm', 0)}, "
                f"applied_rules={stats.get('applied_rule', 0)}, "
                f"rejected={stats.get('rejected', 0)}, "
                f"needs_review={stats.get('needs_review', 0)}"
            )
        except KeyboardInterrupt:
            console.print("\n[bold red]Interrupted by user.[/bold red]")
        except Exception as e:
            console.print(f"[bold red]Failed correcting {slug}: {e}[/bold red]")
            continue


@app.command()
def export(
    config: Annotated[Optional[Path], typer.Option("--config", "-c", help="Config file")] = None,
    video: Annotated[Optional[Path], typer.Option("--video", help="Filter to specific video path")] = None,
    limit: Annotated[Optional[int], typer.Option("--limit", "-n", help="Max videos to process")] = None,
    force: Annotated[bool, typer.Option("--force", "-f", help="Force re-export")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose output")] = False,
) -> None:
    """Export transcripts to SRT, TXT, and Markdown files."""
    settings = get_configured_settings(config=config, verbose=verbose)
    conn = get_connection(settings.db_path)

    if video:
        p_abs = video.resolve().as_posix()
        rows = conn.execute("SELECT * FROM videos WHERE path = ?", (p_abs,)).fetchall()
    else:
        # Export any transcribed or completed videos
        query = "SELECT * FROM videos WHERE status IN ('TRANSCRIBED', 'COMPLETED', 'NEEDS_REVIEW')"
        if limit:
            query += f" LIMIT {limit}"
        rows = conn.execute(query).fetchall()

    if not rows:
        console.print("No videos ready for export.")
        return

    console.print(f"Exporting transcripts for [bold]{len(rows)}[/bold] video(s)...\n")

    for row in rows:
        slug = row["slug"]
        try:
            out_p = export_video(settings, row, force=force, conn=conn)
            console.print(f"  [green]✓[/green] Exported {slug} -> {out_p}")
        except Exception as e:
            console.print(f"  [red]✗[/red] Failed exporting {slug}: {e}")


@app.command()
def resume(
    config: Annotated[Optional[Path], typer.Option("--config", "-c", help="Config file")] = None,
    video: Annotated[Optional[Path], typer.Option("--video", help="Filter to specific video path")] = None,
    limit: Annotated[Optional[int], typer.Option("--limit", "-n", help="Max videos to process")] = None,
    force: Annotated[bool, typer.Option("--force", "-f", help="Force re-processing")] = False,
    no_llm: Annotated[bool, typer.Option("--no-llm", help="Disable LLM correction")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose output")] = False,
) -> None:
    """Resume the pipeline: run transcribe, then correct, then export."""
    console.print("[bold cyan]=== Stage 1: Transcription ===[/bold cyan]")
    transcribe(config=config, video=video, limit=limit, force=force, verbose=verbose)

    console.print("\n[bold cyan]=== Stage 2: Correction ===[/bold cyan]")
    correct(config=config, video=video, limit=limit, force=force, no_llm=no_llm, verbose=verbose)

    console.print("\n[bold cyan]=== Stage 3: Export ===[/bold cyan]")
    export(config=config, video=video, limit=limit, force=force, verbose=verbose)


@app.command()
def process(
    video_dir: Annotated[Optional[Path], typer.Argument(help="Video directory to process")] = None,
    config: Annotated[Optional[Path], typer.Option("--config", "-c", help="Config file")] = None,
    video: Annotated[Optional[Path], typer.Option("--video", help="Process specific video")] = None,
    limit: Annotated[Optional[int], typer.Option("--limit", "-n", help="Max videos to process")] = None,
    force: Annotated[bool, typer.Option("--force", "-f", help="Force re-processing")] = False,
    rescan: Annotated[bool, typer.Option("--rescan", help="Rescan video files")] = False,
    no_llm: Annotated[bool, typer.Option("--no-llm", help="Disable LLM correction")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose output")] = False,
) -> None:
    """Scan video directory and resume pipeline end-to-end."""
    console.print("[bold cyan]=== Step 1: Scan ===[/bold cyan]")
    scan_command(video_dir=video_dir, config=config, rescan=rescan, verbose=verbose)

    console.print("\n[bold cyan]=== Step 2: Pipeline Execution ===[/bold cyan]")
    resume(config=config, video=video, limit=limit, force=force, no_llm=no_llm, verbose=verbose)


@app.command()
def bench(
    videos: Annotated[list[Path], typer.Option("--videos", "-v", help="Sample video paths")] = [],
    backends: Annotated[str, typer.Option("--backends", help="Comma-separated backends: mlx,faster-whisper")] = "mlx",
    models: Annotated[str, typer.Option("--models", help="Comma-separated models")] = "mlx-community/whisper-large-v3-mlx",
    out: Annotated[Path, typer.Option("--out", help="Benchmark output directory")] = Path("benchmark"),
    with_correction: Annotated[bool, typer.Option("--with-correction", help="Include correction stage")] = False,
    config: Annotated[Optional[Path], typer.Option("--config", "-c", help="Config file")] = None,
) -> None:
    """Run quality and performance benchmark on sample videos."""
    from transcriber.benchmark import run_benchmark

    settings = get_configured_settings(config=config)
    backend_list = [b.strip() for b in backends.split(",") if b.strip()]
    model_list = [m.strip() for m in models.split(",") if m.strip()]

    results, report_path = run_benchmark(
        video_paths=videos,
        backends=backend_list,
        models=model_list,
        out_dir=out,
        settings=settings,
        with_correction=with_correction,
    )
    console.print(f"\n[bold green]Benchmark completed ({len(results)} runs):[/bold green] report -> {report_path}")


if __name__ == "__main__":
    app()
