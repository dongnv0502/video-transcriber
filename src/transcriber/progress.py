"""Terminal progress reporting matching the specification display contracts."""

from rich.console import Console

console = Console()


def format_duration_human(seconds: float | None) -> str:
    """Format seconds into human-readable string like '46m12s' or '1h05m20s'."""
    if seconds is None or seconds <= 0:
        return "0s"
    total_s = int(round(seconds))
    m, s = divmod(total_s, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    elif m > 0:
        return f"{m}m{s:02d}s"
    else:
        return f"{s}s"


def print_progress_inflight(
    index: int,
    total: int,
    slug: str,
    progress_pct: float,
    segments: int,
    suspicious: int,
    status: str,
) -> None:
    """Print in-flight progress block per specification §14."""
    pct = int(round(progress_pct * 100))
    msg = (
        f"[{index:03d}/{total:03d}] {slug}\n"
        f"progress: {pct}%\n"
        f"segments: {segments}\n"
        f"suspicious: {suspicious}\n"
        f"status: {status}"
    )
    console.print(msg)


def print_progress_completed(
    slug: str,
    output_path: str,
    duration_s: float,
    processing_s: float,
    segments: int,
    needs_review: int,
) -> None:
    """Print completion summary block per specification §14."""
    dur_str = format_duration_human(duration_s)
    proc_str = format_duration_human(processing_s)
    msg = (
        f"completed\n"
        f"output: {output_path}\n"
        f"duration: {dur_str}\n"
        f"processing: {proc_str}\n"
        f"segments: {segments}\n"
        f"needs_review: {needs_review}"
    )
    console.print(msg)
