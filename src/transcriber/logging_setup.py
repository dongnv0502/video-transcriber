"""Logging configuration with rich console output and rotating file handler."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from rich.logging import RichHandler

class DefaultContextFilter(logging.Filter):
    """Ensure default values for video, stage, attempt on all log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "video"):
            record.video = "-"
        if not hasattr(record, "stage"):
            record.stage = "-"
        if not hasattr(record, "attempt"):
            record.attempt = "-"
        return True


class StageLoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that injects video, stage, and attempt into log records."""

    def __init__(self, logger: logging.Logger, video: str = "-", stage: str = "-", attempt: str | int = "-"):
        super().__init__(logger, {"video": video, "stage": stage, "attempt": str(attempt)})
        return msg, kwargs


def stage_logger(
    logger: logging.Logger | None = None,
    video: str = "-",
    stage: str = "-",
    attempt: str | int = "-",
) -> StageLoggerAdapter:
    """Create a StageLoggerAdapter for structured logging across pipeline stages."""
    base_logger = logger or logging.getLogger("transcriber")
    return StageLoggerAdapter(base_logger, video=video, stage=stage, attempt=attempt)


def setup_logging(log_dir: Path | str, verbose: bool = False) -> logging.Logger:
    """Configure root transcriber logger with file and console handlers."""
    dir_path = Path(log_dir).resolve()
    dir_path.mkdir(parents=True, exist_ok=True)
    log_file = dir_path / "app.log"

    logger = logging.getLogger("transcriber")
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers if setup_logging is called multiple times
    if logger.handlers:
        logger.handlers.clear()

    # Formatter for structured logs
    file_formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s video=%(video)s stage=%(stage)s attempt=%(attempt)s %(message)s"
    )

    # Rotating file handler at DEBUG
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    file_handler.addFilter(DefaultContextFilter())
    logger.addHandler(file_handler)

    # Rich console handler at INFO (or DEBUG if verbose)
    console_handler = RichHandler(
        level=logging.DEBUG if verbose else logging.INFO,
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
    )
    # Console formatting uses standard rich style; extra fields are attached if present
    logger.addHandler(console_handler)

    return logger
