"""Pipeline stages for transcription, correction, and export."""


class StageError(Exception):
    """Raised when an execution stage encounters a fatal error."""

    def __init__(self, stage: str, message: str):
        super().__init__(f"[{stage}] {message}")
        self.stage = stage
        self.message = message
