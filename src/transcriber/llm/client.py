"""OpenAI-compatible HTTP client for transcript correction."""

import json
import logging
import random
import re
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

from transcriber.config import Settings

logger = logging.getLogger("transcriber.llm")


class LlmError(Exception):
    """Base exception for LLM operations."""
    pass


class LlmUnavailable(LlmError):
    """Raised when LLM endpoint is unreachable after retries."""
    pass


class LlmProtocolError(LlmError):
    """Raised when LLM response is malformed or violates protocol."""
    pass


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LlmClient:
    """HTTP client targeting OpenAI-compatible /chat/completions."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = (settings.llm_base_url or "").rstrip("/")
        self.api_key = settings.llm_api_key
        self.model = settings.llm_model
        self.timeout = settings.llm_timeout_s
        self.max_retries = settings.llm_max_retries
        self.response_format = settings.llm_response_format

        parsed = urlparse(self.base_url)
        self.host = parsed.netloc or parsed.path or "unknown"

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        self._client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=self.timeout,
        )

    def close(self) -> None:
        self._client.close()

    def complete(
        self,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        max_tokens: int,
    ) -> tuple[dict[str, Any], TokenUsage]:
        """Send chat completion request with JSON schema / object formatting and retries."""
        url = "/chat/completions"

        for attempt in range(self.max_retries + 1):
            rf_payload: dict[str, Any]
            if self.response_format == "json_schema":
                rf_payload = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "corrections",
                        "strict": True,
                        "schema": schema,
                    },
                }
            else:
                rf_payload = {"type": "json_object"}

            payload: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": self.settings.llm_temperature,
                "max_tokens": max_tokens,
                "response_format": rf_payload,
            }

            try:
                resp = self._client.post(url, json=payload)

                # Check for response_format unsupported errors (400 or 422)
                if resp.status_code in (400, 422) and self.response_format == "json_schema":
                    err_body = resp.text.lower()
                    if "response_format" in err_body or "json_schema" in err_body or "schema" in err_body:
                        logger.warning(
                            "LLM endpoint %s rejected json_schema; permanently downgrading to json_object",
                            self.host,
                        )
                        self.response_format = "json_object"
                        # Retry once with json_object
                        continue

                # Handle auth errors immediately without retrying
                if resp.status_code in (401, 403):
                    logger.error("LLM authentication failed (status %d). Check TRANSCRIBER_LLM_API_KEY", resp.status_code)
                    raise LlmError("Check TRANSCRIBER_LLM_API_KEY (authentication failed)")

                # Raise for status to trigger retries for 429 and 5xx
                resp.raise_for_status()

                # Parse response
                resp_json = resp.json()
                content = resp_json["choices"][0]["message"]["content"]

                # Extract token usage
                usage_raw = resp_json.get("usage", {})
                usage = TokenUsage(
                    prompt_tokens=usage_raw.get("prompt_tokens", 0),
                    completion_tokens=usage_raw.get("completion_tokens", 0),
                    total_tokens=usage_raw.get("total_tokens", 0),
                )

                # Parse JSON content with outermost salvage fallback
                parsed_content: dict[str, Any]
                try:
                    parsed_content = json.loads(content)
                except json.JSONDecodeError:
                    match = re.search(r"\{.*\}", content, re.DOTALL)
                    if match:
                        try:
                            parsed_content = json.loads(match.group(0))
                        except Exception as e:
                            raise LlmProtocolError(f"Failed to salvage JSON from LLM: {e}") from e
                    else:
                        raise LlmProtocolError("LLM response did not contain valid JSON")

                return parsed_content, usage

            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as e:
                status_code = getattr(getattr(e, "response", None), "status_code", None)
                if status_code in (401, 403):
                    raise LlmError("Check TRANSCRIBER_LLM_API_KEY (authentication failed)") from e

                if attempt == self.max_retries:
                    logger.error("LLM request failed after %d retries: %s", self.max_retries, e)
                    raise LlmUnavailable(f"LLM endpoint {self.host} unavailable: {e}") from e

                # Determine backoff duration
                sleep_s = min(2**attempt, 16) * (1.0 + random.random() * 0.25)
                if hasattr(e, "response") and e.response is not None:
                    retry_after = e.response.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        sleep_s = max(float(retry_after), sleep_s)

                logger.warning(
                    "LLM request error (%s). Retrying in %.2f s (attempt %d/%d)...",
                    e,
                    sleep_s,
                    attempt + 1,
                    self.max_retries,
                )
                time.sleep(sleep_s)

        raise LlmUnavailable(f"LLM endpoint {self.host} failed after {self.max_retries} attempts")
