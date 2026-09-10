"""Unit tests for llm/client.py using a hermetic local HTTP test server."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from transcriber.config import Settings
from transcriber.llm.client import LlmClient, LlmError, LlmProtocolError, LlmUnavailable


class MockLlmHandler(BaseHTTPRequestHandler):
    scenario = "happy"
    request_count = 0

    def log_message(self, format: str, *args: Any) -> None:
        pass  # Suppress console log spam

    def do_POST(self) -> None:
        MockLlmHandler.request_count += 1
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len)
        data = json.loads(body.decode("utf-8"))

        if self.scenario == "happy":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            resp = {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({
                                "results": [
                                    {
                                        "id": 1,
                                        "changed": True,
                                        "corrected": "EC2",
                                        "reason": "term",
                                        "confidence": 0.99,
                                        "needs_review": False,
                                    }
                                ]
                            })
                        }
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
            self.wfile.write(json.dumps(resp).encode("utf-8"))

        elif self.scenario == "retry_429":
            if MockLlmHandler.request_count == 1:
                self.send_response(429)
                self.send_header("Retry-After", "1")
                self.end_headers()
                self.wfile.write(b"Rate limit exceeded")
            else:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                resp = {
                    "choices": [{"message": {"content": json.dumps({"results": []})}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
                }
                self.wfile.write(json.dumps(resp).encode("utf-8"))

        elif self.scenario == "downgrade_400":
            # If request_format is json_schema, reject with 400
            rf = data.get("response_format", {})
            if rf.get("type") == "json_schema":
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"json_schema response_format not supported")
            else:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                resp = {
                    "choices": [{"message": {"content": json.dumps({"results": []})}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
                }
                self.wfile.write(json.dumps(resp).encode("utf-8"))

        elif self.scenario == "malformed_json":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            resp = {"choices": [{"message": {"content": "not a json object at all"}}]}
            self.wfile.write(json.dumps(resp).encode("utf-8"))

        elif self.scenario == "auth_401":
            self.send_response(401)
            self.end_headers()
            self.wfile.write(b"Unauthorized")


@pytest.fixture(scope="module")
def mock_server() -> tuple[str, int]:
    server = HTTPServer(("127.0.0.1", 0), MockLlmHandler)
    host, port = server.server_address
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield host, port
    server.shutdown()


def test_llm_client_happy_path(mock_server: tuple[str, int]) -> None:
    host, port = mock_server
    MockLlmHandler.scenario = "happy"
    MockLlmHandler.request_count = 0

    settings = Settings(
        llm_base_url=f"http://{host}:{port}/v1",
        llm_model="test-model",
        llm_timeout_s=5.0,
    )
    client = LlmClient(settings)
    res, usage = client.complete([], {}, max_tokens=100)
    client.close()

    assert "results" in res
    assert len(res["results"]) == 1
    assert res["results"][0]["corrected"] == "EC2"
    assert usage.prompt_tokens == 10


def test_llm_client_downgrade_on_400(mock_server: tuple[str, int]) -> None:
    host, port = mock_server
    MockLlmHandler.scenario = "downgrade_400"
    MockLlmHandler.request_count = 0

    settings = Settings(
        llm_base_url=f"http://{host}:{port}/v1",
        llm_model="test-model",
        llm_response_format="json_schema",
        llm_timeout_s=5.0,
    )
    client = LlmClient(settings)
    res, _ = client.complete([], {}, max_tokens=100)
    client.close()

    assert client.response_format == "json_object"
    assert "results" in res


def test_llm_client_malformed_json_raises_protocol_error(mock_server: tuple[str, int]) -> None:
    host, port = mock_server
    MockLlmHandler.scenario = "malformed_json"
    MockLlmHandler.request_count = 0

    settings = Settings(
        llm_base_url=f"http://{host}:{port}/v1",
        llm_model="test-model",
        llm_timeout_s=5.0,
    )
    client = LlmClient(settings)
    with pytest.raises(LlmProtocolError):
        client.complete([], {}, max_tokens=100)
    client.close()


def test_llm_client_auth_error_raises_immediately(mock_server: tuple[str, int]) -> None:
    host, port = mock_server
    MockLlmHandler.scenario = "auth_401"
    MockLlmHandler.request_count = 0

    settings = Settings(
        llm_base_url=f"http://{host}:{port}/v1",
        llm_model="test-model",
        llm_timeout_s=5.0,
    )
    client = LlmClient(settings)
    with pytest.raises(LlmError) as exc_info:
        client.complete([], {}, max_tokens=100)
    assert "TRANSCRIBER_LLM_API_KEY" in str(exc_info.value)
    client.close()
