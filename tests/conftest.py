"""Shared test fixtures and helpers."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from rebuno.client import Client
from rebuno.types import ClaimResult, Job


def mock_client(routes: dict[tuple[str, str], Any]) -> Client:
    """Return a Client whose underlying httpx transport is mocked.

    ``routes`` maps (method, path) to either:
      - a dict (returned as JSON 200)
      - an int status code with no body
      - a tuple (status_code, json_body)
      - a callable taking the httpx.Request and returning httpx.Response

    Unknown routes return 404 to make missing setups obvious.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        key = (request.method, request.url.path)
        spec = routes.get(key)
        if spec is None:
            return httpx.Response(404, json={"error": f"no mock for {key}"})
        if callable(spec):
            return spec(request)
        if isinstance(spec, int):
            return httpx.Response(spec)
        if isinstance(spec, tuple):
            status, body = spec
            return httpx.Response(status, json=body)
        return httpx.Response(200, json=spec)

    transport = httpx.MockTransport(handler)
    client = Client(base_url="http://test", api_key="test-key")
    client._http = httpx.AsyncClient(
        base_url="http://test",
        headers={"Content-Type": "application/json"},
        transport=transport,
    )
    return client


def make_response(
    status_code: int = 200,
    json_data: Any = None,
    text: str = "",
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    if json_data is not None:
        content = json.dumps(json_data).encode()
        default_headers = {"content-type": "application/json"}
    else:
        content = text.encode()
        default_headers = {}
    if headers:
        default_headers.update(headers)
    return httpx.Response(
        status_code=status_code,
        content=content,
        headers=default_headers,
        request=httpx.Request("GET", "http://test"),
    )


SAMPLE_EXECUTION = {
    "id": "exec-1",
    "status": "running",
    "agent_id": "agent-1",
    "labels": {"env": "test"},
}


def make_claim(**overrides: Any) -> ClaimResult:
    defaults = {
        "execution_id": "exec-1",
        "session_id": "sess-1",
        "agent_id": "agent-1",
        "input": {"prompt": "hello"},
        "labels": {},
        "history": [],
    }
    defaults.update(overrides)
    return ClaimResult(**defaults)


def make_job(**overrides: Any) -> Job:
    defaults = {
        "id": "job-1",
        "execution_id": "exec-1",
        "step_id": "step-1",
        "tool_id": "web.search",
        "arguments": {"q": "test"},
    }
    defaults.update(overrides)
    return Job(**defaults)


@pytest.fixture(autouse=True)
def _clear_tool_registry():
    """Each test starts with an empty @tool registry."""
    from rebuno.tool import _clear_registry

    _clear_registry()
    yield
    _clear_registry()


@pytest.fixture(autouse=True)
def _clear_mcp_registry():
    from rebuno.mcp import _clear_registry

    _clear_registry()
    yield
    _clear_registry()


@pytest.fixture(autouse=True)
def _clear_remote_registry():
    from rebuno.remote import _clear_registry

    _clear_registry()
    yield
    _clear_registry()
