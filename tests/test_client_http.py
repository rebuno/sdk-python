"""Client HTTP semantics: status code mapping, retries, idempotency."""

from __future__ import annotations

import httpx
import pytest
from conftest import mock_client

from rebuno.errors import (
    APIError,
    ConflictError,
    NetworkError,
    NotFoundError,
    PolicyError,
    UnauthorizedError,
    ValidationError,
)


async def test_400_raises_validation_error():
    client = mock_client({("GET", "/v0/executions/x"): (400, {"error": "bad"})})
    with pytest.raises(ValidationError):
        await client.get("x")


async def test_401_raises_unauthorized():
    client = mock_client({("GET", "/v0/executions/x"): (401, {"error": "nope"})})
    with pytest.raises(UnauthorizedError):
        await client.get("x")


async def test_404_raises_not_found():
    client = mock_client({("GET", "/v0/executions/x"): (404, {"error": "missing"})})
    with pytest.raises(NotFoundError):
        await client.get("x")


async def test_403_raises_policy_error_with_rule_id():
    client = mock_client(
        {
            ("GET", "/v0/executions/x"): (403, {"error": "denied", "rule_id": "no-secrets"}),
        }
    )
    with pytest.raises(PolicyError) as exc_info:
        await client.get("x")
    assert exc_info.value.rule_id == "no-secrets"


async def test_409_raises_conflict_error():
    client = mock_client(
        {
            ("POST", "/v0/agents/intent"): (409, {"error": "already resolved"}),
        }
    )
    with pytest.raises(ConflictError):
        await client.submit_intent(
            execution_id="e",
            session_id="s",
            intent_type="invoke_tool",
            tool_id="t",
            arguments={},
        )


async def test_500_retries_then_succeeds():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(500, json={"error": "boom"})
        return httpx.Response(
            200,
            json={
                "id": "e",
                "status": "running",
                "agent_id": "a",
                "labels": {},
            },
        )

    client = mock_client({("GET", "/v0/executions/e"): handler})
    client.retry_base_delay = 0  # fast
    ex = await client.get("e")
    assert ex.id == "e"
    assert calls["n"] == 3


async def test_500_exhausts_retries():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500, json={"error": "boom"})

    client = mock_client({("GET", "/v0/executions/e"): handler})
    client.retry_base_delay = 0
    client.max_retries = 2
    with pytest.raises(APIError):
        await client.get("e")
    assert calls["n"] == 3  # initial + 2 retries


async def test_429_respects_retry_after():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(
            200,
            json={
                "id": "e",
                "status": "running",
                "agent_id": "a",
                "labels": {},
            },
        )

    client = mock_client({("GET", "/v0/executions/e"): handler})
    ex = await client.get("e")
    assert ex.id == "e"
    assert calls["n"] == 2


async def test_post_not_idempotent_does_not_retry_on_500():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500, json={"error": "boom"})

    client = mock_client({("POST", "/v0/executions"): handler})
    client.retry_base_delay = 0
    with pytest.raises(APIError):
        await client.create("agent")
    assert calls["n"] == 1  # not retried


async def test_post_idempotent_retries_on_500():
    """submit_intent uses idempotent=True so it should retry."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 2:
            return httpx.Response(500, json={"error": "boom"})
        return httpx.Response(200, json={"accepted": True, "step_id": "s"})

    client = mock_client({("POST", "/v0/agents/intent"): handler})
    client.retry_base_delay = 0
    result = await client.submit_intent(
        execution_id="e",
        session_id="s",
        intent_type="complete",
    )
    assert result.accepted
    assert calls["n"] == 2


async def test_connect_error_retries():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 2:
            raise httpx.ConnectError("kernel unreachable")
        return httpx.Response(
            200,
            json={
                "id": "e",
                "status": "running",
                "agent_id": "a",
                "labels": {},
            },
        )

    client = mock_client({("GET", "/v0/executions/e"): handler})
    client.retry_base_delay = 0
    ex = await client.get("e")
    assert ex.id == "e"
    assert calls["n"] == 2


async def test_connect_error_on_non_idempotent_post_does_not_retry():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    client = mock_client({("POST", "/v0/executions"): handler})
    with pytest.raises(NetworkError):
        await client.create("agent")
