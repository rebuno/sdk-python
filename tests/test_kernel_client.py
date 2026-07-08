import hashlib
import hmac
import json

import httpx
import pytest

from rebuno._kernel import KernelClient
from rebuno.errors import APIError, StepIDMismatch
from rebuno.types import StepDecision

SECRET = "dev-secret"
AGENT = "dev-agent"


def _sig(body: bytes) -> str:
    return "sha256=" + hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()


@pytest.fixture
def captured():
    return {}


@pytest.fixture
def client(captured):
    def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        captured["body"] = request.content
        if request.url.path.endswith("/steps"):
            return httpx.Response(200, json={"decision": "proceed"})
        return httpx.Response(200, json={"decision": "recorded"})

    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport, base_url="http://k")
    return KernelClient(agent_id=AGENT, secret=SECRET, http=http)


async def test_submit_step_signs_and_embeds_canonical_args(client, captured):
    dec = await client.submit_step(
        "e1",
        kind="tool_call",
        target="t",
        args={"b": 2, "a": 1},
        idempotency="safe_to_retry",
        step_id="sid123",
    )
    assert isinstance(dec, StepDecision)
    req = captured["request"]
    body = captured["body"]
    assert req.headers["Rebuno-Agent-Id"] == AGENT
    assert req.headers["Rebuno-Step-Id"] == "sid123"
    assert req.headers["Rebuno-Signature"] == _sig(body)
    # canonical args embedded verbatim (sorted keys, compact)
    assert b'"args":{"a":1,"b":2}' in body


async def test_complete_step_posts_result(client, captured):
    await client.complete_step("e1", "sid123", result={"ok": True})
    body = json.loads(captured["body"])
    assert body == {"result": {"ok": True}}
    assert captured["request"].headers["Rebuno-Signature"] == _sig(captured["body"])


async def test_step_id_divergence_maps_to_step_id_mismatch():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"code": "step_id_divergence", "message": "mismatch"})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://k")
    client = KernelClient(agent_id=AGENT, secret=SECRET, http=http)
    with pytest.raises(StepIDMismatch):
        await client.get_execution("e1")


async def test_conflict_maps_to_api_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"code": "conflict", "message": "already exists"})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://k")
    client = KernelClient(agent_id=AGENT, secret=SECRET, http=http)
    with pytest.raises(APIError) as exc_info:
        await client.get_execution("e1")
    assert exc_info.value.code == "conflict"
