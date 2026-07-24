import hashlib
import hmac
import json

import httpx
import pytest

from rebuno._kernel import KernelClient
from rebuno.errors import APIError
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
            return httpx.Response(200, json={"decision": "proceed", "step_id": "sid123"})
        return httpx.Response(200, json={"decision": "recorded"})

    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport, base_url="http://k")
    return KernelClient(agent_id=AGENT, secret=SECRET, http=http)


async def test_submit_step_sends_dispatch_and_returns_kernel_step_id(client, captured):
    dec = await client.submit_step(
        "e1",
        dispatch_id="d1",
        kind="tool_call",
        target="t",
        args={"b": 2, "a": 1},
        idempotency="safe_to_retry",
    )
    assert isinstance(dec, StepDecision)
    assert dec.step_id == "sid123"
    req = captured["request"]
    body = captured["body"]
    assert req.headers["Rebuno-Agent-Id"] == AGENT
    assert req.headers["Rebuno-Dispatch-Id"] == "d1"
    assert req.headers["Rebuno-Signature"] == _sig(body)
    # Args go as plain JSON — the kernel canonicalizes what it receives before hashing.
    assert json.loads(body)["args"] == {"b": 2, "a": 1}


async def test_complete_step_posts_result(client, captured):
    await client.complete_step("e1", "sid123", result={"ok": True})
    body = json.loads(captured["body"])
    assert body == {"result": {"ok": True}}
    assert captured["request"].headers["Rebuno-Signature"] == _sig(captured["body"])


async def test_stream_delta_posts_seq_and_data(client, captured):
    await client.stream_delta("e1", "sid123", seq=4, data="tok")
    body = json.loads(captured["body"])
    assert body == {"seq": 4, "data": "tok"}
    req = captured["request"]
    assert req.url.path == "/v0/executions/e1/steps/sid123/stream"
    assert req.headers["Rebuno-Signature"] == _sig(captured["body"])


async def test_conflict_maps_to_api_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"code": "conflict", "message": "already exists"})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://k")
    client = KernelClient(agent_id=AGENT, secret=SECRET, http=http)
    with pytest.raises(APIError) as exc_info:
        await client.get_execution("e1")
    assert exc_info.value.code == "conflict"
