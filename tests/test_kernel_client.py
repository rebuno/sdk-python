import hashlib
import hmac
import json

import httpx2
import pytest

from rebuno._kernel import DispatchLease, KernelClient
from rebuno.errors import APIError, LeaseSuperseded
from rebuno.types import StepDecision

SECRET = "dev-secret"
AGENT = "dev-agent"
LEASE = DispatchLease("d1", 3)


def _sig(body: bytes) -> str:
    return "sha256=" + hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()


@pytest.fixture
def captured():
    return {}


@pytest.fixture
def client(captured):
    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["request"] = request
        captured["body"] = request.content
        if request.url.path.endswith("/steps"):
            return httpx2.Response(
                200, json={"decision": "proceed", "step_id": "sid123"}
            )
        return httpx2.Response(200, json={"decision": "recorded"})

    transport = httpx2.MockTransport(handler)
    http = httpx2.AsyncClient(transport=transport, base_url="http://k")
    return KernelClient(agent_id=AGENT, secret=SECRET, http=http)


async def test_submit_step_returns_the_kernel_step_id(client, captured):
    dec = await client.submit_step(
        "e1",
        lease=LEASE,
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
    assert req.headers["Rebuno-Signature"] == _sig(body)
    # Args go as plain JSON — the kernel canonicalizes what it receives before hashing.
    assert json.loads(body)["args"] == {"b": 2, "a": 1}


async def test_complete_step_posts_result(client, captured):
    await client.complete_step("e1", "sid123", lease=LEASE, result={"ok": True})
    body = json.loads(captured["body"])
    assert body == {"result": {"ok": True}}
    assert captured["request"].headers["Rebuno-Signature"] == _sig(captured["body"])


@pytest.mark.parametrize(
    "call",
    [
        lambda c: c.submit_step(
            "e1",
            lease=LEASE,
            kind="tool_call",
            target="t",
            args={},
            idempotency="safe_to_retry",
        ),
        lambda c: c.complete_step("e1", "sid123", lease=LEASE, result=None),
        lambda c: c.fail_step("e1", "sid123", lease=LEASE, error={"message": "x"}),
        lambda c: c.heartbeat("e1", lease=LEASE),
        lambda c: c.complete_execution("e1", lease=LEASE, output={}),
        lambda c: c.fail_execution("e1", lease=LEASE, error="boom"),
    ],
    ids=[
        "submit_step",
        "complete_step",
        "fail_step",
        "heartbeat",
        "complete_execution",
        "fail_execution",
    ],
)
async def test_every_mutation_carries_the_lease(client, captured, call):
    """The kernel fences each mutation on the delivery attempt that issued it, so
    one sent without the lease is refused outright."""
    await call(client)
    headers = captured["request"].headers
    assert headers["Rebuno-Dispatch-Id"] == "d1"
    assert headers["Rebuno-Dispatch-Attempt"] == "3"


async def test_superseded_lease_maps_to_its_control_flow_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            409,
            json={"code": "lease_superseded", "message": "dispatch lease superseded"},
        )

    http = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://k"
    )
    client = KernelClient(agent_id=AGENT, secret=SECRET, http=http)
    with pytest.raises(LeaseSuperseded):
        await client.complete_execution("e1", lease=LEASE, output={})


async def test_stream_delta_posts_seq_and_data(client, captured):
    await client.stream_delta("e1", "sid123", seq=4, data="tok")
    body = json.loads(captured["body"])
    assert body == {"seq": 4, "data": "tok"}
    req = captured["request"]
    assert req.url.path == "/v0/executions/e1/steps/sid123/stream"
    assert req.headers["Rebuno-Signature"] == _sig(captured["body"])


async def test_conflict_maps_to_api_error():
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            409, json={"code": "conflict", "message": "already exists"}
        )

    http = httpx2.AsyncClient(
        transport=httpx2.MockTransport(handler), base_url="http://k"
    )
    client = KernelClient(agent_id=AGENT, secret=SECRET, http=http)
    with pytest.raises(APIError) as exc_info:
        await client.get_execution("e1")
    assert exc_info.value.code == "conflict"
