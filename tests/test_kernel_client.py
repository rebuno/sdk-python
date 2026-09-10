import json
from pathlib import Path

import httpx2
import pytest

from rebuno._kernel import DispatchLease, KernelClient
from rebuno.errors import APIError, LeaseSuperseded, Terminated
from rebuno.types import StepDecision

SECRET = "dev-secret"
AGENT = "dev-agent"
LEASE = DispatchLease("d1", 3, 120.0)


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
    assert json.loads(body)["args"] == {"b": 2, "a": 1}


async def test_complete_step_posts_result(client, captured):
    await client.complete_step("e1", "sid123", lease=LEASE, result={"ok": True})
    body = json.loads(captured["body"])
    assert body == {"result": {"ok": True}}


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
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            409,
            json={"code": "lease_superseded", "message": "dispatch lease superseded"},
        )

    http = httpx2.AsyncClient(
        transport=httpx2.MockTransport(handler), base_url="http://k"
    )
    client = KernelClient(agent_id=AGENT, secret=SECRET, http=http)
    with pytest.raises(LeaseSuperseded):
        await client.complete_execution("e1", lease=LEASE, output={})


async def test_terminal_execution_maps_to_its_control_flow_error():
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            409, json={"code": "execution_terminal", "message": "execution terminal"}
        )

    http = httpx2.AsyncClient(
        transport=httpx2.MockTransport(handler), base_url="http://k"
    )
    client = KernelClient(agent_id=AGENT, secret=SECRET, http=http)
    with pytest.raises(Terminated):
        await client.complete_execution("e1", lease=LEASE, output={})


async def test_stream_delta_posts_seq_and_data(client, captured):
    await client.stream_delta("e1", "sid123", seq=4, data="tok")
    body = json.loads(captured["body"])
    assert body == {"seq": 4, "data": "tok"}
    req = captured["request"]
    assert req.url.path == "/v0/executions/e1/steps/sid123/stream"


async def test_request_signature_vectors(monkeypatch):
    vectors = json.loads(
        (Path(__file__).parent / "fixtures/request-signatures.json").read_text()
    )
    for vector in vectors:
        monkeypatch.setattr(
            "rebuno._kernel.time.time", lambda v=vector: int(v["timestamp"])
        )
        captured = []

        def handler(request, captured=captured):
            captured.append(request)
            return httpx2.Response(200, json={})

        async with httpx2.AsyncClient(
            base_url="http://kernel", transport=httpx2.MockTransport(handler)
        ) as http:
            client = KernelClient(agent_id=AGENT, secret=vector["secret"], http=http)
            await client._send(
                vector["method"],
                vector["target"],
                vector["body"].encode(),
                {
                    "Rebuno-Dispatch-Id": vector["dispatch_id"],
                    "Rebuno-Dispatch-Attempt": vector["dispatch_attempt"],
                },
            )
        assert captured[0].headers["Rebuno-Signature"] == vector["signature"]


async def test_signature_covers_the_built_url_prefix_and_query(monkeypatch):
    monkeypatch.setattr("rebuno._kernel.time.time", lambda: 1700000000)
    captured = []

    def handler(request):
        captured.append(request)
        return httpx2.Response(200, json={})

    for base_url, params in (
        ("http://kernel", None),
        ("http://kernel/prefix/", {"status": "terminal"}),
    ):
        async with httpx2.AsyncClient(
            base_url=base_url, params=params, transport=httpx2.MockTransport(handler)
        ) as http:
            client = KernelClient(agent_id=AGENT, secret=SECRET, http=http)
            await client._send("GET", "/v0/executions/a%2Fb/steps", b"")
    assert (
        captured[1].url.raw_path == b"/prefix/v0/executions/a%2Fb/steps?status=terminal"
    )
    assert (
        captured[0].headers["Rebuno-Signature"]
        != captured[1].headers["Rebuno-Signature"]
    )


async def test_retry_gets_a_fresh_signature(monkeypatch):
    captured = []

    def handler(request):
        captured.append(request)
        return httpx2.Response(503 if len(captured) == 1 else 200, json={})

    async with httpx2.AsyncClient(
        base_url="http://kernel", transport=httpx2.MockTransport(handler)
    ) as http:
        client = KernelClient(agent_id=AGENT, secret=SECRET, http=http)
        monkeypatch.setattr("rebuno._kernel.time.time", lambda: 1700000000)
        with pytest.raises(APIError):
            await client.heartbeat("e1", lease=LEASE)
        monkeypatch.setattr("rebuno._kernel.time.time", lambda: 1700000030)
        await client.heartbeat("e1", lease=LEASE)
    assert captured[1].headers["Rebuno-Timestamp"] == "1700000030"
    assert (
        captured[0].headers["Rebuno-Signature"]
        != captured[1].headers["Rebuno-Signature"]
    )


async def test_signed_requests_do_not_follow_redirects():
    captured = []

    def handler(request):
        captured.append(request)
        return httpx2.Response(307, headers={"Location": "http://other/resource"})

    async with httpx2.AsyncClient(
        base_url="http://kernel",
        follow_redirects=True,
        transport=httpx2.MockTransport(handler),
    ) as http:
        client = KernelClient(agent_id=AGENT, secret=SECRET, http=http)
        with pytest.raises(APIError):
            await client.heartbeat("e1", lease=LEASE)
    assert len(captured) == 1
