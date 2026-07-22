from typing import Any

import httpx
import pytest

from rebuno.errors import PolicyError
from rebuno.execution import ExecutionContext, _reset_current, _set_current
from rebuno.http_client import RebunoTransport
from rebuno.types import StepDecision


class StepKernel:
    """Minimal kernel: proceed for new step ids, replay for completed ones."""

    def __init__(self) -> None:
        self.steps: dict[str, Any] = {}
        self.submits: list[tuple[str, str, str]] = []  # (kind, target, step_id)
        self.completed: list[str] = []
        self.deltas: list[tuple[str, int, str]] = []  # (step_id, seq, data)

    async def submit_step(self, execution_id, *, kind, target, args, idempotency, step_id):
        self.submits.append((kind, target, step_id))
        if step_id in self.steps:
            return StepDecision(decision="replay", result=self.steps[step_id])
        return StepDecision(decision="proceed")

    async def complete_step(self, execution_id, step_id, *, result):
        self.steps[step_id] = result
        self.completed.append(step_id)

    async def fail_step(self, execution_id, step_id, *, error):
        pass

    async def heartbeat(self, execution_id):
        pass

    async def stream_delta(self, execution_id, step_id, *, seq, data):
        self.deltas.append((step_id, seq, data))


def _client(handler) -> httpx.AsyncClient:
    transport = RebunoTransport(httpx.MockTransport(handler))
    return httpx.AsyncClient(transport=transport, base_url="https://api.test")


def _ctx(kernel) -> ExecutionContext:
    return ExecutionContext(kernel=kernel, execution_id="e1", agent_id="a", input={})


REQUEST = {"model": "claude", "messages": [{"role": "user", "content": "hi"}]}


async def test_llm_call_forwards_then_replays_on_resume():
    kernel = StepKernel()
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"content": "hello", "n": calls["n"]})

    # First dispatch: forwards to the provider and records an llm_call step.
    token = _set_current(_ctx(kernel))
    try:
        async with _client(handler) as client:
            r1 = await client.post("/v1/messages", json=REQUEST)
        assert r1.status_code == 200
        assert r1.json()["content"] == "hello"
        assert calls["n"] == 1
        assert kernel.submits[0][0] == "llm_call"
        assert kernel.submits[0][1] == "claude"
        assert kernel.completed
    finally:
        _reset_current(token)

    # Resume (fresh context → occurrence resets): same request replays, no provider call.
    token = _set_current(_ctx(kernel))
    try:
        async with _client(handler) as client:
            r2 = await client.post("/v1/messages", json=REQUEST)
        assert r2.status_code == 200
        assert r2.json()["content"] == "hello"
        assert calls["n"] == 1  # provider was NOT called again
    finally:
        _reset_current(token)


async def test_denied_propagates_policy_error():
    class DenyKernel(StepKernel):
        async def submit_step(self, execution_id, *, kind, target, args, idempotency, step_id):
            return StepDecision(decision="denied", reason="model not allowed")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    token = _set_current(_ctx(DenyKernel()))
    try:
        async with _client(handler) as client:
            with pytest.raises(PolicyError):
                await client.post("/v1/messages", json=REQUEST)
    finally:
        _reset_current(token)


# A streamed provider response: a >2KB payload so the delta batcher emits
# multiple deltas, exercising size-based flushing and monotonic seq numbering.
SSE = b'data: {"delta":"' + b"x" * 5000 + b'"}\n\ndata: [DONE]\n\n'


def _sse_handler(calls: dict[str, int]):
    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1

        async def body():
            for i in range(0, len(SSE), 512):  # deliver in network-sized chunks
                yield SSE[i : i + 512]

        return httpx.Response(200, content=body(), headers={"content-type": "text/event-stream"})

    return handler


async def _drain_stream(client: httpx.AsyncClient, req: dict) -> bytes:
    got = b""
    async with client.stream("POST", "/v1/messages", json=req) as r:
        assert r.status_code == 200
        async for chunk in r.aiter_raw():
            got += chunk
    return got


async def test_streaming_tees_records_then_replays_as_stream():
    kernel = StepKernel()
    calls = {"n": 0}
    handler = _sse_handler(calls)
    req = {**REQUEST, "stream": True}

    # First run: tee bytes to the caller live, publish deltas, record the whole.
    token = _set_current(_ctx(kernel))
    try:
        async with _client(handler) as client:
            got = await _drain_stream(client, req)
        assert got == SSE  # caller received the full stream
        assert calls["n"] == 1
        assert kernel.submits[0][0] == "llm_call"
        assert kernel.completed  # the assembled whole was recorded
        # Live deltas reassemble to the full body with monotonic seqs from 0.
        assert len(kernel.deltas) >= 2  # size-based flush produced several
        assert "".join(d[2] for d in kernel.deltas) == SSE.decode()
        assert [d[1] for d in kernel.deltas] == list(range(len(kernel.deltas)))
    finally:
        _reset_current(token)

    # Resume: replay the recorded whole as a stream — no provider call, no deltas.
    n_deltas = len(kernel.deltas)
    token = _set_current(_ctx(kernel))
    try:
        async with _client(handler) as client:
            replayed = await _drain_stream(client, req)
        assert replayed == SSE
        assert calls["n"] == 1  # provider was NOT called again
        assert len(kernel.deltas) == n_deltas  # replay publishes nothing
    finally:
        _reset_current(token)

async def test_streaming_recorded_when_consumer_stops_at_done_without_draining():
    # A consumer that stops reading early and closes the response without draining
    # to EOF must still get the step recorded, not left executing.
    kernel = StepKernel()
    calls = {"n": 0}
    token = _set_current(_ctx(kernel))
    try:
        async with (
            _client(_sse_handler(calls)) as client,
            client.stream("POST", "/v1/messages", json={**REQUEST, "stream": True}) as r,
        ):
            got = b""
            async for chunk in r.aiter_raw():
                got += chunk
                if b"[DONE]" in got:
                    break  # stop early — do not pull to EOF
        assert kernel.completed  # recorded on close, not left executing
        assert "".join(d[2] for d in kernel.deltas) == SSE.decode()  # all bytes still teed
    finally:
        _reset_current(token)


async def test_streaming_midstream_error_not_recorded_as_success():
    # A stream that dies mid-flight must not be recorded as a succeeded step.
    kernel = StepKernel()

    def handler(request: httpx.Request) -> httpx.Response:
        async def body():
            yield SSE[:512]
            raise RuntimeError("connection dropped mid-stream")

        return httpx.Response(200, content=body(), headers={"content-type": "text/event-stream"})

    token = _set_current(_ctx(kernel))
    try:
        async with _client(handler) as client:
            with pytest.raises(RuntimeError, match="connection dropped"):
                await _drain_stream(client, {**REQUEST, "stream": True})
        assert kernel.completed == []  # failed the step instead of recording a partial
    finally:
        _reset_current(token)


async def test_streaming_error_status_recorded_like_non_stream():
    kernel = StepKernel()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "rate limited"})

    token = _set_current(_ctx(kernel))
    try:
        async with _client(handler) as client:
            r = await client.post("/v1/messages", json={**REQUEST, "stream": True})
        assert r.status_code == 429
        assert kernel.completed  # the error response was recorded as the result
        assert kernel.deltas == []  # nothing to tee on an error
    finally:
        _reset_current(token)


async def test_passthrough_without_active_context():
    kernel = StepKernel()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    async with _client(handler) as client:
        r = await client.post("/v1/messages", json=REQUEST)
    assert r.status_code == 200
    assert kernel.submits == []  # no execution context → nothing recorded
