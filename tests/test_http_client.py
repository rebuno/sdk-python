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


async def test_streaming_passes_through_without_recording():
    kernel = StepKernel()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    token = _set_current(_ctx(kernel))
    try:
        async with _client(handler) as client:
            r = await client.post("/v1/messages", json={**REQUEST, "stream": True})
        assert r.status_code == 200
        assert kernel.submits == []  # streaming is not recorded
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
