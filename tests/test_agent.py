import hashlib
import hmac
import json

import pytest
from httpx import ASGITransport, AsyncClient

from rebuno.agent import Agent

SECRET = "dev-secret"


def sign(body: bytes) -> str:
    return "sha256=" + hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()


class FakeKernel:
    def __init__(self, input):
        self._input = input
        self.completed = None
        self.failed = None

    async def get_execution(self, execution_id):
        from rebuno.types import Execution

        return Execution(id=execution_id, agent_id="a", input=self._input, status="running")

    async def complete_execution(self, execution_id, *, output):
        self.completed = output

    async def fail_execution(self, execution_id, *, error):
        self.failed = error


def build(agent, kernel):
    agent._kernel = kernel  # inject fake
    return AsyncClient(transport=ASGITransport(app=agent.app), base_url="http://test")


def webhook_body(execution_id="e1", dispatch_id="d1") -> bytes:
    return json.dumps({"execution_id": execution_id, "dispatch_id": dispatch_id}).encode()


async def _process_ok(prompt: str):
    return {"answer": prompt.upper()}


async def test_invalid_signature_401():
    agent = Agent("a", secret=SECRET, base_url="http://k")
    agent.bind(_process_ok)
    async with build(agent, FakeKernel({"prompt": "hi"})) as client:
        body = webhook_body()
        r = await client.post("/webhook", content=body, headers={"Rebuno-Signature": "sha256=bad"})
        assert r.status_code == 401


async def test_completes_execution():
    agent = Agent("a", secret=SECRET, base_url="http://k")
    agent.bind(_process_ok)
    k = FakeKernel({"prompt": "hi"})
    async with build(agent, k) as client:
        body = webhook_body()
        r = await client.post("/webhook", content=body, headers={"Rebuno-Signature": sign(body)})
        assert r.status_code == 200
        await agent.join()
        assert k.completed == {"answer": "HI"}


async def test_blocked_returns_200_without_complete():
    from rebuno.errors import Blocked

    async def proc(prompt: str):
        raise Blocked("ap1")

    agent = Agent("a", secret=SECRET, base_url="http://k")
    agent.bind(proc)
    k = FakeKernel({"prompt": "hi"})
    async with build(agent, k) as client:
        body = webhook_body()
        r = await client.post("/webhook", content=body, headers={"Rebuno-Signature": sign(body)})
        assert r.status_code == 200
        await agent.join()
        assert k.completed is None


async def test_process_exception_fails_execution():
    async def proc(prompt: str):
        raise ValueError("boom")

    agent = Agent("a", secret=SECRET, base_url="http://k")
    agent.bind(proc)
    k = FakeKernel({"prompt": "hi"})
    async with build(agent, k) as client:
        body = webhook_body()
        r = await client.post("/webhook", content=body, headers={"Rebuno-Signature": sign(body)})
        assert r.status_code == 200
        await agent.join()
        assert k.failed and "boom" in k.failed


async def test_rate_limited_fails_execution_cleanly():
    from rebuno.errors import RateLimited

    async def proc(prompt: str):
        raise RateLimited("rate_limit_exceeded")

    agent = Agent("a", secret=SECRET, base_url="http://k")
    agent.bind(proc)
    k = FakeKernel({"prompt": "hi"})
    async with build(agent, k) as client:
        body = webhook_body()
        r = await client.post("/webhook", content=body, headers={"Rebuno-Signature": sign(body)})
        assert r.status_code == 200
        await agent.join()
        assert k.failed and "rate_limit_exceeded" in k.failed


async def test_step_id_mismatch_fails_execution_cleanly():
    from rebuno.errors import StepIDMismatch

    async def proc(prompt: str):
        raise StepIDMismatch("step id divergence", code="step_id_divergence", status_code=409)

    agent = Agent("a", secret=SECRET, base_url="http://k")
    agent.bind(proc)
    k = FakeKernel({"prompt": "hi"})
    async with build(agent, k) as client:
        body = webhook_body()
        r = await client.post("/webhook", content=body, headers={"Rebuno-Signature": sign(body)})
        assert r.status_code == 200
        await agent.join()
        assert k.failed and "step id divergence" in k.failed


def test_empty_secret_raises(monkeypatch):
    monkeypatch.delenv("REBUNO_AGENT_SECRET", raising=False)
    with pytest.raises(ValueError):
        Agent("a", secret="", base_url="http://k")
    with pytest.raises(ValueError):
        Agent("a", base_url="http://k")


def test_default_kernel_timeout_applied():
    agent = Agent("a", secret=SECRET, base_url="http://k")
    assert agent._http.timeout.connect == 35.0


def test_custom_kernel_timeout_applied():
    agent = Agent("a", secret=SECRET, base_url="http://k", kernel_timeout=5.0)
    assert agent._http.timeout.connect == 5.0
