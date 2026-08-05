import asyncio
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


async def test_webhook_without_dispatch_id_is_rejected():
    """Every effect this run submits must carry the dispatch it was sent under, so
    a payload missing one is unusable rather than silently degraded."""
    agent = Agent("a", secret=SECRET, base_url="http://k")
    agent.bind(_process_ok)
    async with build(agent, FakeKernel({"prompt": "hi"})) as client:
        body = json.dumps({"execution_id": "e1"}).encode()
        r = await client.post("/webhook", content=body, headers={"Rebuno-Signature": sign(body)})
        assert r.status_code == 400


async def test_dispatch_id_reaches_the_execution_context():
    seen = {}

    async def proc(prompt: str):
        from rebuno.execution import execution

        seen["dispatch_id"] = execution.dispatch_id
        return {}

    agent = Agent("a", secret=SECRET, base_url="http://k")
    agent.bind(proc)
    async with build(agent, FakeKernel({"prompt": "hi"})) as client:
        body = webhook_body(dispatch_id="d-42")
        r = await client.post("/webhook", content=body, headers={"Rebuno-Signature": sign(body)})
        assert r.status_code == 200
        await agent.join()
    assert seen["dispatch_id"] == "d-42"


async def test_redelivery_supersedes_the_previous_run():
    first_started = asyncio.Event()
    runs = 0
    cancelled = False

    async def proc(prompt: str):
        nonlocal runs, cancelled
        runs += 1
        mine = runs
        if mine == 1:
            first_started.set()
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                cancelled = True
                raise
        return {"run": mine}

    agent = Agent("a", secret=SECRET, base_url="http://k")
    agent.bind(proc)
    k = FakeKernel({"prompt": "hi"})
    async with build(agent, k) as client:
        body = webhook_body()
        r = await client.post("/webhook", content=body, headers={"Rebuno-Signature": sign(body)})
        assert r.status_code == 200
        await first_started.wait()

        r = await client.post("/webhook", content=body, headers={"Rebuno-Signature": sign(body)})
        assert r.status_code == 200
        assert len(agent._tasks) == 1
        await agent.join()

    assert cancelled
    assert k.completed == {"run": 2}
    # The superseded run must not have written: CancelledError unwinds past the
    # handler's except Exception without failing the execution the new run owns.
    assert k.failed is None


async def test_distinct_executions_run_concurrently():
    class MultiKernel(FakeKernel):
        def __init__(self, input):
            super().__init__(input)
            self.all: list[str] = []

        async def complete_execution(self, execution_id, *, output):
            self.all.append(execution_id)

    agent = Agent("a", secret=SECRET, base_url="http://k")
    agent.bind(_process_ok)
    k = MultiKernel({"prompt": "hi"})
    async with build(agent, k) as client:
        for exec_id in ("e1", "e2"):
            body = webhook_body(execution_id=exec_id)
            r = await client.post("/webhook", content=body, headers={"Rebuno-Signature": sign(body)})
            assert r.status_code == 200
        await agent.join()
    assert sorted(k.all) == ["e1", "e2"]
