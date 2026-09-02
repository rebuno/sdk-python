import asyncio
import contextlib
import hashlib
import hmac
import json

import pytest
from httpx2 import ASGITransport, AsyncClient

from rebuno.agent import Agent
from rebuno.errors import LeaseSuperseded, ToolError

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

        return Execution(
            id=execution_id, agent_id="a", input=self._input, status="running"
        )

    async def complete_execution(self, execution_id, *, lease, output):
        self.completed = output

    async def fail_execution(self, execution_id, *, lease, error):
        self.failed = error


def build(agent, kernel):
    agent._kernel = kernel  # inject fake
    return AsyncClient(transport=ASGITransport(app=agent.app), base_url="http://test")


def _payload(
    execution_id="e1", dispatch_id="d1", attempt=1, lease_timeout=120.0
) -> dict:
    return {
        "execution_id": execution_id,
        "dispatch_id": dispatch_id,
        "dispatch_attempt": attempt,
        "lease_timeout_seconds": lease_timeout,
    }


def webhook_body(**kwargs) -> bytes:
    return json.dumps(_payload(**kwargs)).encode()


async def _process_ok(prompt: str):
    return {"answer": prompt.upper()}


async def test_invalid_signature_401():
    agent = Agent("a", secret=SECRET, base_url="http://k")
    agent.bind(_process_ok)
    async with build(agent, FakeKernel({"prompt": "hi"})) as client:
        body = webhook_body()
        r = await client.post(
            "/webhook", content=body, headers={"Rebuno-Signature": "sha256=bad"}
        )
        assert r.status_code == 401


async def test_completes_execution():
    agent = Agent("a", secret=SECRET, base_url="http://k")
    agent.bind(_process_ok)
    k = FakeKernel({"prompt": "hi"})
    async with build(agent, k) as client:
        body = webhook_body()
        r = await client.post(
            "/webhook", content=body, headers={"Rebuno-Signature": sign(body)}
        )
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
        r = await client.post(
            "/webhook", content=body, headers={"Rebuno-Signature": sign(body)}
        )
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
        r = await client.post(
            "/webhook", content=body, headers={"Rebuno-Signature": sign(body)}
        )
        assert r.status_code == 200
        await agent.join()
        assert k.failed and "boom" in k.failed


async def test_tool_failure_reason_names_the_tool():
    async def proc(prompt: str):
        raise ToolError("indeterminate", tool_id="send_email", step_id="s1")

    agent = Agent("a", secret=SECRET, base_url="http://k")
    agent.bind(proc)
    k = FakeKernel({"prompt": "hi"})
    async with build(agent, k) as client:
        body = webhook_body()
        r = await client.post(
            "/webhook", content=body, headers={"Rebuno-Signature": sign(body)}
        )
        assert r.status_code == 200
        await agent.join()
        assert k.failed == "tool_error: send_email: indeterminate"


async def test_rate_limited_fails_execution_cleanly():
    from rebuno.errors import RateLimited

    async def proc(prompt: str):
        raise RateLimited("rate_limit_exceeded")

    agent = Agent("a", secret=SECRET, base_url="http://k")
    agent.bind(proc)
    k = FakeKernel({"prompt": "hi"})
    async with build(agent, k) as client:
        body = webhook_body()
        r = await client.post(
            "/webhook", content=body, headers={"Rebuno-Signature": sign(body)}
        )
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


@pytest.mark.parametrize(
    "payload",
    [
        {"execution_id": "e1"},
        {"execution_id": "e1", "dispatch_id": "d1"},
        {"execution_id": "e1", "dispatch_id": "d1", "dispatch_attempt": 0},
        {"execution_id": "e1", "dispatch_id": "d1", "dispatch_attempt": "2"},
        {"execution_id": "e1", "dispatch_id": "d1", "dispatch_attempt": 1},
        _payload(lease_timeout="120"),
        _payload(lease_timeout=True),
        _payload(lease_timeout=0),
    ],
    ids=[
        "no-dispatch",
        "no-attempt",
        "zero-attempt",
        "attempt-not-a-number",
        "no-timeout",
        "timeout-not-a-number",
        "timeout-is-a-bool",
        "zero-timeout",
    ],
)
async def test_webhook_without_a_usable_lease_is_rejected(payload):
    """Every mutation this run makes must carry the lease it was sent under, so a
    payload that cannot produce one is unusable rather than silently degraded."""
    agent = Agent("a", secret=SECRET, base_url="http://k")
    agent.bind(_process_ok)
    async with build(agent, FakeKernel({"prompt": "hi"})) as client:
        body = json.dumps(payload).encode()
        r = await client.post(
            "/webhook", content=body, headers={"Rebuno-Signature": sign(body)}
        )
        assert r.status_code == 400


async def test_lease_reaches_the_execution_context():
    seen = {}

    async def proc(prompt: str):
        from rebuno.execution import execution

        seen["dispatch"] = (execution().dispatch_id, execution().dispatch_attempt)
        return {}

    agent = Agent("a", secret=SECRET, base_url="http://k")
    agent.bind(proc)
    async with build(agent, FakeKernel({"prompt": "hi"})) as client:
        body = webhook_body(dispatch_id="d-42", attempt=7)
        r = await client.post(
            "/webhook", content=body, headers={"Rebuno-Signature": sign(body)}
        )
        assert r.status_code == 200
        await agent.join()
    assert seen["dispatch"] == ("d-42", 7)


def _blocking_process():
    """A process that parks on its first run and returns on later ones, reporting
    which runs started and whether the first was cancelled."""
    state = {"runs": 0, "cancelled": False, "started": asyncio.Event()}

    async def proc(prompt: str):
        state["runs"] += 1
        mine = state["runs"]
        if mine == 1:
            state["started"].set()
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                state["cancelled"] = True
                raise
        return {"run": mine}

    return proc, state


async def test_a_later_attempt_supersedes_the_running_one():
    proc, state = _blocking_process()
    agent = Agent("a", secret=SECRET, base_url="http://k")
    agent.bind(proc)
    k = FakeKernel({"prompt": "hi"})
    async with build(agent, k) as client:
        first = webhook_body(attempt=1)
        r = await client.post(
            "/webhook", content=first, headers={"Rebuno-Signature": sign(first)}
        )
        assert r.status_code == 200
        await state["started"].wait()

        second = webhook_body(attempt=2)
        r = await client.post(
            "/webhook", content=second, headers={"Rebuno-Signature": sign(second)}
        )
        assert r.status_code == 200
        assert len(agent._tasks) == 1
        await agent.join()

    assert state["cancelled"]
    assert k.completed == {"run": 2}
    # The superseded run must not have written: CancelledError unwinds past the
    # handler's except Exception without failing the execution the new run owns.
    assert k.failed is None


async def test_a_stalled_handler_does_not_hold_up_its_replacement():
    """Cancellation is cooperative, so a handler stalled in a long call unwinds
    whenever it gets round to it. The attempt that replaced it holds the only live
    lease and must start now, not once its predecessor is finally gone."""
    started = {1: asyncio.Event(), 2: asyncio.Event()}
    release = asyncio.Event()
    runs = 0

    async def proc(prompt: str):
        nonlocal runs
        runs += 1
        mine = runs
        started[mine].set()
        if mine == 1:
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                await release.wait()
        return {"run": mine}

    agent = Agent("a", secret=SECRET, base_url="http://k")
    agent.bind(proc)
    async with build(agent, FakeKernel({"prompt": "hi"})) as client:
        first = webhook_body(attempt=1)
        r = await client.post(
            "/webhook", content=first, headers={"Rebuno-Signature": sign(first)}
        )
        assert r.status_code == 200
        await started[1].wait()

        second = webhook_body(attempt=2)
        r = await client.post(
            "/webhook", content=second, headers={"Rebuno-Signature": sign(second)}
        )
        assert r.status_code == 200
        await asyncio.wait_for(started[2].wait(), timeout=1)
        release.set()
        await agent.close()


@pytest.mark.parametrize(
    "attempt", [2, 1], ids=["identical-redelivery", "attempt-already-replaced"]
)
async def test_a_redelivery_the_kernel_has_moved_past_is_ignored(attempt):
    """At-least-once delivery repeats a webhook, and a reclaimed attempt can land
    after the one that replaced it. Neither may restart or cancel the run that
    owns the execution."""
    proc, state = _blocking_process()
    agent = Agent("a", secret=SECRET, base_url="http://k")
    agent.bind(proc)
    k = FakeKernel({"prompt": "hi"})
    async with build(agent, k) as client:
        live = webhook_body(attempt=2)
        r = await client.post(
            "/webhook", content=live, headers={"Rebuno-Signature": sign(live)}
        )
        assert r.status_code == 200
        await state["started"].wait()
        running = agent._tasks["e1"].task

        stale = webhook_body(attempt=attempt)
        r = await client.post(
            "/webhook", content=stale, headers={"Rebuno-Signature": sign(stale)}
        )
        assert r.status_code == 200
        assert agent._tasks["e1"].task is running
        assert state["runs"] == 1
        assert not state["cancelled"]
        await agent.close()

    assert k.completed is None
    assert k.failed is None


async def test_a_superseded_handler_does_not_fail_the_execution():
    """The kernel refuses the superseded attempt's writes. The handler unwinds on
    that refusal, leaving the execution to the attempt that replaced it."""

    class SupersedingKernel(FakeKernel):
        async def complete_execution(self, execution_id, *, lease, output):
            raise LeaseSuperseded

    async def proc(prompt: str):
        return {"answer": "done"}

    agent = Agent("a", secret=SECRET, base_url="http://k")
    agent.bind(proc)
    k = SupersedingKernel({"prompt": "hi"})
    async with build(agent, k) as client:
        body = webhook_body()
        r = await client.post(
            "/webhook", content=body, headers={"Rebuno-Signature": sign(body)}
        )
        assert r.status_code == 200
        await agent.join()
    assert k.failed is None


async def test_distinct_executions_run_concurrently():
    class MultiKernel(FakeKernel):
        def __init__(self, input):
            super().__init__(input)
            self.all: list[str] = []

        async def complete_execution(self, execution_id, *, lease, output):
            self.all.append(execution_id)

    agent = Agent("a", secret=SECRET, base_url="http://k")
    agent.bind(_process_ok)
    k = MultiKernel({"prompt": "hi"})
    async with build(agent, k) as client:
        for exec_id in ("e1", "e2"):
            body = webhook_body(execution_id=exec_id)
            r = await client.post(
                "/webhook", content=body, headers={"Rebuno-Signature": sign(body)}
            )
            assert r.status_code == 200
        await agent.join()
    assert sorted(k.all) == ["e1", "e2"]


class BlockingKernel(FakeKernel):
    """Holds every step for approval, the way a require_approval policy does."""

    async def submit_step(
        self, execution_id, *, lease, kind, target, args, idempotency
    ):
        from rebuno.types import StepDecision

        return StepDecision(decision="blocked", step_id="s1", approval_id="ap1")


async def test_swallowed_block_does_not_complete_execution():
    """Frameworks catch what a tool raises and hand it to the model, so the handler
    can return an answer for work the kernel never allowed to run. That answer must
    not be recorded as the execution's output."""
    from rebuno import tool

    @tool("send_email", idempotency="at_most_once")
    async def send_email(body: str) -> dict:
        return {"sent": True}

    async def proc(prompt: str):
        with contextlib.suppress(Exception):
            await send_email("brief")
        return {"answer": "emailed"}

    agent = Agent("a", secret=SECRET, base_url="http://k")
    agent.bind(proc)
    k = BlockingKernel({"prompt": "hi"})
    async with build(agent, k) as client:
        body = webhook_body()
        r = await client.post(
            "/webhook", content=body, headers={"Rebuno-Signature": sign(body)}
        )
        assert r.status_code == 200
        await agent.join()
        assert k.completed is None
        assert k.failed is None


async def test_swallowed_block_survives_a_later_exception():
    """After swallowing the block a framework calls the LLM again, and that refusal
    surfaces as the provider's own error rather than Blocked."""
    from rebuno import tool

    @tool("send_email", idempotency="at_most_once")
    async def send_email(body: str) -> dict:
        return {"sent": True}

    async def proc(prompt: str):
        with contextlib.suppress(Exception):
            await send_email("brief")
        raise RuntimeError("Error code: 403 - provider rejected the call")

    agent = Agent("a", secret=SECRET, base_url="http://k")
    agent.bind(proc)
    k = BlockingKernel({"prompt": "hi"})
    async with build(agent, k) as client:
        body = webhook_body()
        r = await client.post(
            "/webhook", content=body, headers={"Rebuno-Signature": sign(body)}
        )
        assert r.status_code == 200
        await agent.join()
        assert k.completed is None
        assert k.failed is None


async def test_gateway_refusal_parks_the_execution():
    """A step a gateway refused is never raised in this process; the decision only
    exists inside the provider's error."""

    async def proc(prompt: str):
        raise RuntimeError(
            "Error code: 403 - {'error': {'message': 'rebuno_refusal: execution_blocked'}}"
        )

    agent = Agent("a", secret=SECRET, base_url="http://k")
    agent.bind(proc)
    k = FakeKernel({"prompt": "hi"})
    async with build(agent, k) as client:
        body = webhook_body()
        r = await client.post(
            "/webhook", content=body, headers={"Rebuno-Signature": sign(body)}
        )
        assert r.status_code == 200
        await agent.join()
        assert k.completed is None
        assert k.failed is None


async def test_gateway_denial_fails_the_execution():
    async def proc(prompt: str):
        raise RuntimeError(
            "Error code: 403 - {'error': {'message': 'rebuno_refusal: denied'}}"
        )

    agent = Agent("a", secret=SECRET, base_url="http://k")
    agent.bind(proc)
    k = FakeKernel({"prompt": "hi"})
    async with build(agent, k) as client:
        body = webhook_body()
        r = await client.post(
            "/webhook", content=body, headers={"Rebuno-Signature": sign(body)}
        )
        assert r.status_code == 200
        await agent.join()
        assert k.completed is None
        assert k.failed and "denied" in k.failed


@pytest.mark.parametrize(
    "raise_it,expected",
    [
        (
            lambda: (_ for _ in ()).throw(ValueError("boom")),
            "agent_error: ValueError: boom",
        ),
        (
            lambda: (_ for _ in ()).throw(ToolError("nope", tool_id="send_email")),
            "tool_error: send_email: nope",
        ),
    ],
)
async def test_failure_reason_vocabulary(raise_it, expected):
    async def proc(prompt: str):
        raise_it()

    agent = Agent("a", secret=SECRET, base_url="http://k")
    agent.bind(proc)
    k = FakeKernel({"prompt": "hi"})
    async with build(agent, k) as client:
        body = webhook_body()
        r = await client.post(
            "/webhook", content=body, headers={"Rebuno-Signature": sign(body)}
        )
        assert r.status_code == 200
        await agent.join()
        assert k.failed == expected


async def test_failure_reason_for_bad_input():
    async def proc(prompt: str): ...

    agent = Agent("a", secret=SECRET, base_url="http://k")
    agent.bind(proc)
    k = FakeKernel({"wrong_field": "hi"})
    async with build(agent, k) as client:
        body = webhook_body()
        r = await client.post(
            "/webhook", content=body, headers={"Rebuno-Signature": sign(body)}
        )
        assert r.status_code == 200
        await agent.join()
        assert k.failed.startswith("input_invalid: ")


async def test_denied_llm_call_records_the_kernel_reason():
    import httpx2

    from rebuno.errors import REFUSAL_TYPE
    from rebuno.http_client import RebunoTransport

    REASON = "fs_write not allowed outside /tmp"

    class DenyingKernel(FakeKernel):
        async def submit_step(self, execution_id, **kw):
            from rebuno.types import StepDecision

            return StepDecision(decision="denied", reason=REASON)

    def provider(request):
        return httpx2.Response(200, json={"ok": True}, request=request)

    async def proc(prompt: str):
        async with httpx2.AsyncClient(
            transport=RebunoTransport(httpx2.MockTransport(provider))
        ) as c:
            r = await c.post("http://llm/v1/chat", json={"model": "m"})
            raise RuntimeError(f"Error code: 403 - {r.json()}")

    agent = Agent("a", secret=SECRET, base_url="http://k")
    agent.bind(proc)
    k = DenyingKernel({"prompt": "hi"})
    async with build(agent, k) as client:
        body = webhook_body()
        r = await client.post(
            "/webhook", content=body, headers={"Rebuno-Signature": sign(body)}
        )
        assert r.status_code == 200
        await agent.join()
        assert k.failed == f"policy_denied: {REASON}"
        assert REFUSAL_TYPE not in k.failed
