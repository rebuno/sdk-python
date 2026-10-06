import asyncio
import contextlib

import pytest

from rebuno import subagent
from rebuno._kernel import DispatchLease
from rebuno.errors import Blocked, ToolError
from rebuno.execution import ExecutionContext, _reset_current, _set_current
from rebuno.types import Step, StepDecision


class FakeKernel:
    def __init__(self, *, suspended, steps=None):
        self.suspended = suspended
        self.steps = steps or {}
        self.submits = 0
        self.suspends = 0
        self.completed = []
        self.failed = []

    async def submit_step(
        self, execution_id, *, lease, kind, target, args, idempotency
    ):
        self.submits += 1
        step_id = f"step-{self.submits}"
        await asyncio.sleep(0)
        return StepDecision(decision="proceed", step_id=step_id)

    async def suspend(self, execution_id, *, lease):
        self.suspends += 1
        return self.suspended

    async def get_step(self, execution_id, step_id):
        return self.steps[step_id]

    async def complete_step(self, execution_id, step_id, *, lease, result):
        self.completed.append(step_id)

    async def fail_step(self, execution_id, step_id, *, lease, error):
        self.failed.append(step_id)


class FakeClient:
    def __init__(self):
        self.created = []

    async def create(self, agent_id, input=None, *, spawned_by):
        self.created.append((agent_id, spawned_by.execution_id, spawned_by.step_id))


@contextlib.contextmanager
def current(kernel):
    ctx = ExecutionContext(
        kernel=kernel,
        execution_id="e1",
        lease=DispatchLease("d1", 1, 120.0),
        agent_id="a",
        input=None,
    )
    token = _set_current(ctx)
    try:
        yield ctx
    finally:
        _reset_current(token)


async def test_concurrent_subagents_suspend_once_after_every_call_waits():
    k = FakeKernel(suspended=True)
    client = FakeClient()
    other_done = asyncio.Event()

    async def other():
        await asyncio.sleep(0)
        assert k.suspends == 0
        other_done.set()
        return "ok"

    with current(k) as ctx:
        results = await asyncio.gather(
            ctx.invoke_tool(
                "research", {"q": 1}, run=lambda: subagent("r", client=client)
            ),
            ctx.invoke_tool(
                "research", {"q": 2}, run=lambda: subagent("r", client=client)
            ),
            ctx.invoke_tool("lookup", {}, run=other),
            return_exceptions=True,
        )

    assert other_done.is_set()
    assert k.suspends == 1
    assert isinstance(results[0], Blocked) and isinstance(results[1], Blocked)
    assert results[2] == "ok"
    assert sorted(c[2] for c in client.created) == ["step-1", "step-2"]
    assert all(c[1] == "e1" for c in client.created)
    assert k.completed == ["step-3"] and k.failed == []
    assert isinstance(ctx.suspension, Blocked)


async def test_settled_subagent_returns_the_recorded_outcome():
    k = FakeKernel(
        suspended=False,
        steps={
            "step-1": Step(step_id="step-1", target="research", result={"a": 1}),
            "step-2": Step(
                step_id="step-2",
                target="research",
                error={"reason": "subagent_failed"},
            ),
        },
    )
    client = FakeClient()

    with current(k) as ctx:
        ok, failed = await asyncio.gather(
            ctx.invoke_tool(
                "research", {"q": 1}, run=lambda: subagent("r", client=client)
            ),
            ctx.invoke_tool(
                "research", {"q": 2}, run=lambda: subagent("r", client=client)
            ),
            return_exceptions=True,
        )

    assert ok == {"a": 1}
    assert isinstance(failed, ToolError) and "subagent_failed" in str(failed)
    assert ctx.suspension is None


async def test_subagent_outside_a_tool_body_raises():
    with current(FakeKernel(suspended=True)), pytest.raises(RuntimeError):
        await subagent("r", client=FakeClient())
