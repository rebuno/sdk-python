import asyncio

import pytest

from rebuno._kernel import DispatchLease
from rebuno.errors import (
    Blocked,
    LeaseSuperseded,
    PolicyError,
    RateLimited,
    Terminated,
    ToolError,
)
from rebuno.execution import ExecutionContext, _reset_current, _set_current, execution
from rebuno.types import StepDecision


class FakeKernel:
    """Stands in for the kernel: assigns each submitted step an id, the way the
    real one does, so decisions carry the id the SDK must use to complete them."""

    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.completed = []
        self.failed = []
        self.submits = []

    async def submit_step(
        self, execution_id, *, lease, kind, target, args, idempotency
    ):
        self.submits.append((lease, kind, target, args))
        dec = self.decisions.pop(0)
        if not dec.step_id and dec.decision in ("proceed", "replay"):
            dec = dec.model_copy(update={"step_id": f"step-{len(self.submits)}"})
        return dec

    async def complete_step(self, execution_id, step_id, *, lease, result):
        self.completed.append((step_id, result))

    async def fail_step(self, execution_id, step_id, *, lease, error):
        self.failed.append((step_id, error))


def ctx(kernel, lease_timeout=120.0):
    return ExecutionContext(
        kernel=kernel,
        execution_id="e1",
        lease=DispatchLease("d1", 1, lease_timeout),
        agent_id="a",
        input={"x": 1},
    )


async def test_proceed_runs_body_and_completes():
    k = FakeKernel([StepDecision(decision="proceed")])
    c = ctx(k)

    async def body():
        return {"echo": "hi"}

    out = await c.invoke_tool("search", {"q": "hi"}, run=body)
    assert out == {"echo": "hi"}
    assert k.completed and k.completed[0][1] == {"echo": "hi"}


async def test_replay_returns_recorded_result_without_running():
    k = FakeKernel([StepDecision(decision="replay", result={"cached": 1})])
    ran = False

    async def body():
        nonlocal ran
        ran = True

    out = await ctx(k).invoke_tool("search", {"q": "hi"}, run=body)
    assert out == {"cached": 1}
    assert ran is False


async def test_replay_failed_raises_toolerror():
    k = FakeKernel([StepDecision(decision="replay", error={"message": "boom"})])
    with pytest.raises(ToolError):
        await ctx(k).invoke_tool("t", {}, run=None)


async def test_denied_raises_policyerror():
    k = FakeKernel([StepDecision(decision="denied", reason="nope")])
    with pytest.raises(PolicyError):
        await ctx(k).invoke_tool("t", {}, run=None)


async def test_blocked_raises_blocked():
    k = FakeKernel([StepDecision(decision="blocked", approval_id="ap1")])
    with pytest.raises(Blocked):
        await ctx(k).invoke_tool("t", {}, run=None)


async def test_rate_limited_and_terminal():
    with pytest.raises(RateLimited):
        await ctx(
            FakeKernel([StepDecision(decision="rate_limited", reason="rl")])
        ).invoke_tool("t", {}, run=None)
    with pytest.raises(Terminated):
        await ctx(
            FakeKernel([StepDecision(decision="execution_terminal")])
        ).invoke_tool("t", {}, run=None)


async def test_body_exception_reports_fail_and_reraises():
    k = FakeKernel([StepDecision(decision="proceed")])

    async def body():
        raise ValueError("kaboom")

    with pytest.raises(ToolError):
        await ctx(k).invoke_tool("t", {}, run=body)
    assert k.failed


async def test_submit_forwards_the_lease():
    k = FakeKernel([StepDecision(decision="replay", result=1)])
    await ctx(k).invoke_tool("t", {"a": 1}, run=None)
    assert k.submits[0][0] == DispatchLease("d1", 1, 120.0)


async def test_identical_calls_take_the_kernels_distinct_ids():
    k = FakeKernel([StepDecision(decision="proceed"), StepDecision(decision="proceed")])
    c = ctx(k)

    async def body():
        return 1

    await c.invoke_tool("t", {"a": 1}, run=body)
    await c.invoke_tool("t", {"a": 1}, run=body)
    assert [sid for sid, _ in k.completed] == ["step-1", "step-2"]


async def test_contextvar_proxy():
    c = ctx(FakeKernel([]))
    token = _set_current(c)
    try:
        assert execution().id == "e1"
        assert execution().input == {"x": 1}
    finally:
        _reset_current(token)


async def test_nested_blocked_propagates_without_failing_outer_step():
    k = FakeKernel(
        [
            StepDecision(decision="proceed"),  # outer step
            StepDecision(decision="blocked", approval_id="ap1"),  # nested inner step
        ]
    )
    c = ctx(k)

    async def outer_body():
        # A nested tool/step call on the same context, as happens when a
        # tool's body itself awaits another @tool or rebuno.step call.
        return await c.invoke_tool("inner", {}, run=None)

    with pytest.raises(Blocked):
        await c.invoke_tool("outer", {}, run=outer_body)
    assert k.failed == []
    assert k.completed == []


async def test_nested_rate_limited_propagates_without_failing_outer_step():
    k = FakeKernel(
        [
            StepDecision(decision="proceed"),
            StepDecision(decision="rate_limited", reason="rl"),
        ]
    )
    c = ctx(k)

    async def outer_body():
        return await c.invoke_tool("inner", {}, run=None)

    with pytest.raises(RateLimited):
        await c.invoke_tool("outer", {}, run=outer_body)
    assert k.failed == []


async def test_fail_step_failure_does_not_mask_original_exception():
    class FlakyKernel(FakeKernel):
        async def fail_step(self, execution_id, step_id, *, lease, error):
            raise RuntimeError("network blip")

    k = FlakyKernel([StepDecision(decision="proceed")])

    async def body():
        raise ValueError("kaboom")

    with pytest.raises(ToolError) as exc_info:
        await ctx(k).invoke_tool("t", {}, run=body)
    assert "kaboom" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, ValueError)


async def test_kernel_calls_from_a_second_loop_run_on_the_owner_loop():
    """A framework that runs tools on its own thread awaits them on a second event
    loop. The kernel client's connections belong to the loop the context was
    created on, so the round-trips are handed back to it."""

    class LoopRecordingKernel(FakeKernel):
        def __init__(self, decisions):
            super().__init__(decisions)
            self.loops = []

        async def submit_step(self, execution_id, **kw):
            self.loops.append(asyncio.get_running_loop())
            return await super().submit_step(execution_id, **kw)

        async def complete_step(self, execution_id, step_id, *, lease, result):
            self.loops.append(asyncio.get_running_loop())
            await super().complete_step(
                execution_id, step_id, lease=lease, result=result
            )

    k = LoopRecordingKernel([StepDecision(decision="proceed")])
    owner = asyncio.get_running_loop()
    c = ctx(k)
    body_loops = []

    async def body():
        body_loops.append(asyncio.get_running_loop())
        return {"echo": "hi"}

    out = await asyncio.to_thread(
        lambda: asyncio.run(c.invoke_tool("search", {"q": "hi"}, run=body))
    )
    assert out == {"echo": "hi"}
    assert k.loops == [owner, owner]
    assert body_loops[0] is not owner


async def test_a_lost_lease_stops_the_handler():
    """A stalled handler whose dispatch was reclaimed learns of it from its own
    heartbeat, and stops there rather than working on beside its replacement."""

    class SupersedingKernel(FakeKernel):
        async def heartbeat(self, execution_id, *, lease):
            raise LeaseSuperseded

    c = ctx(SupersedingKernel([]), lease_timeout=0.03)
    finished = False

    async def work():
        nonlocal finished
        async with c.lease():
            await asyncio.sleep(5)
            finished = True

    with pytest.raises(LeaseSuperseded):
        await asyncio.create_task(work())
    assert not finished


async def test_a_lost_lease_inside_a_tool_body_is_not_a_tool_failure():
    """An LLM call inside the body reaches the kernel too, so the refusal can
    surface from user code. It unwinds instead of being recorded as the tool
    failing, which the replacing attempt would then have to replay."""
    k = FakeKernel([StepDecision(decision="proceed")])

    async def body():
        raise LeaseSuperseded

    with pytest.raises(LeaseSuperseded):
        await ctx(k).invoke_tool("t", {}, run=body)
    assert k.failed == []
