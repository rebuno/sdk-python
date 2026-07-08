import pytest

from rebuno.errors import Blocked, PolicyError, RateLimited, Terminated, ToolError
from rebuno.execution import ExecutionContext, _reset_current, _set_current, execution
from rebuno.identity import args_hash, compute_step_id
from rebuno.types import Step, StepDecision


class FakeKernel:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.completed = []
        self.failed = []

    async def submit_step(self, execution_id, *, kind, target, args, idempotency, step_id):
        return self.decisions.pop(0)

    async def complete_step(self, execution_id, step_id, *, result):
        self.completed.append((step_id, result))

    async def fail_step(self, execution_id, step_id, *, error):
        self.failed.append((step_id, error))


def ctx(kernel):
    return ExecutionContext(kernel=kernel, execution_id="e1", agent_id="a", input={"x": 1})


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
    with pytest.raises(Blocked) as e:
        await ctx(k).invoke_tool("t", {}, run=None)
    assert e.value.approval_id == "ap1"


async def test_rate_limited_and_terminal():
    with pytest.raises(RateLimited):
        await ctx(FakeKernel([StepDecision(decision="rate_limited", reason="rl")])).invoke_tool("t", {}, run=None)
    with pytest.raises(Terminated):
        await ctx(FakeKernel([StepDecision(decision="execution_terminal")])).invoke_tool("t", {}, run=None)


async def test_body_exception_reports_fail_and_reraises():
    k = FakeKernel([StepDecision(decision="proceed")])

    async def body():
        raise ValueError("kaboom")

    with pytest.raises(ToolError):
        await ctx(k).invoke_tool("t", {}, run=body)
    assert k.failed


async def test_occurrence_increments_for_identical_calls():
    k = FakeKernel([StepDecision(decision="replay", result=1), StepDecision(decision="replay", result=2)])
    c = ctx(k)
    seen = []

    async def fake_submit(execution_id, *, kind, target, args, idempotency, step_id):
        seen.append(step_id)
        return k.decisions.pop(0)

    k.submit_step = fake_submit
    await c.invoke_tool("t", {"a": 1}, run=None)
    await c.invoke_tool("t", {"a": 1}, run=None)
    assert seen[0] != seen[1]  # different occurrence -> different step id


class HydratingKernel(FakeKernel):
    """FakeKernel that also serves a terminal-step list for hydration and tracks
    whether submit_step was reached."""

    def __init__(self, decisions, terminal_steps):
        super().__init__(decisions)
        self.terminal_steps = terminal_steps
        self.submits = 0

    async def list_terminal_steps(self, execution_id):
        return list(self.terminal_steps)

    async def submit_step(self, execution_id, *, kind, target, args, idempotency, step_id):
        self.submits += 1
        return self.decisions.pop(0)


def _step_id_for(c, kind, target, args, occ=0):
    return compute_step_id(c.id, kind, target, args_hash(args), occ)


async def test_hydrated_replay_serves_from_map_without_submit():
    args = {"q": "hi"}
    # No decisions queued: if submit_step is reached, .pop(0) raises — proving
    # the replay came from the hydrated map, not the kernel.
    k = HydratingKernel([], [])
    c = ctx(k)
    sid = _step_id_for(c, "tool_call", "search", args)
    k.terminal_steps = [Step(step_id=sid, status="succeeded", result={"cached": 1})]
    await c.hydrate()

    ran = False

    async def body():
        nonlocal ran
        ran = True

    out = await c.invoke_tool("search", args, run=body)
    assert out == {"cached": 1}
    assert ran is False
    assert k.submits == 0  # served entirely from the hydrated map


async def test_hydrated_miss_falls_through_to_submit():
    k = HydratingKernel([StepDecision(decision="proceed")], [])  # empty map -> miss
    c = ctx(k)
    await c.hydrate()

    async def body():
        return {"fresh": True}

    out = await c.invoke_tool("search", {"q": "hi"}, run=body)
    assert out == {"fresh": True}
    assert k.submits == 1  # miss went to the kernel
    assert k.completed


async def test_hydrated_denied_step_raises_policyerror_without_submit():
    args = {"x": 1}
    k = HydratingKernel([], [])
    c = ctx(k)
    sid = _step_id_for(c, "tool_call", "danger", args)
    k.terminal_steps = [Step(step_id=sid, status="denied")]
    await c.hydrate()
    with pytest.raises(PolicyError):
        await c.invoke_tool("danger", args, run=None)
    assert k.submits == 0


async def test_hydrate_failure_falls_back_to_per_step():
    class BrokenHydrate(FakeKernel):
        async def list_terminal_steps(self, execution_id):
            raise RuntimeError("no such endpoint")

    k = BrokenHydrate([StepDecision(decision="replay", result={"ok": 1})])
    c = ctx(k)
    await c.hydrate()  # swallows the error
    assert c._replay is None  # un-hydrated -> per-step path
    out = await c.invoke_tool("t", {}, run=None)
    assert out == {"ok": 1}


async def test_contextvar_proxy():
    c = ctx(FakeKernel([]))
    token = _set_current(c)
    try:
        assert execution.id == "e1"
        assert execution.input == {"x": 1}
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
        async def fail_step(self, execution_id, step_id, *, error):
            raise RuntimeError("network blip")

    k = FlakyKernel([StepDecision(decision="proceed")])

    async def body():
        raise ValueError("kaboom")

    with pytest.raises(ToolError) as exc_info:
        await ctx(k).invoke_tool("t", {}, run=body)
    assert "kaboom" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, ValueError)
