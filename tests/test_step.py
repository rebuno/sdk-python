from rebuno.execution import ExecutionContext, _reset_current, _set_current
from rebuno.step import step
from rebuno.types import StepDecision


class FakeKernel:
    def __init__(self, decision):
        self.decision = decision
        self.completed = []

    async def submit_step(self, execution_id, *, kind, target, args, idempotency, step_id):
        self.captured = dict(target=target, args=args, idempotency=idempotency)
        return self.decision

    async def complete_step(self, execution_id, step_id, *, result):
        self.completed.append(result)


async def test_step_records_local_work():
    k = FakeKernel(StepDecision(decision="proceed"))
    token = _set_current(ExecutionContext(kernel=k, execution_id="e1", agent_id="a", input=None))
    try:
        out = await step("pick_id", lambda: 42)
    finally:
        _reset_current(token)
    assert out == 42
    assert k.captured["target"] == "pick_id"
    assert k.captured["args"] == {}
    assert k.completed == [42]


async def test_step_replays():
    k = FakeKernel(StepDecision(decision="replay", result=7))
    token = _set_current(ExecutionContext(kernel=k, execution_id="e1", agent_id="a", input=None))
    try:
        out = await step("pick_id", lambda: 999)
    finally:
        _reset_current(token)
    assert out == 7


async def test_step_forwards_idempotency():
    k = FakeKernel(StepDecision(decision="proceed"))
    token = _set_current(ExecutionContext(kernel=k, execution_id="e1", agent_id="a", input=None))
    try:
        await step("send_email", lambda: "ok", idempotency="at_most_once")
    finally:
        _reset_current(token)
    assert k.captured["idempotency"] == "at_most_once"


async def test_step_defaults_idempotency_to_safe_to_retry():
    k = FakeKernel(StepDecision(decision="proceed"))
    token = _set_current(ExecutionContext(kernel=k, execution_id="e1", agent_id="a", input=None))
    try:
        await step("now", lambda: 0)
    finally:
        _reset_current(token)
    assert k.captured["idempotency"] == "safe_to_retry"


async def test_step_records_args_dict():
    k = FakeKernel(StepDecision(decision="proceed"))
    token = _set_current(ExecutionContext(kernel=k, execution_id="e1", agent_id="a", input=None))
    try:
        out = await step("pick", lambda n: n * 2, args={"n": 21})
    finally:
        _reset_current(token)
    assert out == 42
    assert k.captured["args"] == {"n": 21}


async def test_step_handles_arg_named_name():
    # A tool whose argument is literally `name` must not collide with step()'s
    # own `name` parameter now that args are passed as a dict.
    k = FakeKernel(StepDecision(decision="proceed"))
    token = _set_current(ExecutionContext(kernel=k, execution_id="e1", agent_id="a", input=None))
    try:
        out = await step("greet", lambda name: f"Hello, {name}", args={"name": "World"})
    finally:
        _reset_current(token)
    assert out == "Hello, World"
    assert k.captured["args"] == {"name": "World"}
