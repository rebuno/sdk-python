from conftest import FakeKernel, install_context

from rebuno.execution import _reset_current
from rebuno.step import step
from rebuno.types import StepDecision


async def test_step_records_local_work():
    k = FakeKernel(StepDecision(decision="proceed"))
    token = install_context(k)
    try:
        out = await step("pick_id", lambda: 42)
    finally:
        _reset_current(token)
    assert out == 42
    assert k.captured["target"] == "pick_id"
    assert k.captured["args"] == {}
    assert k.captured["kind"] == "local"
    assert k.completed == [42]


async def test_step_replays():
    k = FakeKernel(StepDecision(decision="replay", result=7))
    token = install_context(k)
    try:
        out = await step("pick_id", lambda: 999)
    finally:
        _reset_current(token)
    assert out == 7


async def test_step_forwards_idempotency():
    k = FakeKernel(StepDecision(decision="proceed"))
    token = install_context(k)
    try:
        await step("send_email", lambda: "ok", idempotency="at_most_once")
    finally:
        _reset_current(token)
    assert k.captured["idempotency"] == "at_most_once"


async def test_step_records_args_dict():
    k = FakeKernel(StepDecision(decision="proceed"))
    token = install_context(k)
    try:
        out = await step("pick", lambda n: n * 2, args={"n": 21})
    finally:
        _reset_current(token)
    assert out == 42
    assert k.captured["args"] == {"n": 21}
