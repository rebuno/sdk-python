"""ExecutionState: invoke_tool / wait_signal / complete / fail flows."""

from __future__ import annotations

import asyncio

import pytest
from conftest import make_claim

from rebuno._internal.correlation import CorrelationMap
from rebuno.errors import PolicyError, ToolError
from rebuno.execution import (
    ExecutionState,
    _reset_current,
    _set_current,
)
from rebuno.execution import (
    execution as proxy,
)
from rebuno.types import IntentResult


class _RecordingClient:
    """Capture calls; configurable intent + step result responses."""

    def __init__(self, *, intent: IntentResult | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._intent = intent or IntentResult(accepted=True, step_id="step-1")

    async def submit_intent(self, **kwargs):
        self.calls.append(("submit_intent", kwargs))
        return self._intent

    async def report_step_result(self, **kwargs):
        self.calls.append(("report_step_result", kwargs))


def _state(client, *, claim=None) -> ExecutionState:
    return ExecutionState(client, claim or make_claim(), CorrelationMap(), wait_timeout=2.0)


async def test_invoke_local_tool_executes_body_and_reports_success():
    client = _RecordingClient()
    state = _state(client)

    async def body(a: int, b: int) -> int:
        return a + b

    result = await state.invoke_tool("test.add", {"a": 2, "b": 3}, local_runner=body)
    assert result == 5

    methods = [name for name, _ in client.calls]
    assert methods == ["submit_intent", "report_step_result"]
    intent_call = client.calls[0][1]
    assert intent_call["intent_type"] == "invoke_tool"
    assert intent_call["tool_id"] == "test.add"
    assert intent_call["remote"] is False
    step_call = client.calls[1][1]
    assert step_call["success"] is True
    assert step_call["data"] == 5


async def test_invoke_local_tool_failure_reports_step_failure_and_raises_tool_error():
    client = _RecordingClient()
    state = _state(client)

    async def body() -> None:
        raise ValueError("boom")

    with pytest.raises(ToolError) as exc:
        await state.invoke_tool("test.bad", {}, local_runner=body)
    assert "boom" in str(exc.value)

    step_call = client.calls[1][1]
    assert step_call["success"] is False
    assert "boom" in step_call["error"]


async def test_invoke_denied_intent_raises_policy_error():
    client = _RecordingClient(intent=IntentResult(accepted=False, error="forbidden"))
    state = _state(client)

    async def body() -> int:
        return 1

    with pytest.raises(PolicyError, match="forbidden"):
        await state.invoke_tool("test.denied", {}, local_runner=body)
    # only the intent was submitted; no step result
    assert [name for name, _ in client.calls] == ["submit_intent"]


async def test_invoke_remote_waits_on_correlation_result():
    client = _RecordingClient()
    state = _state(client)

    async def go():
        return await state.invoke_tool("compute.heavy", {"x": 1})

    waiter = asyncio.create_task(go())
    await asyncio.sleep(0)  # let task subscribe

    state._correlation.resolve("result", "step-1", {"status": "succeeded", "result": 99})
    assert (await waiter) == 99

    intent_call = client.calls[0][1]
    assert intent_call["remote"] is True


async def test_invoke_remote_failed_result_raises_tool_error():
    client = _RecordingClient()
    state = _state(client)

    async def go():
        return await state.invoke_tool("compute.heavy", {})

    waiter = asyncio.create_task(go())
    await asyncio.sleep(0)
    state._correlation.resolve("result", "step-1", {"status": "failed", "error": "runner died"})

    with pytest.raises(ToolError, match="runner died"):
        await waiter


async def test_invoke_remote_timeout_raises_rebuno_error():
    from rebuno.errors import RebunoError

    client = _RecordingClient()
    state = ExecutionState(client, make_claim(), CorrelationMap(), wait_timeout=0.05)

    with pytest.raises(RebunoError, match="Timed out"):
        await state.invoke_tool("compute.slow", {})


async def test_invoke_pending_approval_waits_then_proceeds():
    client = _RecordingClient(
        intent=IntentResult(accepted=True, step_id="step-1", pending_approval=True),
    )
    state = _state(client)

    async def body() -> str:
        return "ran"

    async def go():
        return await state.invoke_tool("test.gated", {}, local_runner=body)

    task = asyncio.create_task(go())
    await asyncio.sleep(0)
    state._correlation.resolve("approval", "step-1", {"approved": True})
    assert (await task) == "ran"


async def test_invoke_pending_approval_denied_raises_policy_error():
    client = _RecordingClient(
        intent=IntentResult(accepted=True, step_id="step-1", pending_approval=True),
    )
    state = _state(client)

    async def body() -> None: ...

    async def go():
        await state.invoke_tool("test.gated", {}, local_runner=body)

    task = asyncio.create_task(go())
    await asyncio.sleep(0)
    state._correlation.resolve("approval", "step-1", {"approved": False})

    with pytest.raises(PolicyError, match="denied by human approval"):
        await task


async def test_wait_signal_submits_wait_intent_and_returns_payload():
    client = _RecordingClient()
    state = _state(client)

    async def go():
        return await state.wait_signal("approval")

    task = asyncio.create_task(go())
    await asyncio.sleep(0)
    state._correlation.resolve("signal", "approval", {"answer": 42})

    assert (await task) == {"answer": 42}
    assert client.calls[0][1]["intent_type"] == "wait"
    assert client.calls[0][1]["signal_type"] == "approval"


async def test_wait_signal_denied_intent_raises():
    client = _RecordingClient(intent=IntentResult(accepted=False, error="cannot wait"))
    state = _state(client)
    with pytest.raises(PolicyError, match="cannot wait"):
        await state.wait_signal("approval")


async def test_complete_submits_intent():
    client = _RecordingClient()
    state = _state(client)
    await state.complete({"out": 1})
    assert client.calls[0][1]["intent_type"] == "complete"
    assert client.calls[0][1]["output"] == {"out": 1}


async def test_fail_submits_intent():
    client = _RecordingClient()
    state = _state(client)
    await state.fail("oops")
    assert client.calls[0][1]["intent_type"] == "fail"
    assert client.calls[0][1]["error"] == "oops"


async def test_proxy_resolves_to_state_attributes():
    client = _RecordingClient()
    state = _state(client, claim=make_claim(execution_id="abc", session_id="xyz"))

    token = _set_current(state)
    try:
        assert proxy.id == "abc"
        assert proxy.session_id == "xyz"
        assert proxy.input == {"prompt": "hello"}
    finally:
        _reset_current(token)


async def test_proxy_outside_context_raises():
    with pytest.raises(RuntimeError, match="outside an active agent execution"):
        _ = proxy.id
