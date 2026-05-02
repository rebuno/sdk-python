"""Agent: handler binding, dispatch, error handling at the execution level."""

from __future__ import annotations

from conftest import make_claim

from rebuno import Agent
from rebuno.types import IntentResult


class _SpyClient:
    """Replaces Client on a real Agent. Captures intent + report calls.
    Returns IntentResult.accepted=True for everything."""

    def __init__(self):
        self.intents: list[dict] = []
        self.steps: list[dict] = []

    async def submit_intent(self, **kwargs):
        self.intents.append(kwargs)
        return IntentResult(accepted=True, step_id=f"step-{len(self.intents)}")

    async def report_step_result(self, **kwargs):
        self.steps.append(kwargs)

    async def close(self):
        pass


def _agent_with_spy() -> tuple[Agent, _SpyClient]:
    agent = Agent("test", kernel_url="http://test", api_key="k")
    spy = _SpyClient()
    agent._client = spy  # type: ignore[assignment]
    return agent, spy


async def test_handler_kwargs_passed_and_output_completes():
    agent, spy = _agent_with_spy()

    async def handler(prompt: str, repo_url: str = "x") -> dict:
        return {"got": prompt, "repo": repo_url}

    from rebuno._internal.inputs import InputBinder

    agent._handler = handler
    agent._binder = InputBinder(handler)

    await agent._handle_execution(make_claim(input={"prompt": "hi"}))

    completes = [i for i in spy.intents if i["intent_type"] == "complete"]
    assert len(completes) == 1
    assert completes[0]["output"] == {"got": "hi", "repo": "x"}


async def test_handler_missing_required_input_fails_execution_with_clear_error():
    agent, spy = _agent_with_spy()

    async def handler(prompt: str): ...

    from rebuno._internal.inputs import InputBinder

    agent._handler = handler
    agent._binder = InputBinder(handler)

    await agent._handle_execution(make_claim(input={}))  # missing prompt

    fails = [i for i in spy.intents if i["intent_type"] == "fail"]
    assert len(fails) == 1
    assert "missing required input fields" in fails[0]["error"]
    assert "prompt" in fails[0]["error"]


async def test_handler_raising_exception_fails_execution():
    agent, spy = _agent_with_spy()

    async def handler(prompt: str):
        raise RuntimeError("kaboom")

    from rebuno._internal.inputs import InputBinder

    agent._handler = handler
    agent._binder = InputBinder(handler)

    await agent._handle_execution(make_claim(input={"prompt": "hi"}))

    fails = [i for i in spy.intents if i["intent_type"] == "fail"]
    assert len(fails) == 1
    assert "kaboom" in fails[0]["error"]


async def test_execution_context_is_set_inside_handler_and_cleared_after():
    from rebuno.execution import _get_current

    agent, spy = _agent_with_spy()
    seen = {}

    async def handler(prompt: str) -> str:
        state = _get_current()
        seen["state"] = state
        seen["id"] = state.id  # type: ignore[union-attr]
        return prompt

    from rebuno._internal.inputs import InputBinder

    agent._handler = handler
    agent._binder = InputBinder(handler)

    claim = make_claim(execution_id="e-42", input={"prompt": "yo"})
    await agent._handle_execution(claim)

    assert seen["id"] == "e-42"
    assert _get_current() is None


async def test_dispatch_tool_result_resolves_correlation_future():
    import json as _json

    from rebuno._internal.correlation import CorrelationMap
    from rebuno._internal.sse import SSEEvent

    agent, _ = _agent_with_spy()
    cm = CorrelationMap()
    agent._exec_correlation["e-1"] = cm
    fut = cm.future("result", "step-1")

    await agent._dispatch(
        SSEEvent(
            type="tool.result",
            data=_json.dumps(
                {
                    "execution_id": "e-1",
                    "step_id": "step-1",
                    "status": "succeeded",
                    "result": 42,
                }
            ),
        )
    )
    assert fut.done()
    payload = await fut
    assert payload["result"] == 42


async def test_dispatch_signal_received_resolves_correlation():
    import json as _json

    from rebuno._internal.correlation import CorrelationMap
    from rebuno._internal.sse import SSEEvent

    agent, _ = _agent_with_spy()
    cm = CorrelationMap()
    agent._exec_correlation["e-1"] = cm
    fut = cm.future("signal", "approval")

    await agent._dispatch(
        SSEEvent(
            type="signal.received",
            data=_json.dumps(
                {
                    "execution_id": "e-1",
                    "signal_type": "approval",
                    "payload": {"ok": True},
                }
            ),
        )
    )
    assert (await fut) == {"ok": True}


async def test_dispatch_unknown_execution_id_is_silently_ignored():
    """SSE for an execution we no longer track must not crash."""
    import json as _json

    from rebuno._internal.sse import SSEEvent

    agent, _ = _agent_with_spy()
    # No correlation registered for execution "ghost".
    await agent._dispatch(
        SSEEvent(
            type="tool.result",
            data=_json.dumps({"execution_id": "ghost", "step_id": "x"}),
        )
    )
    # No exception is success.
