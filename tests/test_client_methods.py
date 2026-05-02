"""Client method shapes: request body / params / response parsing."""

from __future__ import annotations

import json

import httpx
from conftest import SAMPLE_EXECUTION, mock_client

from rebuno.types import ExecutionStatus


def _captured(captured: dict, request: httpx.Request, body: dict | None = None):
    captured["method"] = request.method
    captured["path"] = request.url.path
    captured["query"] = dict(request.url.params)
    if request.content:
        captured["body"] = json.loads(request.content)
    return httpx.Response(200, json=body or SAMPLE_EXECUTION)


async def test_create_sends_agent_id_and_input_and_labels():
    captured: dict = {}

    def h(request: httpx.Request) -> httpx.Response:
        return _captured(captured, request)

    client = mock_client({("POST", "/v0/executions"): h})
    await client.create("swe", input={"prompt": "hi"}, labels={"env": "prod"})

    assert captured["body"] == {
        "agent_id": "swe",
        "input": {"prompt": "hi"},
        "labels": {"env": "prod"},
    }


async def test_create_omits_optional_fields_when_absent():
    captured: dict = {}

    def h(request: httpx.Request) -> httpx.Response:
        return _captured(captured, request)

    client = mock_client({("POST", "/v0/executions"): h})
    await client.create("swe")
    assert captured["body"] == {"agent_id": "swe"}


async def test_get_returns_execution_model():
    client = mock_client({("GET", "/v0/executions/exec-1"): SAMPLE_EXECUTION})
    ex = await client.get("exec-1")
    assert ex.id == "exec-1"
    assert ex.status == ExecutionStatus.RUNNING
    assert ex.agent_id == "agent-1"


async def test_list_serializes_label_filters_correctly():
    captured: dict = {}

    def h(request: httpx.Request) -> httpx.Response:
        captured["query_pairs"] = [(k, v) for k, v in request.url.params.multi_items()]
        return httpx.Response(200, json={"executions": [], "next_cursor": ""})

    client = mock_client({("GET", "/v0/executions"): h})
    await client.list(
        status=ExecutionStatus.RUNNING,
        agent_id="swe",
        labels={"env": "prod", "team": "core"},
    )

    pairs = captured["query_pairs"]
    assert ("status", "running") in pairs
    assert ("agent_id", "swe") in pairs
    # multi-value label encoding
    label_values = [v for k, v in pairs if k == "label"]
    assert "env:prod" in label_values
    assert "team:core" in label_values


async def test_cancel_returns_updated_execution():
    cancelled = {**SAMPLE_EXECUTION, "status": "cancelled"}
    client = mock_client(
        {
            ("POST", "/v0/executions/exec-1/cancel"): cancelled,
        }
    )
    ex = await client.cancel("exec-1")
    assert ex.status == ExecutionStatus.CANCELLED


async def test_send_signal_includes_payload():
    captured: dict = {}

    def h(request: httpx.Request) -> httpx.Response:
        return _captured(captured, request, body={"status": "queued"})

    client = mock_client(
        {
            ("POST", "/v0/executions/exec-1/signal"): h,
        }
    )
    result = await client.send_signal("exec-1", "approval", payload={"approved": True})
    assert result.status == "queued"
    assert captured["body"] == {"signal_type": "approval", "payload": {"approved": True}}


async def test_submit_intent_marshals_remote_flag():
    captured: dict = {}

    def h(request: httpx.Request) -> httpx.Response:
        return _captured(captured, request, body={"accepted": True, "step_id": "s-1"})

    client = mock_client({("POST", "/v0/agents/intent"): h})
    result = await client.submit_intent(
        execution_id="e",
        session_id="s",
        intent_type="invoke_tool",
        tool_id="x",
        arguments={"a": 1},
        remote=True,
    )
    assert result.accepted
    assert result.step_id == "s-1"
    intent = captured["body"]["intent"]
    assert intent["remote"] is True
    assert intent["tool_id"] == "x"
    assert intent["arguments"] == {"a": 1}


async def test_report_step_result_success():
    captured: dict = {}

    def h(request: httpx.Request) -> httpx.Response:
        return _captured(captured, request, body={})

    client = mock_client({("POST", "/v0/agents/step-result"): h})
    await client.report_step_result(
        execution_id="e",
        session_id="s",
        step_id="step-1",
        success=True,
        data={"out": 42},
    )
    assert captured["body"] == {
        "execution_id": "e",
        "session_id": "s",
        "step_id": "step-1",
        "success": True,
        "data": {"out": 42},
    }


async def test_run_until_complete_invokes_callback_then_returns_final():
    """run_until_complete: stream events through SSE, call on_event for each,
    fetch final execution. Uses a fake events() iterator since SSE-stream
    mocking via MockTransport is awkward."""
    from rebuno.types import Event

    async def fake_events(execution_id, after_sequence=0):
        for t in ("execution.started", "tool.invoked", "execution.completed"):
            yield Event(id="e", execution_id=execution_id, type=t)

    final = {**SAMPLE_EXECUTION, "status": "completed", "output": {"ok": True}}
    client = mock_client(
        {
            ("POST", "/v0/executions"): SAMPLE_EXECUTION,
            ("GET", "/v0/executions/exec-1"): final,
        }
    )
    client.events = fake_events  # type: ignore[method-assign]

    seen: list[str] = []
    result = await client.run_until_complete(
        "swe",
        input={"x": 1},
        on_event=lambda e: seen.append(e.type),
    )
    assert result.status.value == "completed"
    assert seen == ["execution.started", "tool.invoked", "execution.completed"]


async def test_run_iterator_terminates_on_terminal_event():
    from rebuno.types import Event

    async def fake_events(execution_id, after_sequence=0):
        yield Event(id="1", execution_id=execution_id, type="tool.invoked")
        yield Event(id="2", execution_id=execution_id, type="execution.failed")
        # Should not be reached
        yield Event(id="3", execution_id=execution_id, type="phantom")

    client = mock_client({("POST", "/v0/executions"): SAMPLE_EXECUTION})
    client.events = fake_events  # type: ignore[method-assign]

    seen = []
    async for e in client.run("swe", input={}):
        seen.append(e.type)
    assert seen == ["tool.invoked", "execution.failed"]
