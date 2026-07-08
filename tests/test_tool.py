import inspect
from types import SimpleNamespace
from typing import Any

import pytest

from rebuno.execution import ExecutionContext, _reset_current, _set_current
from rebuno.tool import tool, wrap_tool
from rebuno.types import StepDecision


class FakeKernel:
    def __init__(self, decision):
        self.decision = decision
        self.completed = []

    async def submit_step(self, execution_id, *, kind, target, args, idempotency, step_id):
        self.captured = dict(kind=kind, target=target, args=args, idempotency=idempotency)
        return self.decision

    async def complete_step(self, execution_id, step_id, *, result):
        self.completed.append(result)


@tool
async def search(query: str, limit: int = 10) -> dict:
    return {"q": query, "limit": limit}


@tool("custom_id", idempotency="at_most_once")
async def danger(x: int) -> int:
    return x + 1


@tool
async def variadic(a: int, *rest: int, **opts: Any) -> dict:
    return {"a": a, "rest": rest, "opts": opts}


async def test_tool_runs_under_context_and_binds_args():
    k = FakeKernel(StepDecision(decision="proceed"))
    token = _set_current(ExecutionContext(kernel=k, execution_id="e1", agent_id="a", input=None))
    try:
        out = await search("hi")
    finally:
        _reset_current(token)
    assert out == {"q": "hi", "limit": 10}
    assert k.captured["target"] == "search"
    assert k.captured["args"] == {"query": "hi", "limit": 10}


async def test_tool_id_and_idempotency_override():
    k = FakeKernel(StepDecision(decision="proceed"))
    token = _set_current(ExecutionContext(kernel=k, execution_id="e1", agent_id="a", input=None))
    try:
        await danger(1)
    finally:
        _reset_current(token)
    assert k.captured["target"] == "custom_id"
    assert k.captured["idempotency"] == "at_most_once"


async def test_tool_outside_context_raises():
    with pytest.raises(RuntimeError):
        await search("hi")


async def test_tool_variadic_args_expand_correctly():
    k = FakeKernel(StepDecision(decision="proceed"))
    token = _set_current(ExecutionContext(kernel=k, execution_id="e1", agent_id="a", input=None))
    try:
        out = await variadic(1, 2, 3, x=9)
    finally:
        _reset_current(token)
    assert out == {"a": 1, "rest": (2, 3), "opts": {"x": 9}}


# --- wrap_tool: generic primitive for non-decorator tool shapes ---

_SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": "string"}, "body": {"type": "string"}},
    "required": ["title"],
}


def _ctx(kernel):
    return _set_current(ExecutionContext(kernel=kernel, execution_id="e1", agent_id="a", input=None))


async def test_wrap_tool_uses_name_verbatim_as_target():
    k = FakeKernel(StepDecision(decision="proceed"))
    seen: list = []

    async def invoke(args):
        seen.append(args)
        return {"id": 1}

    fn = wrap_tool("github_create_issue", invoke, args_schema=_SCHEMA)
    token = _ctx(k)
    try:
        out = await fn(title="bug")
    finally:
        _reset_current(token)
    assert k.captured["kind"] == "tool_call"
    assert k.captured["target"] == "github_create_issue"  # name is the id, no prefix machinery
    assert k.captured["args"] == {"title": "bug"}
    assert seen == [{"title": "bug"}]  # invoke receives the args dict
    assert out == {"id": 1}
    assert k.completed == [{"id": 1}]


async def test_wrap_tool_accepts_sync_invoke():
    k = FakeKernel(StepDecision(decision="proceed"))
    fn = wrap_tool("t", lambda args: {"echo": args})
    token = _ctx(k)
    try:
        out = await fn(x=1)
    finally:
        _reset_current(token)
    assert out == {"echo": {"x": 1}}


async def test_wrap_tool_to_result_applied_before_recording():
    k = FakeKernel(StepDecision(decision="proceed"))

    async def invoke(args):
        return SimpleNamespace(data={"v": 5})

    fn = wrap_tool("t", invoke, to_result=lambda r: r.data)
    token = _ctx(k)
    try:
        out = await fn()
    finally:
        _reset_current(token)
    assert out == {"v": 5}
    assert k.completed == [{"v": 5}]


async def test_wrap_tool_transform_args_applied_before_record_and_invoke():
    k = FakeKernel(StepDecision(decision="proceed"))
    seen: list = []

    async def invoke(args):
        seen.append(args)
        return None

    fn = wrap_tool("t", invoke, transform_args=lambda a: {k: v for k, v in a.items() if v is not None})
    token = _ctx(k)
    try:
        await fn(x=1, y=None)
    finally:
        _reset_current(token)
    assert k.captured["args"] == {"x": 1}
    assert seen == [{"x": 1}]


async def test_wrap_tool_synthetic_signature_and_metadata():
    fn = wrap_tool("create_issue", lambda a: None, description="Open an issue", args_schema=_SCHEMA)
    sig = inspect.signature(fn)
    assert list(sig.parameters) == ["title", "body"]
    assert sig.parameters["title"].default is inspect.Parameter.empty  # required
    assert sig.parameters["body"].default is None  # optional
    assert fn.__name__ == "create_issue"
    assert fn.__doc__ == "Open an issue"
    assert fn.__input_schema__["properties"]["title"] == {"type": "string"}


async def test_wrap_tool_without_schema_has_empty_signature_but_passes_kwargs():
    k = FakeKernel(StepDecision(decision="proceed"))
    seen: list = []
    fn = wrap_tool("t", lambda a: seen.append(a))
    assert list(inspect.signature(fn).parameters) == []
    token = _ctx(k)
    try:
        await fn(anything=1)
    finally:
        _reset_current(token)
    assert seen == [{"anything": 1}]


async def test_wrap_tool_forwards_idempotency():
    k = FakeKernel(StepDecision(decision="proceed"))
    fn = wrap_tool("t", lambda a: None, idempotency="at_most_once")
    token = _ctx(k)
    try:
        await fn()
    finally:
        _reset_current(token)
    assert k.captured["idempotency"] == "at_most_once"


async def test_wrap_tool_replays_without_calling_invoke():
    k = FakeKernel(StepDecision(decision="replay", result={"cached": 1}))
    seen: list = []
    fn = wrap_tool("t", lambda a: seen.append(a))
    token = _ctx(k)
    try:
        out = await fn(x=1)
    finally:
        _reset_current(token)
    assert out == {"cached": 1}
    assert seen == []


async def test_wrap_tool_outside_context_raises():
    fn = wrap_tool("t", lambda a: None)
    with pytest.raises(RuntimeError):
        await fn()
