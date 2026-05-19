"""@tool decorator: wrapper attributes, signature preservation, execution gating."""

from __future__ import annotations

import asyncio
import inspect

import pytest

from rebuno import tool
from rebuno.execution import _reset_current, _set_current


def test_decorator_sets_tool_id():
    @tool("test.add")
    async def add(a: int, b: int) -> int:
        return a + b

    assert add.__rebuno_tool_id__ == "test.add"
    assert add.__rebuno_remote__ is False


def test_tool_id_defaults_to_function_name():
    @tool
    async def my_op() -> int:
        return 1

    assert my_op.__rebuno_tool_id__ == "my_op"


def test_remote_flag_persisted():
    @tool("test.heavy", remote=True)
    async def heavy(x: int) -> int: ...

    assert heavy.__rebuno_remote__ is True
    assert heavy.__rebuno_tool_id__ == "test.heavy"


def test_wrapper_preserves_name_and_signature():
    @tool("test.echo")
    async def echo(text: str, count: int = 1) -> str:
        """Echo a string back."""
        return text * count

    sig = inspect.signature(echo)
    assert list(sig.parameters) == ["text", "count"]
    assert sig.parameters["count"].default == 1
    assert echo.__name__ == "echo"
    assert echo.__doc__ == "Echo a string back."


def test_wrapper_exposes_raw_function():
    @tool("test.raw")
    async def raw(x: int) -> int:
        return x * 2

    assert callable(raw.__wrapped__)
    assert raw.__wrapped__.__name__ == "raw"


def test_input_schema_built_from_annotations():
    @tool("test.schema")
    async def fn(name: str, count: int = 1) -> None: ...

    schema = fn.__input_schema__
    assert schema["type"] == "object"
    assert schema["properties"]["name"] == {"type": "string"}
    assert schema["properties"]["count"] == {"type": "integer"}
    assert schema["required"] == ["name"]


def test_calling_tool_outside_execution_raises():
    @tool("test.nope")
    async def nope() -> None:
        return None

    async def go():
        with pytest.raises(RuntimeError, match="outside an active execution"):
            await nope()

    asyncio.run(go())


async def test_local_tool_calls_invoke_through_state(monkeypatch):
    """End-to-end: decorated local tool, called inside an execution context,
    should round-trip through ExecutionState.invoke_tool with local_runner=fn.
    """

    @tool("test.add")
    async def add(a: int, b: int) -> int:
        return a + b

    captured = {}

    class _MockState:
        id = "exec-1"
        session_id = "sess-1"

        async def invoke_tool(self, tool_id, arguments, *, idempotency_key="", local_runner=None):
            captured["tool_id"] = tool_id
            captured["arguments"] = arguments
            captured["local_runner"] = local_runner
            return await local_runner(**arguments)

    token = _set_current(_MockState())  # type: ignore[arg-type]
    try:
        result = await add(2, 3)
    finally:
        _reset_current(token)

    assert result == 5
    assert captured["tool_id"] == "test.add"
    assert captured["arguments"] == {"a": 2, "b": 3}
    assert captured["local_runner"] is add.__wrapped__


async def test_remote_tool_dispatches_without_local_runner():
    @tool("test.remote", remote=True)
    async def heavy(x: int) -> int: ...

    captured = {}

    class _MockState:
        id = "exec-1"
        session_id = "sess-1"

        async def invoke_tool(self, tool_id, arguments, *, idempotency_key="", local_runner=None):
            captured["tool_id"] = tool_id
            captured["local_runner"] = local_runner
            return 42

    token = _set_current(_MockState())  # type: ignore[arg-type]
    try:
        result = await heavy(7)
    finally:
        _reset_current(token)

    assert result == 42
    assert captured["tool_id"] == "test.remote"
    assert captured["local_runner"] is None
