"""@tool decorator: registry, wrapper preserves metadata, execution gating."""

from __future__ import annotations

import asyncio
import inspect

import pytest

from rebuno import tool
from rebuno.execution import _reset_current, _set_current
from rebuno.tool import all_tools, get_tool


def test_decorator_registers():
    @tool("test.add")
    async def add(a: int, b: int) -> int:
        return a + b

    entry = get_tool("test.add")
    assert entry is not None
    assert entry.tool_id == "test.add"
    assert entry.remote is False


def test_remote_flag_persisted():
    @tool("test.heavy", remote=True)
    async def heavy(x: int) -> int: ...

    entry = get_tool("test.heavy")
    assert entry is not None
    assert entry.remote is True
    assert heavy.__rebuno_remote__ is True


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
    assert echo.__rebuno_tool_id__ == "test.echo"


def test_calling_tool_outside_execution_raises():
    @tool("test.nope")
    async def nope() -> None:
        return None

    async def go():
        with pytest.raises(RuntimeError, match="outside an active execution"):
            await nope()

    asyncio.run(go())


def test_all_tools_returns_registered_entries():
    @tool("a.one")
    async def one(): ...

    @tool("a.two")
    async def two(): ...

    ids = {e.tool_id for e in all_tools()}
    assert ids == {"a.one", "a.two"}


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
            # simulate kernel allow + local execution
            return await local_runner(**arguments)

    token = _set_current(_MockState())  # type: ignore[arg-type]
    try:
        result = await add(2, 3)
    finally:
        _reset_current(token)

    assert result == 5
    assert captured["tool_id"] == "test.add"
    assert captured["arguments"] == {"a": 2, "b": 3}
    assert captured["local_runner"] is add.__wrapped__  # functools.wraps target


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
