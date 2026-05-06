from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rebuno.execution import _get_current


@dataclass
class ToolEntry:
    tool_id: str
    fn: Callable[..., Any]
    wrapper: Callable[..., Any]
    remote: bool


_REGISTRY: dict[str, ToolEntry] = {}


def tool(
    tool_id: str | Callable[..., Any] | None = None,
    *,
    remote: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a function as a Rebuno tool.

    Args:
        tool_id: Stable identifier used by policies and the kernel. If omitted,
            defaults to the function's ``__name__``.
        remote: If True, the body is treated as a stub. The kernel dispatches
            to a runner; the wrapper awaits the result via SSE.

    Usable as ``@tool``, ``@tool()``, ``@tool("custom_id")``, or
    ``@tool(remote=True)``.
    """

    def decorate(fn: Callable[..., Any], explicit_id: str | None) -> Callable[..., Any]:
        resolved_id = explicit_id if explicit_id is not None else fn.__name__
        wrapper = _build_wrapper(resolved_id, fn, remote)
        _REGISTRY[resolved_id] = ToolEntry(
            tool_id=resolved_id, fn=fn, wrapper=wrapper, remote=remote
        )
        return wrapper

    if callable(tool_id):
        return decorate(tool_id, None)

    explicit_id = tool_id

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        return decorate(fn, explicit_id)

    return decorator


def _build_wrapper(
    tool_id: str,
    fn: Callable[..., Any],
    remote: bool,
) -> Callable[..., Any]:
    sig = inspect.signature(fn)

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        state = _get_current()
        if state is None:
            raise RuntimeError(
                f"@tool '{tool_id}' called outside an active execution. "
                "Tools can only be invoked inside a handler running under "
                "agent.run() (or in tests where you set up an execution context)."
            )
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        if remote:
            return await state.invoke_tool(tool_id, arguments)
        return await state.invoke_tool(tool_id, arguments, local_runner=fn)

    wrapper.__rebuno_tool_id__ = tool_id  # type: ignore[attr-defined]
    wrapper.__rebuno_remote__ = remote  # type: ignore[attr-defined]
    return wrapper


def all_tools() -> list[ToolEntry]:
    """Snapshot of all registered tools. Used by Agent and Runner at startup."""
    return list(_REGISTRY.values())


def get_tool(tool_id: str) -> ToolEntry | None:
    return _REGISTRY.get(tool_id)


def _clear_registry() -> None:
    """Test helper. Do not use in production."""
    _REGISTRY.clear()
