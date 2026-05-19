from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Annotated, Any, Union, get_args, get_origin, get_type_hints

from rebuno.execution import _get_current


_JSON_TYPES: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _json_type(annotation: Any) -> dict[str, Any]:
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}
    origin = get_origin(annotation)
    if origin is Annotated:
        return _json_type(get_args(annotation)[0])
    if origin is Union:
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        if len(non_none) == 1:
            return _json_type(non_none[0])
        return {}
    base = origin or annotation
    if isinstance(base, type) and base in _JSON_TYPES:
        return {"type": _JSON_TYPES[base]}
    return {}


def _build_input_schema(fn: Callable[..., Any]) -> dict[str, Any]:
    sig = inspect.signature(fn)
    try:
        hints = get_type_hints(fn, include_extras=True)
    except Exception:
        hints = {}
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in sig.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        properties[name] = _json_type(hints.get(name, param.annotation))
        if param.default is inspect.Parameter.empty:
            required.append(name)
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


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
        return _build_wrapper(resolved_id, fn, remote)

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
    wrapper.__input_schema__ = _build_input_schema(fn)  # type: ignore[attr-defined]
    return wrapper
