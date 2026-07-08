from __future__ import annotations

import functools
import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from rebuno.execution import _get_current


def tool(
    tool_id: str | Callable[..., Any] | None = None,
    *,
    idempotency: str = "safe_to_retry",
) -> Any:
    """Register an async function as a Rebuno tool.

    Usable as ``@tool``, ``@tool()``, ``@tool("custom_id")``, or
    ``@tool("id", idempotency="at_most_once")``. The wrapped callable keeps the
    original signature so frameworks bind it unchanged.
    """

    def decorate(fn: Callable[..., Any], explicit_id: str | None) -> Callable[..., Any]:
        resolved_id = explicit_id if explicit_id is not None else fn.__name__
        return _build_wrapper(resolved_id, fn, idempotency)

    if callable(tool_id):
        return decorate(tool_id, None)

    explicit_id = tool_id

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        return decorate(fn, explicit_id)

    return decorator


def wrap_tool(
    name: str,
    invoke: Callable[[dict[str, Any]], Any],
    *,
    description: str = "",
    args_schema: dict[str, Any] | None = None,
    idempotency: str = "safe_to_retry",
    to_result: Callable[[Any], Any] | None = None,
    transform_args: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> Callable[..., Any]:
    """Wrap an arbitrary tool as a Rebuno-routed callable.

    ``wrap_tool`` builds the callable from a ``name`` plus an ``invoke(args)``
    seam — so it fits tools that are not plain callables: framework tool objects
    or schema-only tools (see :func:`rebuno.mcp.wrap_mcp_tool`). The returned callable routes
    every call through the kernel for policy, replay, and audit.

    Args:
        name: The tool id. Both the LLM (via ``__name__``) and the kernel see
            this exact string, so put any namespace prefix directly in it.
        invoke: ``invoke(args)`` performs the call and returns the result
            (awaitable or plain). ``args`` is the recorded argument dict.
        description: Shown to the LLM as the tool description (``__doc__``).
        args_schema: JSON schema whose ``properties``/``required`` build the
            synthetic signature frameworks introspect, and which is exposed on
            ``__input_schema__`` for frameworks that read it. ``None`` → no params.
        idempotency: ``safe_to_retry`` (default) for reads; ``at_most_once`` for
            writes that must not re-run on resume.
        to_result: Maps the raw ``invoke`` return to a JSON-serializable value
            before it is recorded as the step result. Defaults to identity.
        transform_args: Maps the caller's argument dict before it is recorded and
            passed to ``invoke``. Defaults to identity. (e.g. null-stripping.)
    """
    schema = args_schema or {}

    async def wrapper(**kwargs: Any) -> Any:
        ctx = _get_current()
        if ctx is None:
            raise RuntimeError(
                f"tool '{name}' called outside an active execution. "
                "Tools run inside a handler under agent.run() (or a test context)."
            )
        args = transform_args(kwargs) if transform_args is not None else dict(kwargs)

        async def run() -> Any:
            result = invoke(args)
            if isinstance(result, Awaitable):
                result = await result
            return to_result(result) if to_result is not None else result

        return await ctx.invoke_tool(name, args, idempotency=idempotency, run=run)

    wrapper.__name__ = name
    wrapper.__qualname__ = name
    wrapper.__doc__ = description or None
    wrapper.__signature__ = _signature_from_schema(schema)  # type: ignore[attr-defined]
    wrapper.__input_schema__ = schema  # type: ignore[attr-defined]
    return wrapper


def _signature_from_schema(schema: dict[str, Any]) -> inspect.Signature:
    """Build a keyword-only signature from a JSON-schema ``properties`` map.

    Required properties get no default; optional ones default to ``None``. The
    signature is for framework introspection only — the wrapper accepts
    ``**kwargs``, so an arg outside the schema is still passed through at runtime.
    """
    props = schema.get("properties", {}) if isinstance(schema, dict) else {}
    required = set(schema.get("required", [])) if isinstance(schema, dict) else set()
    params = [
        inspect.Parameter(
            prop,
            inspect.Parameter.KEYWORD_ONLY,
            default=inspect.Parameter.empty if prop in required else None,
        )
        for prop in props
    ]
    return inspect.Signature(params)


def _build_wrapper(tool_id: str, fn: Callable[..., Any], idempotency: str) -> Callable[..., Any]:
    sig = inspect.signature(fn)

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        ctx = _get_current()
        if ctx is None:
            raise RuntimeError(
                f"@tool '{tool_id}' called outside an active execution. "
                "Tools run inside a handler under agent.run() (or a test context)."
            )
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        return await ctx.invoke_tool(
            tool_id, arguments, idempotency=idempotency, run=lambda: fn(*bound.args, **bound.kwargs)
        )

    return wrapper
