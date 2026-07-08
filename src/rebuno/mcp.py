from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from typing import Any

from rebuno.tool import wrap_tool

CallFn = Callable[[str, dict[str, Any]], Awaitable[Any]]
ToResult = Callable[[Any], Any]


def wrap_mcp_tools(
    descriptors: Iterable[Any],
    *,
    call: CallFn,
    prefix: str = "",
    idempotency: str = "safe_to_retry",
    to_result: ToResult | None = None,
) -> list[Callable[..., Any]]:
    """Wrap a list of MCP tool descriptors (e.g. the output of ``list_tools()``).

    See :func:`wrap_mcp_tool` for the per-tool behaviour and arguments.
    """
    return [
        wrap_mcp_tool(d, call=call, prefix=prefix, idempotency=idempotency, to_result=to_result) for d in descriptors
    ]


def wrap_mcp_tool(
    descriptor: Any,
    *,
    call: CallFn,
    prefix: str = "",
    idempotency: str = "safe_to_retry",
    to_result: ToResult | None = None,
) -> Callable[..., Any]:
    """Manufacture a Rebuno-routed callable from one MCP tool descriptor.

    Args:
        descriptor: An MCP tool with ``name``, ``description``, and ``inputSchema``
            (the spec field names). Attribute or dict access both work, so the
            official ``mcp`` SDK's ``Tool``, a fastmcp tool, or a plain dict all fit.
        call: ``call(tool_name, args)`` — your MCP client's invocation, the only
            seam to the transport. Receives the bare tool name.
        prefix: Tool-id namespace. The LLM and the kernel both see ``f"{prefix}_{name}"``;
            only the MCP server (via ``call``) sees the bare ``name``. Empty prefix
            uses ``name`` as-is.
        idempotency: ``safe_to_retry`` (default) for reads; ``at_most_once`` for
            writes that must not re-run on resume.
        to_result: Maps the raw ``call`` return to a JSON-serializable value before
            it is recorded. Defaults to flattening standard MCP ``CallToolResult``
            shapes (structured content, else text blocks).

    Returns:
        A plain async callable (see :func:`rebuno.wrap_tool`) whose ``__name__`` and
        kernel target are the prefixed id, with the raw ``inputSchema`` on
        ``__input_schema__``.
    """
    name = _field(descriptor, "name")
    description = _field(descriptor, "description", default="") or ""
    schema = _field(descriptor, "inputSchema", default=None) or {}
    tool_id = f"{prefix}_{name}" if prefix else name

    return wrap_tool(
        tool_id,
        lambda args: call(name, args),  # the wire call uses the bare name
        description=description,
        args_schema=schema,
        idempotency=idempotency,
        to_result=to_result if to_result is not None else _default_flatten,
        transform_args=_strip_none,
    )


def _field(descriptor: Any, key: str, *, default: Any = None) -> Any:
    """Read ``key`` from a descriptor by attribute, falling back to dict access."""
    if isinstance(descriptor, dict):
        return descriptor.get(key, default)
    return getattr(descriptor, key, default)


def _strip_none(args: dict[str, Any]) -> dict[str, Any]:
    """Drop null-valued args: LLMs often fill optional fields with null, but MCP
    servers typically reject null for typed parameters."""
    return {k: v for k, v in args.items() if v is not None}


def _default_flatten(raw: Any) -> Any:
    """Flatten a standard MCP ``CallToolResult`` to a JSON-serializable value.

    Prefers structured content (``structured_content`` in fastmcp,
    ``structuredContent`` in the official SDK). Otherwise joins text content
    blocks. A value that is neither (already a dict/str the caller flattened
    itself) is passed through unchanged.
    """
    structured = getattr(raw, "structured_content", None)
    if structured is None:
        structured = getattr(raw, "structuredContent", None)
    if structured is not None:
        return structured

    content = getattr(raw, "content", None)
    if content is not None:
        texts = [b.text if getattr(b, "type", None) == "text" else str(b) for b in content]
        return texts[0] if len(texts) == 1 else "\n".join(texts)

    return raw
