"""MCPServer: lazy connect, headers, tool wrapping."""

from __future__ import annotations

import inspect

import pytest

from rebuno import (
    MCPServer,
    mcp,  # internal access for all_servers
)


def test_server_registration_appends_to_registry():
    s = MCPServer("github", url="https://example/mcp")
    assert s in mcp.all_servers()
    assert s.name == "github"
    assert s.prefix == "github"  # default to name


def test_server_explicit_prefix_overrides_default():
    s = MCPServer("github", url="https://example/mcp", prefix="gh")
    assert s.prefix == "gh"


def test_server_requires_url_or_command():
    with pytest.raises(ValueError, match="requires either url or command"):
        MCPServer("bad")


def test_tools_property_raises_before_connect():
    s = MCPServer("github", url="https://example/mcp")
    with pytest.raises(RuntimeError, match="has not been connected"):
        _ = s.tools


def test_resolve_headers_callable_invoked_per_call():
    calls = {"n": 0}

    def headers():
        calls["n"] += 1
        return {"X-Token": str(calls["n"])}

    s = MCPServer("svc", url="https://example/mcp", headers=headers)
    assert s._resolve_headers() == {"X-Token": "1"}
    assert s._resolve_headers() == {"X-Token": "2"}


def test_resolve_headers_dict_returned_directly():
    s = MCPServer("svc", url="https://example/mcp", headers={"X-Token": "static"})
    assert s._resolve_headers() == {"X-Token": "static"}
    assert s._resolve_headers() == {"X-Token": "static"}


def test_resolve_headers_none_returns_empty():
    s = MCPServer("svc", url="https://example/mcp")
    assert s._resolve_headers() == {}


class _FakeRawTool:
    """Mimics fastmcp's tool object surface."""

    def __init__(self, name: str, description: str, schema: dict | None = None):
        self.name = name
        self.description = description
        self.inputSchema = schema or {}


def test_wrap_tool_builds_signature_from_input_schema():
    s = MCPServer("github", url="https://example/mcp")
    raw = _FakeRawTool(
        name="issue_read",
        description="Read an issue.",
        schema={
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "number": {"type": "integer"},
            },
            "required": ["owner", "repo", "number"],
        },
    )
    fn = s._wrap_tool(raw)

    assert fn.__name__ == "issue_read"
    assert fn.__doc__ == "Read an issue."
    assert fn.__rebuno_tool_id__ == "github.issue_read"
    assert fn.__rebuno_mcp__ is True

    sig = inspect.signature(fn)
    assert list(sig.parameters) == ["owner", "repo", "number"]
    for name in ("owner", "repo", "number"):
        assert sig.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_wrap_tool_optional_params_have_default_none():
    s = MCPServer("github", url="https://example/mcp")
    raw = _FakeRawTool(
        "search",
        "",
        schema={
            "properties": {"q": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["q"],
        },
    )
    fn = s._wrap_tool(raw)
    sig = inspect.signature(fn)
    assert sig.parameters["q"].default is inspect.Parameter.empty
    assert sig.parameters["limit"].default is None


def test_wrap_tool_calling_outside_execution_raises():
    import asyncio

    s = MCPServer("github", url="https://example/mcp")
    fn = s._wrap_tool(_FakeRawTool("noop", "", schema={}))

    async def go():
        with pytest.raises(RuntimeError, match="outside an active execution"):
            await fn()

    asyncio.run(go())
