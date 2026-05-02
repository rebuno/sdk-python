from __future__ import annotations

import asyncio
import inspect
import logging
import uuid
from collections.abc import Callable
from typing import Any

from rebuno.errors import PolicyError, RebunoError, ToolError
from rebuno.execution import _get_current

try:
    from fastmcp import Client as _FastMCPClient
    from fastmcp.client.transports import StdioTransport, StreamableHttpTransport

    _HAS_FASTMCP = True
except ImportError:
    _FastMCPClient = None  # type: ignore[assignment,misc]
    StdioTransport = None  # type: ignore[assignment,misc]
    StreamableHttpTransport = None  # type: ignore[assignment,misc]
    _HAS_FASTMCP = False

logger = logging.getLogger("rebuno.mcp")


HeadersLike = dict[str, str] | Callable[[], dict[str, str]]


_REGISTRY: list[MCPServer] = []


class MCPServer:
    """A locally-connected MCP server.

    The agent (or runner) holds the credentials and opens the MCP transport
    directly. Tool calls route through the kernel for policy/audit but the
    MCP I/O happens in this process.

    To consume tools that *runners* host (without holding credentials in the
    agent), use ``rebuno.remote.Tools(prefix)`` instead.

    Args:
        name: Display name; also the default tool ID prefix.
        url: HTTP MCP server URL. Mutually exclusive with ``command``.
        command: Stdio MCP command. Mutually exclusive with ``url``.
        args: Args for the stdio command.
        env: Env vars for the stdio subprocess.
        headers: HTTP headers — dict or zero-arg callable for token refresh.
        prefix: Tool ID prefix override (defaults to ``name``).
    """

    def __init__(
        self,
        name: str,
        *,
        url: str = "",
        command: str = "",
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        headers: HeadersLike | None = None,
        prefix: str = "",
    ):
        if not _HAS_FASTMCP:
            raise ImportError("fastmcp is required for MCP support. Install with: pip install 'rebuno[mcp]'")
        if not (url or command):
            raise ValueError(f"Server '{name}' requires either url or command")

        self.name = name
        self.prefix = prefix or name
        self.url = url
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.headers = headers

        self._tools: list[Callable[..., Any]] | None = None
        self._client: Any = None
        self._connected = False
        self._connect_lock = asyncio.Lock()

        _REGISTRY.append(self)

    @property
    def tools(self) -> list[Callable[..., Any]]:
        """List of wrapped callables. Connection is lazy: this triggers connect
        and discovery on first access.
        """
        if self._tools is None:
            raise RuntimeError(
                f"MCP server '{self.name}' has not been connected. "
                "Call await server.connect() or run inside agent.run() (which "
                "auto-connects all registered MCP servers at startup)."
            )
        return self._tools

    async def connect(self) -> None:
        """Establish the MCP connection and discover tools.

        Idempotent: safe to call multiple times. Called automatically by
        ``Agent.run()`` for every registered server.
        """
        async with self._connect_lock:
            if self._connected:
                return
            self._client = self._build_client()
            await self._client.__aenter__()
            raw_tools = await self._client.list_tools()
            self._tools = [self._wrap_tool(t) for t in raw_tools]
            self._connected = True
            logger.info("MCP server connected: %s (%d tools)", self.name, len(self._tools))

    async def disconnect(self) -> None:
        if self._client is not None and self._connected:
            try:
                await self._client.__aexit__(None, None, None)
            except Exception:
                logger.debug("Error disconnecting MCP server %s", self.name, exc_info=True)
        self._client = None
        self._connected = False
        self._tools = None

    def _build_client(self) -> Any:
        if self.url:
            # Resolve headers at connect time so each (re)connect picks up
            # fresh values from a callable headers source.
            resolved = self._resolve_headers() if self.headers else None
            transport: Any = StreamableHttpTransport(url=self.url, headers=resolved) if resolved else self.url
            return _FastMCPClient(transport)
        transport = StdioTransport(command=self.command, args=self.args, env=self.env)
        return _FastMCPClient(transport)

    def _resolve_headers(self) -> dict[str, str]:
        if callable(self.headers):
            return self.headers()
        return self.headers or {}

    def _wrap_tool(self, raw: Any) -> Callable[..., Any]:
        tool_name = raw.name
        tool_id = f"{self.prefix}.{tool_name}"
        description = getattr(raw, "description", "") or ""
        schema = getattr(raw, "inputSchema", None) or {}
        props = schema.get("properties", {}) if isinstance(schema, dict) else {}
        required = set(schema.get("required", [])) if isinstance(schema, dict) else set()

        params = []
        for prop_name in props:
            kind = inspect.Parameter.KEYWORD_ONLY
            default = inspect.Parameter.empty if prop_name in required else None
            params.append(inspect.Parameter(prop_name, kind, default=default))

        server = self

        async def wrapper(**kwargs: Any) -> Any:
            state = _get_current()
            if state is None:
                raise RuntimeError(f"MCP tool '{tool_id}' called outside an active execution.")
            # Strip None values: LLMs often fill optional fields with null,
            # but MCP servers typically reject null for typed parameters.
            args = {k: v for k, v in kwargs.items() if v is not None}
            return await _invoke_mcp(state, server, tool_id, tool_name, args)

        wrapper.__name__ = tool_name
        wrapper.__doc__ = description
        wrapper.__signature__ = inspect.Signature(params)  # type: ignore[attr-defined]
        wrapper.__rebuno_tool_id__ = tool_id  # type: ignore[attr-defined]
        wrapper.__rebuno_mcp__ = True  # type: ignore[attr-defined]
        wrapper.__rebuno_input_schema__ = schema  # type: ignore[attr-defined]
        return wrapper


async def _invoke_mcp(
    state: Any,
    server: MCPServer,
    tool_id: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> Any:
    """Submit invoke_tool intent, then call the MCP server, then report result."""
    idempotency_key = f"{state.id}:{tool_id}:{uuid.uuid4().hex[:8]}"
    intent_result = await state._client.submit_intent(
        execution_id=state.id,
        session_id=state.session_id,
        intent_type="invoke_tool",
        tool_id=tool_id,
        arguments=arguments,
        idempotency_key=idempotency_key,
        remote=False,
    )
    if not intent_result.accepted:
        raise PolicyError(intent_result.error or "Intent denied by policy")
    if not intent_result.step_id:
        raise RebunoError("No step_id returned for MCP invoke_tool intent")

    step_id = intent_result.step_id
    if intent_result.pending_approval:
        approval = await state._wait(
            "approval",
            step_id,
            f"Timed out waiting for approval (step_id={step_id})",
        )
        if not approval.get("approved", False):
            raise PolicyError("Tool invocation denied by human approval")

    try:
        raw = await server._client.call_tool(tool_name, arguments)
        output = _flatten_mcp_result(raw)
    except Exception as e:
        await state._client.report_step_result(
            execution_id=state.id,
            session_id=state.session_id,
            step_id=step_id,
            success=False,
            error=str(e),
        )
        raise ToolError(message=str(e), tool_id=tool_id, step_id=step_id) from e

    await state._client.report_step_result(
        execution_id=state.id,
        session_id=state.session_id,
        step_id=step_id,
        success=True,
        data=output,
    )
    return output


def _flatten_mcp_result(result: Any) -> str:
    content = getattr(result, "content", None)
    if content is None:
        return str(result)
    texts = [item.text if getattr(item, "type", None) == "text" else str(item) for item in content]
    return texts[0] if len(texts) == 1 else "\n".join(texts)


async def connect_all() -> None:
    """Connect every registered MCP server. Used by Agent.run() startup."""
    for server in _REGISTRY:
        try:
            await server.connect()
        except Exception:
            logger.warning("MCP server %s failed to connect; will retry lazily", server.name, exc_info=True)


async def disconnect_all() -> None:
    for server in _REGISTRY:
        await server.disconnect()


def all_servers() -> list[MCPServer]:
    return list(_REGISTRY)


def _clear_registry() -> None:
    """Test helper. Do not use in production."""
    _REGISTRY.clear()
