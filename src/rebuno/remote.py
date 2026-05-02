"""Remote tool discovery — fetch schemas from the kernel directory.

Use this to consume tools that runners host elsewhere in the cluster. Works
for any source the runner publishes, whether @tool functions or MCP servers.

Example::

    from rebuno import Agent, remote

    agent = Agent("swe")
    compute = remote.Tools("compute")  # discover everything under "compute.*"
    github  = remote.Tools("github")   # discover everything under "github.*"

    async def process(prompt: str):
        graph = create_agent(llm, [my_local_tool, *compute.tools, *github.tools])
        ...

    agent.run(process)

Each callable in `.tools` submits a kernel intent with ``remote=True``. The
kernel routes to a runner advertising that tool ID; the runner executes;
result returns via SSE. The MCP transport (if any) is a runner-side concern.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import uuid
from collections.abc import Callable
from typing import Any

from rebuno.errors import PolicyError, RebunoError, ToolError
from rebuno.execution import _get_current

logger = logging.getLogger("rebuno.remote")


_REGISTRY: list[Tools] = []


class Tools:
    """A discovery handle for tools matching a prefix.

    Lazy: schemas are fetched at agent startup (or on first ``.tools`` access
    if used outside an agent). Each call submits a remote intent through the
    kernel.
    """

    def __init__(self, prefix: str):
        if not prefix:
            raise ValueError("Tools requires a non-empty prefix")
        self.prefix = prefix
        self._tools: list[Callable[..., Any]] | None = None
        self._connected = False
        self._lock = asyncio.Lock()
        _REGISTRY.append(self)

    @property
    def tools(self) -> list[Callable[..., Any]]:
        if self._tools is None:
            raise RuntimeError(
                f"remote.Tools(prefix='{self.prefix}') has not been connected. "
                "Call await tools.connect() or run inside agent.run() (which "
                "auto-connects all registered remote.Tools at startup)."
            )
        return self._tools

    async def connect(self, client: Any) -> None:
        """Fetch schemas from the kernel and build wrapped callables.

        Idempotent. Called automatically by Agent.run_async() with the agent's
        Client instance.
        """
        async with self._lock:
            if self._connected:
                return
            schemas = await client.list_tools(prefix=self.prefix)
            self._tools = [self._wrap(s) for s in schemas]
            self._connected = True
            logger.info(
                "remote tools resolved: prefix=%s count=%d",
                self.prefix,
                len(self._tools),
            )

    async def disconnect(self) -> None:
        self._connected = False
        self._tools = None

    def _wrap(self, schema: dict[str, Any]) -> Callable[..., Any]:
        tool_id = schema["id"]
        description = schema.get("description", "") or ""
        input_schema = schema.get("input_schema") or {}
        props = input_schema.get("properties", {}) if isinstance(input_schema, dict) else {}
        required = set(input_schema.get("required", [])) if isinstance(input_schema, dict) else set()

        params: list[inspect.Parameter] = []
        for prop_name in props:
            default = inspect.Parameter.empty if prop_name in required else None
            params.append(
                inspect.Parameter(
                    prop_name,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=default,
                )
            )

        async def wrapper(**kwargs: Any) -> Any:
            state = _get_current()
            if state is None:
                raise RuntimeError(f"remote tool '{tool_id}' called outside an active execution.")
            return await _invoke_remote(state, tool_id, kwargs)

        # Pretty name: github.create_pr -> create_pr
        wrapper.__name__ = tool_id.rsplit(".", 1)[-1]
        wrapper.__qualname__ = wrapper.__name__
        wrapper.__doc__ = description
        wrapper.__signature__ = inspect.Signature(params)  # type: ignore[attr-defined]
        wrapper.__rebuno_tool_id__ = tool_id  # type: ignore[attr-defined]
        wrapper.__rebuno_remote__ = True  # type: ignore[attr-defined]
        return wrapper


async def _invoke_remote(state: Any, tool_id: str, arguments: dict[str, Any]) -> Any:
    """Submit a remote invoke_tool intent and await the runner's result."""
    idempotency_key = f"{state.id}:{tool_id}:{uuid.uuid4().hex[:8]}"
    intent_result = await state._client.submit_intent(
        execution_id=state.id,
        session_id=state.session_id,
        intent_type="invoke_tool",
        tool_id=tool_id,
        arguments=arguments,
        idempotency_key=idempotency_key,
        remote=True,
    )
    if not intent_result.accepted:
        raise PolicyError(intent_result.error or "Intent denied by policy")
    if not intent_result.step_id:
        raise RebunoError("No step_id returned for remote invoke_tool intent")

    step_id = intent_result.step_id
    if intent_result.pending_approval:
        approval = await state._wait(
            "approval",
            step_id,
            f"Timed out waiting for approval (step_id={step_id})",
        )
        if not approval.get("approved", False):
            raise PolicyError("Tool invocation denied by human approval")

    data = await state._wait(
        "result",
        step_id,
        f"Timed out waiting for tool result (step_id={step_id})",
    )
    if data.get("status") == "failed":
        raise ToolError(
            message=data.get("error", "Tool execution failed"),
            tool_id=tool_id,
            step_id=step_id,
        )
    return data.get("result")


def all_handles() -> list[Tools]:
    return list(_REGISTRY)


async def connect_all(client: Any) -> None:
    """Connect every registered remote.Tools handle. Used by Agent startup."""
    for handle in _REGISTRY:
        try:
            await handle.connect(client)
        except Exception:
            logger.warning(
                "remote.Tools(prefix=%s) failed to resolve from kernel; will be empty",
                handle.prefix,
                exc_info=True,
            )


async def disconnect_all() -> None:
    for handle in _REGISTRY:
        await handle.disconnect()


def _clear_registry() -> None:
    """Test helper."""
    _REGISTRY.clear()
