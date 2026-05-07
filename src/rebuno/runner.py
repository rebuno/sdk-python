from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Iterable
from typing import Any

from rebuno._internal import install_shutdown_handlers, jittered_backoff
from rebuno._internal.schema import fn_to_json_schema
from rebuno.client import Client
from rebuno.mcp import MCPServer, _flatten_mcp_result
from rebuno.tool import all_tools, get_tool
from rebuno.types import Job

logger = logging.getLogger("rebuno.runner")


class Runner:
    """Tool runner.

    Args:
        runner_id: Unique runner identifier.
        kernel_url: Override REBUNO_URL env var.
        api_key: Override REBUNO_API_KEY env var.
        capabilities: Optional explicit list of tool IDs to advertise. If
            empty, advertises every tool in the @tool registry.
        consumer_id: Unique consumer identifier (auto-generated if empty).
    """

    def __init__(
        self,
        runner_id: str,
        *,
        kernel_url: str | None = None,
        api_key: str | None = None,
        capabilities: Iterable[str] | None = None,
        consumer_id: str = "",
        reconnect_delay: float = 2.0,
        max_reconnect_delay: float = 60.0,
    ):
        if not runner_id:
            raise ValueError("runner_id must not be empty")
        self.runner_id = runner_id
        self.consumer_id = consumer_id or f"{runner_id}-{uuid.uuid4().hex[:8]}"
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay

        self._client = Client(base_url=kernel_url, api_key=api_key)
        self._explicit_caps: list[str] | None = list(capabilities) if capabilities else None
        self._mcp_servers: list[MCPServer] = []
        self._running = False
        self._shutdown = asyncio.Event()
        self._connect_task: asyncio.Task[None] | None = None

    def host(self, server: MCPServer) -> None:
        """Host an MCP server on this runner. Tools become advertised
        capabilities and dispatched calls forward to the MCP transport.
        """
        self._mcp_servers.append(server)

    def run(self) -> None:
        asyncio.run(self.run_async())

    async def run_async(self) -> None:
        self._running = True
        self._shutdown.clear()
        install_shutdown_handlers(self._handle_shutdown)

        # Connect MCP servers hosted by this runner so we know their tool IDs.
        for server in self._mcp_servers:
            try:
                await server.connect()
            except Exception:
                logger.exception("Failed to connect MCP server %s", server.name)

        capabilities = self._capabilities()
        logger.info(
            "Runner starting: runner_id=%s capabilities=%s",
            self.runner_id,
            capabilities,
        )

        try:
            await self._publish_schemas()
        except Exception:
            logger.exception("Failed to publish tool schemas to kernel")

        consecutive_failures = 0
        try:
            while self._running:
                try:
                    self._connect_task = asyncio.ensure_future(self._stream_loop(capabilities))
                    await self._connect_task
                    consecutive_failures = 0
                except asyncio.CancelledError:
                    break
                except Exception:
                    consecutive_failures += 1
                    logger.exception("SSE connection error, reconnecting")
                    if not self._running:
                        break
                    delay = jittered_backoff(
                        self.reconnect_delay,
                        consecutive_failures,
                        self.max_reconnect_delay,
                    )
                    try:
                        await asyncio.wait_for(self._shutdown.wait(), timeout=delay)
                        break
                    except TimeoutError:
                        pass
        finally:
            for server in self._mcp_servers:
                await server.disconnect()
            await self._client.close()
            logger.info("Runner stopped")

    def _capabilities(self) -> list[str]:
        if self._explicit_caps is not None:
            return list(self._explicit_caps)
        caps = [t.tool_id for t in all_tools()]
        for server in self._mcp_servers:
            for fn in server._tools or []:
                tid = getattr(fn, "__rebuno_tool_id__", None)
                if tid:
                    caps.append(tid)
        return list(dict.fromkeys(caps))  # de-dupe, preserve order

    def _build_schemas(self) -> list[dict[str, Any]]:
        """Build the schema list for every tool this runner can service."""
        schemas: list[dict[str, Any]] = []
        seen: set[str] = set()

        for entry in all_tools():
            if entry.tool_id in seen:
                continue
            seen.add(entry.tool_id)
            schemas.append(
                {
                    "id": entry.tool_id,
                    "description": (entry.fn.__doc__ or "").strip(),
                    "input_schema": fn_to_json_schema(entry.fn),
                }
            )

        for server in self._mcp_servers:
            for fn in server._tools or []:
                tool_id = getattr(fn, "__rebuno_tool_id__", None)
                if not tool_id or tool_id in seen:
                    continue
                seen.add(tool_id)
                schemas.append(
                    {
                        "id": tool_id,
                        "description": (fn.__doc__ or "").strip(),
                        "input_schema": getattr(fn, "__rebuno_input_schema__", {}) or {},
                    }
                )

        return schemas

    async def _publish_schemas(self) -> None:
        schemas = self._build_schemas()
        if not schemas:
            logger.debug("Runner has no tools to publish")
            return
        await self._client.publish_tools(self.runner_id, schemas)
        logger.info("Published %d tool schemas to kernel", len(schemas))

    def _handle_shutdown(self) -> None:
        logger.info("Shutdown signal received")
        self._running = False
        self._shutdown.set()
        if self._connect_task and not self._connect_task.done():
            self._connect_task.cancel()

    async def _stream_loop(self, capabilities: list[str]) -> None:
        logger.info("Runner SSE connection established")
        async for sse in self._client.runner_stream(
            self.runner_id,
            self.consumer_id,
            capabilities,
        ):
            if not self._running:
                return
            if sse.type == "job.assigned":
                job = Job(**json.loads(sse.data))
                asyncio.create_task(self._handle_job(job))

    async def _handle_job(self, job: Job) -> None:
        logger.info(
            "Job received: job_id=%s tool_id=%s step_id=%s",
            job.id,
            job.tool_id,
            job.step_id,
        )
        try:
            await self._client.step_started(job.step_id, job.execution_id, self.runner_id)
        except Exception:
            logger.debug("step_started failed", exc_info=True)

        try:
            result = await self._dispatch(job.tool_id, job.arguments)
        except Exception as e:
            await self._submit_failure(job, e)
            return

        try:
            await self._client.submit_job_result(
                runner_id=self.runner_id,
                job_id=job.id,
                execution_id=job.execution_id,
                step_id=job.step_id,
                success=True,
                data=result,
            )
            logger.info("Job completed: job_id=%s", job.id)
        except Exception:
            logger.exception("Failed to submit success result: job_id=%s", job.id)

    async def _dispatch(self, tool_id: str, arguments: Any) -> Any:
        for server in self._mcp_servers:
            for fn in server._tools or []:
                if getattr(fn, "__rebuno_tool_id__", None) == tool_id:
                    kwargs = arguments if isinstance(arguments, dict) else {}
                    args = {k: v for k, v in kwargs.items() if v is not None}
                    raw = await server._client.call_tool(fn.__name__, args)
                    return _flatten_mcp_result(raw)

        entry = get_tool(tool_id)
        if entry is None:
            raise RuntimeError(f"No handler registered for tool '{tool_id}'")
        kwargs = arguments if isinstance(arguments, dict) else {}
        result = entry.fn(**kwargs)
        if asyncio.iscoroutine(result):
            result = await result
        return result

    async def _submit_failure(self, job: Job, exc: Exception) -> None:
        try:
            await self._client.submit_job_result(
                runner_id=self.runner_id,
                job_id=job.id,
                execution_id=job.execution_id,
                step_id=job.step_id,
                success=False,
                error=str(exc),
                retryable=getattr(exc, "retryable", False),
            )
        except Exception:
            logger.exception("Failed to submit failure result: job_id=%s", job.id)
        logger.warning("Job failed: job_id=%s error=%s", job.id, exc)
