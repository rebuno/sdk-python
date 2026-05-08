from __future__ import annotations

import asyncio
import inspect
import json
import logging
import uuid
from collections.abc import Callable
from typing import Any

from rebuno._internal import install_shutdown_handlers, jittered_backoff
from rebuno._internal.correlation import CorrelationMap
from rebuno._internal.inputs import InputBinder
from rebuno._internal.sse import SSEEvent
from rebuno.client import Client
from rebuno.errors import ClientClosedError, PolicyError, ToolError, UnauthorizedError
from rebuno.execution import ExecutionState, _reset_current, _set_current
from rebuno.mcp import connect_all as connect_all_mcp
from rebuno.mcp import disconnect_all as disconnect_all_mcp
from rebuno.remote import connect_all as connect_all_remote
from rebuno.remote import disconnect_all as disconnect_all_remote
from rebuno.remote import refresh_all as refresh_all_remote
from rebuno.types import ClaimResult

logger = logging.getLogger("rebuno.agent")


def _agent_logger(agent_id: str) -> logging.Logger:
    # Dots in agent_id would create unintended logger hierarchy levels.
    safe = agent_id.replace(".", "-")
    return logger.getChild(safe)


class Agent:
    """A long-lived consumer of executions for a single agent_id.

    Args:
        agent_id: The agent ID to register as.
        kernel_url: Override REBUNO_URL env var.
        api_key: Override REBUNO_API_KEY env var.
        consumer_id: Unique consumer identifier (auto-generated if empty).
        reconnect_delay: Initial reconnect delay in seconds.
        max_reconnect_delay: Cap on backoff delay.
    """

    def __init__(
        self,
        agent_id: str,
        *,
        kernel_url: str | None = None,
        api_key: str | None = None,
        consumer_id: str = "",
        reconnect_delay: float = 3.0,
        max_reconnect_delay: float = 60.0,
    ):
        if not agent_id:
            raise ValueError("agent_id must not be empty")
        self.agent_id = agent_id
        self.consumer_id = consumer_id or f"{agent_id}-{uuid.uuid4().hex[:8]}"
        self._logger = _agent_logger(agent_id)
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay

        self._client = Client(base_url=kernel_url, api_key=api_key)
        self._handler: Callable[..., Any] | None = None
        self._binder: InputBinder | None = None
        self._running = False
        self._shutdown = asyncio.Event()
        self._connect_task: asyncio.Task[None] | None = None
        self._exec_correlation: dict[str, CorrelationMap] = {}
        self._exec_tasks: set[asyncio.Task[None]] = set()
        self._shutdown_drain_timeout: float = 5.0

    def run(self, handler: Callable[..., Any]) -> None:
        """Block on the agent loop. Convenience wrapper over ``asyncio.run``."""
        asyncio.run(self.run_async(handler))

    async def run_async(self, handler: Callable[..., Any]) -> None:
        """Run the agent loop in the current event loop."""
        self._handler = handler
        self._binder = InputBinder(handler)
        self._running = True
        self._shutdown.clear()

        install_shutdown_handlers(self._handle_shutdown)

        self._logger.info(
            "Agent started: agent_id=%s consumer_id=%s",
            self.agent_id,
            self.consumer_id,
        )

        # Best-effort MCP + remote startup; failures don't block agent boot.
        try:
            await connect_all_mcp()
        except Exception:
            self._logger.exception("MCP startup error")
        try:
            await connect_all_remote(self._client)
        except Exception:
            self._logger.exception("Remote tools startup error")

        consecutive_failures = 0
        try:
            while self._running:
                try:
                    self._connect_task = asyncio.ensure_future(self._stream_loop())
                    await self._connect_task
                    consecutive_failures = 0
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    consecutive_failures += 1
                    self._logger.warning(
                        "SSE connection dropped, reconnecting: %s: %s",
                        type(e).__name__,
                        e,
                    )
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
            await self._drain_exec_tasks()
            await disconnect_all_mcp()
            await disconnect_all_remote()
            await self._client.close()
            self._logger.info("Agent stopped")

    async def _drain_exec_tasks(self) -> None:
        """Wait for in-flight execution tasks to finish before closing the client.

        Gives them a brief grace period to complete their terminal intent, then
        cancels any stragglers and awaits cancellation. Without this, the
        client closes underneath running tasks and they crash trying to submit
        their final intent.
        """
        if not self._exec_tasks:
            return
        pending = list(self._exec_tasks)
        self._logger.info("Draining %d in-flight execution task(s)", len(pending))
        done, still_pending = await asyncio.wait(
            pending, timeout=self._shutdown_drain_timeout
        )
        if still_pending:
            self._logger.warning(
                "Cancelling %d execution task(s) that did not finish within %.1fs",
                len(still_pending),
                self._shutdown_drain_timeout,
            )
            for t in still_pending:
                t.cancel()
            await asyncio.gather(*still_pending, return_exceptions=True)

    def _handle_shutdown(self) -> None:
        self._logger.info("Shutdown signal received")
        self._running = False
        self._shutdown.set()
        if self._connect_task and not self._connect_task.done():
            self._connect_task.cancel()

    async def _stream_loop(self) -> None:
        self._logger.info("SSE connection established")
        async for sse in self._client.agent_stream(self.agent_id, self.consumer_id):
            if not self._running:
                return
            await self._dispatch(sse)

    async def _dispatch(self, sse: SSEEvent) -> None:
        if sse.type == "execution.assigned":
            data = json.loads(sse.data)
            claim = ClaimResult(**data)
            task = asyncio.create_task(self._handle_execution(claim))
            self._exec_tasks.add(task)
            task.add_done_callback(self._exec_tasks.discard)
            task.add_done_callback(self._log_task_exception)
            return

        data = json.loads(sse.data)
        execution_id = data.get("execution_id", "")
        correlation = self._exec_correlation.get(execution_id)
        if correlation is None:
            return

        if sse.type == "tool.result":
            correlation.resolve("result", data.get("step_id", ""), data)
        elif sse.type == "signal.received":
            correlation.resolve("signal", data.get("signal_type", ""), data.get("payload"))
        elif sse.type == "approval.resolved":
            correlation.resolve("approval", data.get("step_id", ""), data)

    async def _handle_execution(self, claim: ClaimResult) -> None:
        eid = claim.execution_id
        self._logger.info(
            "Execution assigned: execution_id=%s session_id=%s",
            eid,
            claim.session_id,
        )

        correlation = CorrelationMap()
        self._exec_correlation[eid] = correlation
        state = ExecutionState(self._client, claim, correlation)

        token = _set_current(state)
        try:
            await refresh_all_remote(self._client)

            assert self._binder is not None
            try:
                kwargs = self._binder.bind(claim.input)
            except ValueError as e:
                await self._fail(eid, claim.session_id, str(e))
                return

            try:
                output = self._handler(**kwargs) if self._handler else None
                if inspect.isawaitable(output):
                    output = await output
            except (PolicyError, ToolError) as e:
                self._logger.warning("Execution failed: execution_id=%s error=%s", eid, e)
                await self._fail(eid, claim.session_id, str(e))
                return
            except Exception as e:
                self._logger.exception("Process error: execution_id=%s", eid)
                await self._fail(eid, claim.session_id, str(e))
                return

            try:
                await self._client.submit_intent(
                    execution_id=eid,
                    session_id=claim.session_id,
                    intent_type="complete",
                    output=output,
                )
                self._logger.info("Execution completed: execution_id=%s", eid)
            except ClientClosedError:
                self._logger.info(
                    "Client closed before completion intent could be submitted; "
                    "agent is shutting down: execution_id=%s",
                    eid,
                )
            except UnauthorizedError as e:
                if _is_session_gone(e):
                    self._logger.warning(
                        "Session expired before completion; kernel will reassign: execution_id=%s",
                        eid,
                    )
                else:
                    raise
        finally:
            _reset_current(token)
            correlation.cancel_all()
            self._exec_correlation.pop(eid, None)

    async def _fail(self, execution_id: str, session_id: str, error: str) -> None:
        try:
            await self._client.submit_intent(
                execution_id=execution_id,
                session_id=session_id,
                intent_type="fail",
                error=error,
            )
        except ClientClosedError:
            self._logger.info(
                "Client closed before fail intent could be submitted; "
                "agent is shutting down: execution_id=%s original_error=%s",
                execution_id,
                error,
            )
        except UnauthorizedError as e:
            if _is_session_gone(e):
                self._logger.warning(
                    "Session expired before fail intent could be submitted: "
                    "execution_id=%s original_error=%s",
                    execution_id,
                    error,
                )
            else:
                self._logger.exception("Failed to submit fail intent: execution_id=%s", execution_id)
        except Exception:
            self._logger.exception("Failed to submit fail intent: execution_id=%s", execution_id)

    def _log_task_exception(self, task: asyncio.Task[Any]) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            self._logger.error("Unhandled exception in execution task: %s", exc, exc_info=exc)


def _is_session_gone(e: UnauthorizedError) -> bool:
    msg = str(e).lower()
    return "session expired" in msg or "session not found" in msg
