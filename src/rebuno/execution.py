from __future__ import annotations

import asyncio
import inspect
import logging
import uuid
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from rebuno._internal.correlation import CorrelationMap
from rebuno.errors import PolicyError, RebunoError, ToolError
from rebuno.types import ClaimResult, HistoryEntry

if TYPE_CHECKING:
    from rebuno.client import Client

logger = logging.getLogger("rebuno.execution")


class ExecutionState:
    """Backs the `execution` proxy. One instance per assigned execution."""

    def __init__(
        self,
        client: Client,
        claim: ClaimResult,
        correlation: CorrelationMap,
        wait_timeout: float = 3600.0,
    ):
        self._client = client
        self._claim = claim
        self._correlation = correlation
        self._wait_timeout = wait_timeout
        self.id: str = claim.execution_id
        self.session_id: str = claim.session_id
        self.agent_id: str = claim.agent_id
        self.input: Any = claim.input
        self.labels: dict[str, str] = claim.labels
        self.history: list[HistoryEntry] = claim.history

    async def _wait(self, kind: str, key: str, timeout_msg: str) -> Any:
        fut = self._correlation.future(kind, key)
        try:
            return await asyncio.wait_for(fut, timeout=self._wait_timeout)
        except TimeoutError as e:
            raise RebunoError(timeout_msg) from e

    async def invoke_tool(
        self,
        tool_id: str,
        arguments: Any = None,
        *,
        idempotency_key: str = "",
        local_runner: Any = None,
    ) -> Any:
        """Submit an invoke_tool intent and await the result.

        If ``local_runner`` is provided it's a callable that executes the tool
        body locally after the kernel allows it. Otherwise the kernel routes
        the call to a runner (remote tool).
        """
        if not idempotency_key:
            idempotency_key = f"{self.id}:{tool_id}:{uuid.uuid4().hex[:8]}"

        result = await self._client.submit_intent(
            execution_id=self.id,
            session_id=self.session_id,
            intent_type="invoke_tool",
            tool_id=tool_id,
            arguments=arguments,
            idempotency_key=idempotency_key,
            remote=local_runner is None,
        )
        if not result.accepted:
            raise PolicyError(result.error or "Intent denied by policy")
        if not result.step_id:
            raise RebunoError("No step_id returned for invoke_tool intent")

        if result.pending_approval:
            approval = await self._wait(
                "approval",
                result.step_id,
                f"Timed out waiting for approval (step_id={result.step_id})",
            )
            if not approval.get("approved", False):
                raise PolicyError("Tool invocation denied by human approval")

        if local_runner is not None:
            return await self._execute_local(result.step_id, tool_id, arguments, local_runner)

        data = await self._wait(
            "result",
            result.step_id,
            f"Timed out waiting for tool result (step_id={result.step_id})",
        )
        if data.get("status") == "failed":
            raise ToolError(
                message=data.get("error", "Tool execution failed"),
                tool_id=tool_id,
                step_id=result.step_id,
            )
        return data.get("result")

    async def _execute_local(
        self,
        step_id: str,
        tool_id: str,
        arguments: Any,
        fn: Any,
    ) -> Any:
        kwargs = arguments if isinstance(arguments, dict) else {}
        try:
            output = fn(**kwargs)
            if inspect.isawaitable(output):
                output = await output
        except Exception as e:
            await self._client.report_step_result(
                execution_id=self.id,
                session_id=self.session_id,
                step_id=step_id,
                success=False,
                error=str(e),
            )
            if isinstance(e, ToolError):
                raise
            raise ToolError(message=str(e), tool_id=tool_id, step_id=step_id) from e

        await self._client.report_step_result(
            execution_id=self.id,
            session_id=self.session_id,
            step_id=step_id,
            success=True,
            data=output,
        )
        return output

    async def wait_signal(self, signal_type: str) -> Any:
        """Wait for a signal of the given type."""
        result = await self._client.submit_intent(
            execution_id=self.id,
            session_id=self.session_id,
            intent_type="wait",
            signal_type=signal_type,
        )
        if not result.accepted:
            raise PolicyError(result.error or "Wait intent denied")
        return await self._wait(
            "signal",
            signal_type,
            f"Timed out waiting for signal: {signal_type}",
        )

    async def complete(self, output: Any = None) -> None:
        """Mark the execution as completed. Usually unnecessary — return from the
        handler instead."""
        await self._client.submit_intent(
            execution_id=self.id,
            session_id=self.session_id,
            intent_type="complete",
            output=output,
        )

    async def fail(self, error: str) -> None:
        """Mark the execution as failed. Usually unnecessary — raise from the
        handler instead."""
        await self._client.submit_intent(
            execution_id=self.id,
            session_id=self.session_id,
            intent_type="fail",
            error=error,
        )


_current: ContextVar[ExecutionState | None] = ContextVar("rebuno_execution", default=None)


class _ExecutionProxy:
    """Module-level accessor that resolves to the current ExecutionState."""

    __slots__ = ()

    def _state(self) -> ExecutionState:
        state = _current.get()
        if state is None:
            raise RuntimeError(
                "execution.* accessed outside an active agent execution. "
                "This proxy is only valid inside @tool functions or handlers "
                "running under agent.run()."
            )
        return state

    def __getattr__(self, name: str) -> Any:
        return getattr(self._state(), name)


execution = _ExecutionProxy()


def _set_current(state: ExecutionState | None) -> Any:
    return _current.set(state)


def _reset_current(token: Any) -> None:
    _current.reset(token)


def _get_current() -> ExecutionState | None:
    return _current.get()
