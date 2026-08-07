from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
from collections.abc import Callable, Coroutine
from contextvars import ContextVar
from typing import Any, TypeVar

from rebuno.errors import Blocked, PolicyError, RateLimited, RebunoError, Terminated, ToolError
from rebuno.types import StepDecision

logger = logging.getLogger("rebuno.execution")

_T = TypeVar("_T")


class ExecutionContext:
    """One per dispatch. Submits effects to the kernel and applies its decisions."""

    def __init__(
        self,
        *,
        kernel: Any,
        execution_id: str,
        dispatch_id: str,
        agent_id: str,
        input: Any,
        status: str = "running",
    ):
        self._kernel = kernel
        self.id = execution_id
        self.dispatch_id = dispatch_id
        self.agent_id = agent_id
        self.input = input
        self.status = status
        # The Blocked or Terminated this context raised, if any.
        self.suspension: Blocked | Terminated | None = None
        try:
            self._loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

    async def _on_owner_loop(self, coro: Coroutine[Any, Any, _T]) -> _T:
        """Await ``coro`` on the loop this context was created on, which is the
        one the kernel client's connections are bound to."""
        if self._loop is None or asyncio.get_running_loop() is self._loop:
            return await coro
        return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(coro, self._loop))

    async def _heartbeat_loop(self, interval: float) -> None:
        while True:
            await asyncio.sleep(interval)
            try:
                await self._kernel.heartbeat(self.id)
            except Exception:
                logger.warning("dispatch heartbeat failed", exc_info=True)

    async def _submit(self, *, kind: str, target: str, args: Any, idempotency: str) -> tuple[str, StepDecision]:
        """Ask the kernel to decide this effect, and return ``(step_id, decision)``.

        The kernel assigns the step id: it counts occurrences of this effect within
        the dispatch under its own lock, so concurrent identical calls get distinct
        steps without any coordination here. ``step_id`` is empty for decisions that
        recorded no step (``rate_limited``, ``execution_*``), which
        :meth:`_raise_for_decision` turns into an exception before it is used.
        """
        dec = await self._on_owner_loop(
            self._kernel.submit_step(
                self.id,
                dispatch_id=self.dispatch_id,
                kind=kind,
                target=target,
                args=args,
                idempotency=idempotency,
            )
        )
        return dec.step_id, dec

    def _raise_for_decision(self, dec: StepDecision) -> None:
        """Map a non-proceed step decision to its control-flow exception.

        Returns normally only for ``proceed``. ``replay`` carries an
        effect-specific result/error and is handled by the caller before this.
        """
        if dec.decision == "denied":
            raise PolicyError(dec.reason or "denied by policy")
        if dec.decision == "rate_limited":
            raise RateLimited(dec.reason or "rate_limit_exceeded")
        if dec.decision in ("blocked", "execution_blocked"):
            self.suspension = Blocked()
            raise self.suspension
        if dec.decision == "execution_terminal":
            self.suspension = Terminated("execution is terminal")
            raise self.suspension
        if dec.decision != "proceed":
            raise RebunoError(f"unexpected step decision: {dec.decision}")

    async def invoke_tool(
        self,
        target: str,
        args: dict[str, Any],
        *,
        idempotency: str = "safe_to_retry",
        run: Callable[[], Any] | None = None,
    ) -> Any:
        """Submit a step and, if the kernel says proceed, run the body.

        ``run`` is called with no arguments — callers close over whatever
        inputs the body needs. ``args`` is only the JSON-recorded payload
        used for step identity/hashing, not ``run``'s call signature.
        """
        kind = "tool_call"
        step_id, dec = await self._submit(kind=kind, target=target, args=args, idempotency=idempotency)

        if dec.decision == "replay":
            if dec.error is not None:
                raise ToolError(_error_message(dec.error), tool_id=target, step_id=step_id)
            return dec.result
        self._raise_for_decision(dec)

        # proceed: run the body, record the outcome.
        if run is None:
            await self._on_owner_loop(self._kernel.complete_step(self.id, step_id, result=None))
            return None
        try:
            result = run()
            if inspect.isawaitable(result):
                result = await result
        except (Blocked, Terminated, PolicyError, RateLimited):
            raise
        except Exception as e:
            await self._fail_step_quietly(step_id, e)
            if isinstance(e, ToolError):
                raise
            raise ToolError(str(e), tool_id=target, step_id=step_id) from e
        await self._on_owner_loop(self._kernel.complete_step(self.id, step_id, result=result))
        return result

    async def begin_llm(self, target: str, request: Any) -> tuple[str, StepDecision]:
        """Submit an ``llm_call`` step and return ``(step_id, decision)``.

        The decision is ``proceed`` (run the provider call and record it via
        :meth:`record_llm`) or ``replay`` (rebuild the response from
        ``decision.result``). Any other decision raises the matching control-flow
        error.
        """
        step_id, dec = await self._submit(kind="llm_call", target=target, args=request, idempotency="safe_to_retry")
        if dec.decision == "replay":
            if dec.error is not None:
                raise RebunoError(_error_message(dec.error))
            return step_id, dec
        self._raise_for_decision(dec)
        return step_id, dec

    async def publish_llm_delta(self, step_id: str, seq: int, data: str) -> None:
        """Publish a live delta for an in-flight streamed step. Best-effort:
        failures are logged and swallowed."""
        try:
            await self._on_owner_loop(self._kernel.stream_delta(self.id, step_id, seq=seq, data=data))
        except Exception:
            logger.debug("stream delta publish failed for step_id=%s", step_id, exc_info=True)

    async def record_llm(self, step_id: str, result: Any) -> None:
        """Record the assembled streamed response as the step's durable result."""
        await self._on_owner_loop(self._kernel.complete_step(self.id, step_id, result=result))

    def start_heartbeat(self, interval: float = 30.0) -> asyncio.Task:
        """Start a background lease-renewal task and return it. The caller must
        cancel it when the effect finishes."""
        return asyncio.create_task(self._heartbeat_loop(interval))

    @contextlib.asynccontextmanager
    async def lease(self, interval: float = 30.0):
        """Renew the dispatch lease for the duration of the block, so the kernel
        doesn't reclaim the dispatch and re-deliver it to a second handler.

        The block must yield to the event loop for the heartbeat to fire — a fully
        blocking sync body starves it. Everything long in a handler (LLM/provider
        calls, MCP tools, kernel round-trips) is I/O-bound and async, so this
        holds; wrap CPU-bound sync work in a thread if it ever doesn't.
        """
        hb = self.start_heartbeat(interval)
        try:
            yield
        finally:
            hb.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await hb

    async def _fail_step_quietly(self, step_id: str, error: Exception) -> None:
        try:
            await self._on_owner_loop(self._kernel.fail_step(self.id, step_id, error={"message": str(error)}))
        except Exception:
            logger.exception("failed to record step failure for step_id=%s", step_id)


def _error_message(error: Any) -> str:
    if isinstance(error, dict):
        return str(error.get("message") or error.get("reason") or error)
    return str(error)


_current: ContextVar[ExecutionContext | None] = ContextVar("rebuno_execution", default=None)


class _ExecutionAccessor:
    __slots__ = ()

    def __call__(self) -> ExecutionContext:
        state = _current.get()
        if state is None:
            raise RuntimeError("execution() called without an active execution context")
        return state

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(f"rebuno.execution is called, not read — use execution().{name}")


execution = _ExecutionAccessor()


def _set_current(state: ExecutionContext | None) -> Any:
    return _current.set(state)


def _reset_current(token: Any) -> None:
    _current.reset(token)


def _get_current() -> ExecutionContext | None:
    return _current.get()
