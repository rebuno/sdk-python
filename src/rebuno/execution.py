from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
from collections.abc import Callable, Coroutine
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, TypeVar

from rebuno._internal.checkpoints import capture
from rebuno._kernel import DispatchLease
from rebuno.errors import (
    Blocked,
    LeaseSuperseded,
    PolicyError,
    RateLimited,
    RebunoError,
    Terminated,
    ToolError,
)
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
        lease: DispatchLease,
        agent_id: str,
        input: Any,
        status: str = "running",
    ):
        self._kernel = kernel
        self.id = execution_id
        self._lease = lease
        self.agent_id = agent_id
        self.input = input
        self.status = status
        self.suspension: Blocked | Terminated | None = None
        self._superseded = False
        self._resources: dict[str, dict[str, Any]] = {}
        self._effects = asyncio.Lock()
        self._effects_holder: str | None = None
        self._in_flight = 0
        self._waiting: list[asyncio.Future[bool]] = []
        try:
            self._loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

    @property
    def dispatch_id(self) -> str:
        return self._lease.dispatch_id

    @property
    def dispatch_attempt(self) -> int:
        return self._lease.attempt

    async def _on_owner_loop(self, coro: Coroutine[Any, Any, _T]) -> _T:
        """Await ``coro`` on the loop this context was created on, which is the
        one the kernel client's connections are bound to."""
        if self._loop is None or asyncio.get_running_loop() is self._loop:
            return await coro
        return await asyncio.wrap_future(
            asyncio.run_coroutine_threadsafe(coro, self._loop)
        )

    async def previous(self) -> Any:
        return await self._on_owner_loop(self._kernel.previous_state(self.id))

    async def _heartbeat_loop(self, owner: asyncio.Task | None) -> None:
        while True:
            await asyncio.sleep(self._lease.heartbeat_interval)
            try:
                await self._kernel.heartbeat(self.id, lease=self._lease)
            except LeaseSuperseded:
                self._superseded = True
                if owner is not None:
                    owner.cancel()
                return
            except Exception:
                logger.warning("dispatch heartbeat failed", exc_info=True)

    @contextlib.asynccontextmanager
    async def _exclusive(self, needed: bool = True):
        """Tools may run on another loop, so the lock is taken and released on
        the owner loop."""
        if not needed:
            yield
            return
        await self._on_owner_loop(self._effects.acquire())
        try:
            yield
        finally:
            self._release_effects()

    def _release_effects(self) -> None:
        self._effects_holder = None
        if self._loop is None or asyncio.get_running_loop() is self._loop:
            self._effects.release()
        else:
            self._loop.call_soon_threadsafe(self._effects.release)

    async def _call_started(self) -> None:
        self._in_flight += 1

    async def _call_finished(self) -> None:
        self._in_flight -= 1
        await self._suspend_if_idle()

    async def _await_idle(self) -> bool:
        """Resolves once every in-flight call waits, with whether the execution
        suspended."""
        waiter = asyncio.get_running_loop().create_future()
        self._waiting.append(waiter)
        await self._suspend_if_idle()
        return await waiter

    async def _suspend_if_idle(self) -> None:
        if not self._waiting or len(self._waiting) < self._in_flight:
            return
        waiters, self._waiting = self._waiting, []
        try:
            suspended = self.suspension is not None or await self._kernel.suspend(
                self.id, lease=self._lease
            )
        except Exception as e:
            for w in waiters:
                w.set_exception(e)
        else:
            for w in waiters:
                w.set_result(suspended)
        finally:
            for w in waiters:
                if not w.done():
                    w.cancel()

    async def await_subagent(self, step_id: str) -> Any:
        """The call's effects lock is released while it waits, so concurrent
        calls can start their own subagents, and taken back before its step
        completes."""
        held = self._effects_holder == step_id
        if held:
            self._release_effects()
        try:
            if await self._on_owner_loop(self._await_idle()):
                self.suspension = self.suspension or Blocked()
                raise self.suspension
        finally:
            if held:
                await self._on_owner_loop(self._effects.acquire())
                self._effects_holder = step_id
        step = await self._on_owner_loop(self._kernel.get_step(self.id, step_id))
        if step.error is not None:
            raise ToolError(
                _error_message(step.error), tool_id=step.target, step_id=step_id
            )
        return step.result

    async def _submit(
        self,
        *,
        kind: str,
        target: str,
        args: Any,
        idempotency: str,
        resources: list[str] | None = None,
    ) -> tuple[str, StepDecision]:
        """The kernel counts occurrences of this effect under its own lock, so
        concurrent identical calls get distinct step ids without coordination here.
        """
        declared = {} if resources is None else {"resources": resources}
        dec = await self._on_owner_loop(
            self._kernel.submit_step(
                self.id,
                lease=self._lease,
                kind=kind,
                target=target,
                args=args,
                idempotency=idempotency,
                **declared,
            )
        )
        return dec.step_id, dec

    def _raise_for_decision(self, dec: StepDecision) -> None:
        """Returns normally only for ``proceed``. ``replay`` carries an
        effect-specific result/error and is handled by the caller before this.
        """
        if dec.decision == "denied":
            raise PolicyError(dec.reason, rule_id=dec.rule_id)
        if dec.decision == "rate_limited":
            raise RateLimited(dec.reason)
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
        kind: str = "tool_call",
        resources: list[str] | None = None,
    ) -> Any:
        """Submit a step and, if the kernel says proceed, run the body.

        ``run`` is called with no arguments: callers close over whatever
        inputs the body needs. ``args`` is only the JSON-recorded payload
        used for step identity/hashing, not ``run``'s call signature.

        ``kind`` is the step kind the kernel records and policy matches on.

        ``resources`` names the registered resources the body may change.
        Defaults to none.
        """
        await self._on_owner_loop(self._call_started())
        try:
            return await self._invoke_tool(
                target, args, idempotency, run, kind, resources
            )
        finally:
            await self._on_owner_loop(self._call_finished())

    async def _invoke_tool(
        self,
        target: str,
        args: dict[str, Any],
        idempotency: str,
        run: Callable[[], Any] | None,
        kind: str,
        resources: list[str] | None,
    ) -> Any:
        locked = bool(self._resources)
        async with self._exclusive(locked):
            step_id, dec = await self._submit(
                kind=kind,
                target=target,
                args=args,
                idempotency=idempotency,
                resources=resources,
            )
            if locked:
                self._effects_holder = step_id
            due = [r for r in dec.resources if r.due]

            if dec.decision == "replay":
                if dec.error is not None:
                    raise ToolError(
                        _error_message(dec.error), tool_id=target, step_id=step_id
                    )
                return dec.result
            self._raise_for_decision(dec)

            token = _current_step.set(step_id)
            try:
                result = run() if run is not None else None
                if inspect.isawaitable(result):
                    result = await result
            except (Blocked, Terminated, PolicyError, RateLimited, LeaseSuperseded):
                raise
            except Exception as e:
                captures = await capture(self, due)
                await self._fail_step_quietly(step_id, e, **captures)
                if isinstance(e, ToolError):
                    raise
                raise ToolError(str(e), tool_id=target, step_id=step_id) from e
            finally:
                _current_step.reset(token)
            captures = await capture(self, due)
            await self._on_owner_loop(
                self._kernel.complete_step(
                    self.id, step_id, lease=self._lease, result=result, **captures
                )
            )
            return result

    async def begin_llm(self, target: str, request: Any) -> tuple[str, StepDecision]:
        """Submit an ``llm_call`` step and return ``(step_id, decision)``.

        The decision is ``proceed`` (run the provider call and record it via
        :meth:`record_llm`) or ``replay`` (rebuild the response from
        ``decision.result``). Any other decision raises the matching control-flow
        error.
        """
        step_id, dec = await self._submit(
            kind="llm_call", target=target, args=request, idempotency="safe_to_retry"
        )
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
            await self._on_owner_loop(
                self._kernel.stream_delta(
                    self.id, step_id, lease=self._lease, seq=seq, data=data
                )
            )
        except Exception:
            logger.debug(
                "stream delta publish failed for step_id=%s", step_id, exc_info=True
            )

    async def record_llm(self, step_id: str, result: Any) -> None:
        await self._on_owner_loop(
            self._kernel.complete_step(
                self.id, step_id, lease=self._lease, result=result
            )
        )

    def start_heartbeat(self) -> asyncio.Task:
        """The caller must cancel the returned task when the effect finishes.

        Losing the lease cancels the task that started the heartbeat, so a
        handler the kernel has replaced stops instead of working on."""
        return asyncio.create_task(self._heartbeat_loop(asyncio.current_task()))

    @contextlib.asynccontextmanager
    async def lease(self):
        """A blocking sync body starves the heartbeat: the block must yield to
        the event loop, or the kernel reclaims the dispatch mid-handler.
        """
        hb = self.start_heartbeat()
        try:
            yield
        except asyncio.CancelledError:
            if self._superseded:
                raise LeaseSuperseded from None
            raise
        finally:
            hb.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await hb

    async def _fail_step_quietly(
        self, step_id: str, error: Exception, **captures: Any
    ) -> None:
        try:
            await self._on_owner_loop(
                self._kernel.fail_step(
                    self.id,
                    step_id,
                    lease=self._lease,
                    error={"message": str(error)},
                    **captures,
                )
            )
        except LeaseSuperseded:
            raise
        except Exception:
            logger.exception("failed to record step failure for step_id=%s", step_id)


def _error_message(error: dict[str, Any]) -> str:
    return str(error.get("message") or error.get("reason") or error)


_current: ContextVar[ExecutionContext | None] = ContextVar(
    "rebuno_execution", default=None
)
_current_step: ContextVar[str | None] = ContextVar("rebuno_step", default=None)


class _ExecutionAccessor:
    __slots__ = ()

    def __call__(self) -> ExecutionContext:
        state = _current.get()
        if state is None:
            raise RuntimeError("execution() called without an active execution context")
        return state

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(f"rebuno.execution must be called: use execution().{name}")


execution = _ExecutionAccessor()


@dataclass
class Result:
    output: Any = None
    state: Any = None


async def previous() -> Any:
    return await execution().previous()


def _set_current(state: ExecutionContext | None) -> Any:
    return _current.set(state)


def _reset_current(token: Any) -> None:
    _current.reset(token)


def _get_current() -> ExecutionContext | None:
    return _current.get()
