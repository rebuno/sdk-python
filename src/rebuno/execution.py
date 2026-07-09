from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

from rebuno.errors import Blocked, PolicyError, RateLimited, RebunoError, Terminated, ToolError
from rebuno.identity import args_hash, compute_step_id
from rebuno.types import Step, StepDecision

logger = logging.getLogger("rebuno.execution")


class ExecutionContext:
    """One per dispatch. Drives effect submission, replay, and occurrence counting."""

    def __init__(self, *, kernel: Any, execution_id: str, agent_id: str, input: Any, status: str = "running"):
        self._kernel = kernel
        self.id = execution_id
        self.agent_id = agent_id
        self.input = input
        self.status = status
        self._occurrences: dict[tuple[str, str, str], int] = {}
        self._replay: dict[str, Step] | None = None

    async def hydrate(self) -> None:
        """Preload this execution's terminal steps in one read so replay costs a
        single bulk fetch instead of one kernel round trip per replayed step.
        """
        try:
            steps = await self._kernel.list_terminal_steps(self.id)
        except Exception:
            logger.warning("step hydration failed; falling back to per-step replay", exc_info=True)
            self._replay = None
            return
        self._replay = {s.step_id: s for s in steps}

    async def _decide(self, *, kind: str, target: str, args: Any, idempotency: str, step_id: str) -> StepDecision:
        """Resolve a step decision: from the hydrated replay map when the step is
        already terminal, otherwise by asking the kernel.

        A map miss is not "new" — the step may be non-terminal (an orphan still
        ``executing``, or ``awaiting_approval``) and must re-hit the kernel so
        idempotency/approval logic runs. Only the kernel mints new steps.
        """
        if self._replay is not None:
            hit = self._replay.get(step_id)
            if hit is not None:
                return _decision_from_step(hit)
        return await self._kernel.submit_step(
            self.id, kind=kind, target=target, args=args, idempotency=idempotency, step_id=step_id
        )

    async def _run_with_heartbeat(self, run: Callable[[], Any], interval: float = 30.0) -> Any:
        """Run an effect body while a background task renews the dispatch lease, so a
        long-running but live body isn't reclaimed and double-invoked mid-step.

        The body must yield to the event loop (be async / await something) for the
        heartbeat to fire — a fully blocking sync body starves it. All the long
        effects here (LLM/provider calls, MCP tools) are I/O-bound and async, so
        this holds; wrap CPU-bound sync work in a thread if it ever doesn't.
        """
        hb = asyncio.create_task(self._heartbeat_loop(interval))
        try:
            result = run()
            if inspect.isawaitable(result):
                result = await result
            return result
        finally:
            hb.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await hb

    async def _heartbeat_loop(self, interval: float) -> None:
        while True:
            await asyncio.sleep(interval)
            try:
                await self._kernel.heartbeat(self.id)
            except Exception:
                logger.warning("dispatch heartbeat failed", exc_info=True)

    def _next_occurrence(self, kind: str, target: str, ah: str) -> int:
        key = (kind, target, ah)
        n = self._occurrences.get(key, 0)
        self._occurrences[key] = n + 1
        return n

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
            raise Blocked(dec.approval_id)
        if dec.decision == "execution_terminal":
            raise Terminated("execution is terminal")
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
        ah = args_hash(args)
        occ = self._next_occurrence(kind, target, ah)
        step_id = compute_step_id(self.id, kind, target, ah, occ)

        dec = await self._decide(kind=kind, target=target, args=args, idempotency=idempotency, step_id=step_id)

        if dec.decision == "replay":
            if dec.error is not None:
                raise ToolError(_error_message(dec.error), tool_id=target, step_id=step_id)
            return dec.result
        self._raise_for_decision(dec)

        # proceed: run the body, record the outcome.
        if run is None:
            await self._kernel.complete_step(self.id, step_id, result=None)
            return None
        try:
            result = await self._run_with_heartbeat(run)
        except (Blocked, Terminated, PolicyError, RateLimited):
            raise
        except Exception as e:
            await self._fail_step_quietly(step_id, e)
            if isinstance(e, ToolError):
                raise
            raise ToolError(str(e), tool_id=target, step_id=step_id) from e
        await self._kernel.complete_step(self.id, step_id, result=result)
        return result

    async def invoke_llm(self, target: str, request: Any, *, run: Callable[[], Any]) -> Any:
        """Submit an ``llm_call`` step; replay the recorded response or run the
        provider call and record it.

        ``request`` is the JSON request payload used for step identity/hashing
        (the same path tool calls use, with ``kind=llm_call``). ``run`` performs
        the provider call and returns a JSON-serializable response record.
        Returns that record — replayed from the log when available, otherwise
        fresh and newly recorded.
        """
        kind = "llm_call"
        ah = args_hash(request)
        occ = self._next_occurrence(kind, target, ah)
        step_id = compute_step_id(self.id, kind, target, ah, occ)

        dec = await self._decide(kind=kind, target=target, args=request, idempotency="safe_to_retry", step_id=step_id)

        if dec.decision == "replay":
            if dec.error is not None:
                raise RebunoError(_error_message(dec.error))
            return dec.result
        self._raise_for_decision(dec)

        # proceed: forward to the provider, record the response.
        try:
            result = await self._run_with_heartbeat(run)
        except (Blocked, Terminated, PolicyError, RateLimited):
            raise
        except Exception as e:
            await self._fail_step_quietly(step_id, e)
            raise
        await self._kernel.complete_step(self.id, step_id, result=result)
        return result

    async def _fail_step_quietly(self, step_id: str, error: Exception) -> None:
        try:
            await self._kernel.fail_step(self.id, step_id, error={"message": str(error)})
        except Exception:
            logger.exception("failed to record step failure for step_id=%s", step_id)


def _decision_from_step(step: Step) -> StepDecision:
    """Mirror the kernel's decision for an already-terminal step, so a hydrated
    replay is byte-for-byte what ``submit_step`` would have returned.

    Terminal statuses: ``succeeded``/``failed`` replay the recorded result/error;
    ``denied`` is a policy denial, not a replay. Anything else is unexpected for a
    terminal-filtered step and is sent back to the kernel via ``proceed``.
    """
    if step.status == "succeeded":
        return StepDecision(decision="replay", result=step.result)
    if step.status == "failed":
        return StepDecision(decision="replay", error=step.error)
    if step.status == "denied":
        return StepDecision(decision="denied", reason="policy_denied")
    return StepDecision(decision="proceed")


def _error_message(error: Any) -> str:
    if isinstance(error, dict):
        return str(error.get("message") or error.get("reason") or error)
    return str(error)


_current: ContextVar[ExecutionContext | None] = ContextVar("rebuno_execution", default=None)


class _ExecutionProxy:
    __slots__ = ()

    def _state(self) -> ExecutionContext:
        state = _current.get()
        if state is None:
            raise RuntimeError("execution.* accessed without an active execution context")
        return state

    def __getattr__(self, name: str) -> Any:
        return getattr(self._state(), name)


execution = _ExecutionProxy()


def _set_current(state: ExecutionContext | None) -> Any:
    return _current.set(state)


def _reset_current(token: Any) -> None:
    _current.reset(token)


def _get_current() -> ExecutionContext | None:
    return _current.get()
