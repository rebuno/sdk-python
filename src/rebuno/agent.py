from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any, NamedTuple

import httpx2
from fastapi import FastAPI, Request, Response

from rebuno._internal import InputBinder
from rebuno._kernel import DispatchLease, KernelClient
from rebuno.errors import (
    Blocked,
    LeaseSuperseded,
    PolicyError,
    RateLimited,
    Terminated,
    ToolError,
    failure_reason,
    raise_for_refusal,
)
from rebuno.execution import ExecutionContext, _reset_current, _set_current

logger = logging.getLogger("rebuno.agent")


class _Running(NamedTuple):
    lease: DispatchLease
    task: asyncio.Task


class Agent:
    """A webhook-driven consumer of executions for a single agent_id."""

    def __init__(
        self,
        agent_id: str,
        *,
        secret: str | None = None,
        base_url: str | None = None,
        webhook_path: str = "/webhook",
        kernel_timeout: float = 35.0,
    ):
        if not agent_id:
            raise ValueError("agent_id must not be empty")
        self.agent_id = agent_id
        self.secret = (
            secret if secret is not None else os.environ.get("REBUNO_AGENT_SECRET", "")
        )
        if not self.secret:
            raise ValueError(
                "secret required (set REBUNO_AGENT_SECRET or pass secret=)"
            )
        self.base_url = (base_url or os.environ.get("REBUNO_URL", "")).rstrip("/")
        if not self.base_url:
            raise ValueError("base_url required (set REBUNO_URL or pass base_url=)")
        self.webhook_path = webhook_path
        self._process: Callable[..., Any] | None = None
        self._binder: InputBinder | None = None
        self._http = httpx2.AsyncClient(base_url=self.base_url, timeout=kernel_timeout)
        self._kernel = KernelClient(
            agent_id=agent_id, secret=self.secret, http=self._http
        )
        self._app: FastAPI | None = None
        self._closed = False
        self._tasks: dict[str, _Running] = {}
        self._superseded: set[asyncio.Task] = set()

    def bind(self, process: Callable[..., Any]) -> None:
        self._process = process
        self._binder = InputBinder(process)

    @property
    def app(self) -> FastAPI:
        if self._app is None:
            self._app = self._build_app()
        return self._app

    def _build_app(self) -> FastAPI:
        @asynccontextmanager
        async def lifespan(_: FastAPI) -> AsyncIterator[None]:
            # Close the kernel HTTP client during ASGI shutdown, on the same
            # event loop that opened its connections. Doing it here (rather than
            # on a fresh loop after the server stops) avoids touching transports
            # whose loop has already been torn down.
            yield
            await self.close()

        app = FastAPI(lifespan=lifespan)

        @app.post(self.webhook_path)
        async def webhook(request: Request) -> Response:
            raw = await request.body()
            sig = request.headers.get("Rebuno-Signature", "")
            if not self._verify(raw, sig):
                return Response(status_code=401)
            payload = _safe_json(raw) or {}
            execution_id = payload.get("execution_id")
            lease = _lease_from(payload)
            if not execution_id or lease is None:
                return Response(status_code=400)
            running = self._tasks.get(execution_id)
            if running is not None:
                if (
                    lease.dispatch_id == running.lease.dispatch_id
                    and lease.attempt <= running.lease.attempt
                ):
                    return Response(status_code=200)
                self._supersede(running.task)
            task = asyncio.create_task(self._safe_handle(execution_id, lease))
            self._tasks[execution_id] = _Running(lease, task)
            task.add_done_callback(lambda t: self._discard(execution_id, t))
            return Response(status_code=200)

        return app

    def _verify(self, raw: bytes, header: str) -> bool:
        if not header.startswith("sha256="):
            return False
        expected = hmac.new(self.secret.encode(), raw, hashlib.sha256).hexdigest()
        return hmac.compare_digest(header[len("sha256=") :], expected)

    async def _handle(self, execution_id: str, lease: DispatchLease) -> None:
        assert self._process is not None and self._binder is not None
        exec = await self._kernel.get_execution(execution_id)
        if exec.status in ("completed", "failed", "cancelled"):
            return

        ctx = ExecutionContext(
            kernel=self._kernel,
            execution_id=execution_id,
            lease=lease,
            agent_id=self.agent_id,
            input=exec.input,
            status=exec.status,
        )
        token = _set_current(ctx)
        try:
            try:
                kwargs = self._binder.bind(exec.input)
            except ValueError as e:
                await self._kernel.fail_execution(
                    execution_id, lease=lease, error=f"input_invalid: {e}"
                )
                return
            try:
                async with ctx.lease():
                    output = self._process(**kwargs)
                    if hasattr(output, "__await__"):
                        output = await output
                if ctx.suspension is not None:
                    raise ctx.suspension
            except (Blocked, Terminated, LeaseSuperseded):
                raise
            except Exception as e:
                if ctx.suspension is not None:
                    raise ctx.suspension from e
                # Blocked and Terminated propagate; a denial or rate limit is
                # rebound onto e and fails the execution below.
                try:
                    raise_for_refusal(e)
                except (PolicyError, RateLimited) as refused:
                    e = refused
                if not isinstance(e, (PolicyError, ToolError, RateLimited)):
                    logger.exception("process error: execution_id=%s", execution_id)
                await self._kernel.fail_execution(
                    execution_id, lease=lease, error=failure_reason(e)
                )
                return
            await self._kernel.complete_execution(
                execution_id, lease=lease, output=output
            )
        finally:
            _reset_current(token)

    def _discard(self, execution_id: str, task: asyncio.Task) -> None:
        running = self._tasks.get(execution_id)
        if running is not None and running.task is task:
            del self._tasks[execution_id]

    def _supersede(self, task: asyncio.Task) -> None:
        """Cancel a replaced handler without waiting on it."""
        task.cancel()
        self._superseded.add(task)
        task.add_done_callback(self._superseded.discard)

    async def _safe_handle(self, execution_id: str, lease: DispatchLease) -> None:
        try:
            await self._handle(execution_id, lease)
        except (Blocked, Terminated, LeaseSuperseded):
            pass
        except Exception:
            logger.exception("unhandled error handling execution %s", execution_id)

    def _all_tasks(self) -> list[asyncio.Task]:
        return [r.task for r in self._tasks.values()] + list(self._superseded)

    async def join(self) -> None:
        """Wait for all in-flight execution handlers to finish (best-effort)."""
        tasks = self._all_tasks()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            self._tasks.clear()
            self._superseded.clear()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        tasks = self._all_tasks()
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception:
                logger.debug("ignoring task cleanup error during close", exc_info=True)
            self._tasks.clear()
            self._superseded.clear()
        try:
            await self._http.aclose()
        except RuntimeError:
            # The transport's event loop was already torn down (e.g. an abrupt
            # shutdown closed connections out from under us). Nothing left to do.
            logger.debug("ignoring transport error during close", exc_info=True)

    def run(
        self, process: Callable[..., Any], *, host: str = "0.0.0.0", port: int = 5000
    ) -> None:
        """Bind the process and serve the webhook app with uvicorn (blocking)."""
        import uvicorn

        self.bind(process)
        try:
            uvicorn.run(self.app, host=host, port=port)
        finally:
            asyncio.run(self.close())


def _safe_json(raw: bytes) -> dict[str, Any] | None:
    try:
        payload = json.loads(raw)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _lease_from(payload: dict[str, Any]) -> DispatchLease | None:
    """The lease a webhook carries, or None if it is unusable."""
    dispatch_id = payload.get("dispatch_id")
    attempt = payload.get("dispatch_attempt")
    if not dispatch_id or type(attempt) is not int or attempt <= 0:
        return None
    timeout = payload.get("lease_timeout_seconds")
    if type(timeout) not in (int, float) or timeout <= 0:
        return None
    return DispatchLease(dispatch_id, attempt, float(timeout))
