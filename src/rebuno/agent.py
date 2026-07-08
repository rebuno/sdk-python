from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response

from rebuno._internal import InputBinder
from rebuno._kernel import KernelClient
from rebuno.errors import Blocked, PolicyError, RateLimited, StepIDMismatch, Terminated, ToolError
from rebuno.execution import ExecutionContext, _reset_current, _set_current

logger = logging.getLogger("rebuno.agent")


class Agent:
    """A webhook-driven consumer of executions for a single agent_id."""

    def __init__(
        self,
        agent_id: str,
        *,
        secret: str | None = None,
        kernel_url: str | None = None,
        webhook_path: str = "/webhook",
        kernel_timeout: float = 35.0,
    ):
        if not agent_id:
            raise ValueError("agent_id must not be empty")
        self.agent_id = agent_id
        self.secret = secret if secret is not None else os.environ.get("REBUNO_AGENT_SECRET", "")
        if not self.secret:
            raise ValueError("secret required (set REBUNO_AGENT_SECRET or pass secret=)")
        self.kernel_url = (kernel_url or os.environ.get("REBUNO_URL", "")).rstrip("/")
        if not self.kernel_url:
            raise ValueError("kernel_url required (set REBUNO_URL or pass kernel_url=)")
        self.webhook_path = webhook_path
        self._process: Callable[..., Any] | None = None
        self._binder: InputBinder | None = None
        self._http = httpx.AsyncClient(base_url=self.kernel_url, timeout=kernel_timeout)
        self._kernel = KernelClient(agent_id=agent_id, secret=self.secret, http=self._http)
        self._app: FastAPI | None = None
        self._closed = False

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
            payload = _safe_json(raw)
            execution_id = (payload or {}).get("execution_id")
            if not execution_id:
                return Response(status_code=400)
            try:
                await self._handle(execution_id)
                return Response(status_code=200)
            except (Blocked, Terminated):
                return Response(status_code=200)
            except Exception:
                logger.exception("internal error handling execution %s", execution_id)
                return Response(status_code=500)

        return app

    def _verify(self, raw: bytes, header: str) -> bool:
        if not header.startswith("sha256="):
            return False
        expected = hmac.new(self.secret.encode(), raw, hashlib.sha256).hexdigest()
        return hmac.compare_digest(header[len("sha256=") :], expected)

    async def _handle(self, execution_id: str) -> None:
        assert self._process is not None and self._binder is not None
        exec = await self._kernel.get_execution(execution_id)
        if exec.status in ("completed", "failed", "cancelled"):
            return

        ctx = ExecutionContext(
            kernel=self._kernel,
            execution_id=execution_id,
            agent_id=self.agent_id,
            input=exec.input,
            status=exec.status,
        )
        # Preload prior terminal steps in one read so re-dispatch replays from a
        # local map instead of a kernel round trip per step.
        await ctx.hydrate()
        token = _set_current(ctx)
        try:
            try:
                kwargs = self._binder.bind(exec.input)
            except ValueError as e:
                await self._kernel.fail_execution(execution_id, error=str(e))
                return
            try:
                output = self._process(**kwargs)
                if hasattr(output, "__await__"):
                    output = await output
            except (Blocked, Terminated):
                raise
            except (PolicyError, ToolError, RateLimited, StepIDMismatch) as e:
                await self._kernel.fail_execution(execution_id, error=str(e))
                return
            except Exception as e:
                logger.exception("process error: execution_id=%s", execution_id)
                await self._kernel.fail_execution(execution_id, error=str(e))
                return
            await self._kernel.complete_execution(execution_id, output=output)
        finally:
            _reset_current(token)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._http.aclose()
        except RuntimeError:
            # The transport's event loop was already torn down (e.g. an abrupt
            # shutdown closed connections out from under us). Nothing left to do.
            logger.debug("ignoring transport error during close", exc_info=True)

    def run(self, process: Callable[..., Any], *, host: str = "0.0.0.0", port: int = 5000) -> None:
        """Bind the process and serve the webhook app with uvicorn (blocking)."""
        import asyncio

        import uvicorn

        self.bind(process)
        try:
            uvicorn.run(self.app, host=host, port=port)
        finally:
            asyncio.run(self.close())


def _safe_json(raw: bytes) -> dict[str, Any] | None:
    try:
        return json.loads(raw)
    except Exception:
        return None
