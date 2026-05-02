from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import httpx

from rebuno._internal import SSEEvent, api_error, async_parse_sse
from rebuno.errors import APIError, NetworkError, PolicyError, RebunoError
from rebuno.types import (
    Event,
    Execution,
    ExecutionStatus,
    IntentResult,
    ListExecutionsResult,
    SignalResult,
)

_TERMINAL_EVENT_TYPES = frozenset(
    {
        "execution.completed",
        "execution.failed",
        "execution.cancelled",
    }
)

logger = logging.getLogger("rebuno.client")

USER_AGENT = "rebuno-python-sdk"


class Client:
    """Async HTTP client for the Rebuno kernel.

    Defaults to env vars REBUNO_URL and REBUNO_API_KEY. Pass explicit values
    to override.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        *,
        timeout: float = 35.0,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 10.0,
    ):
        url = base_url or os.environ.get("REBUNO_URL", "")
        if not url:
            raise ValueError("Client base_url is required (set REBUNO_URL or pass base_url=)")
        self.base_url = url.rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("REBUNO_API_KEY", "")
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay

        headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        self._http = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout,
        )

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> Client:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def create(
        self,
        agent_id: str,
        input: Any = None,
        labels: dict[str, str] | None = None,
    ) -> Execution:
        """Create a new execution for the given agent."""
        body: dict[str, Any] = {"agent_id": agent_id}
        if input is not None:
            body["input"] = input
        if labels:
            body["labels"] = labels
        resp = await self._request("POST", "/v0/executions", json=body)
        return Execution(**resp.json())

    async def get(self, execution_id: str) -> Execution:
        """Fetch the current state of an execution."""
        resp = await self._request("GET", f"/v0/executions/{execution_id}")
        return Execution(**resp.json())

    async def list(
        self,
        *,
        status: ExecutionStatus | None = None,
        agent_id: str = "",
        labels: dict[str, str] | None = None,
        limit: int = 50,
        cursor: str = "",
    ) -> ListExecutionsResult:
        """List executions with optional filters."""
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status.value
        if agent_id:
            params["agent_id"] = agent_id
        if labels:
            params["label"] = [f"{k}:{v}" for k, v in labels.items()]
        if cursor:
            params["cursor"] = cursor
        resp = await self._request("GET", "/v0/executions", params=params)
        return ListExecutionsResult(**resp.json())

    async def cancel(self, execution_id: str) -> Execution:
        """Cancel a running execution."""
        resp = await self._request(
            "POST",
            f"/v0/executions/{execution_id}/cancel",
            idempotent=True,
        )
        return Execution(**resp.json())

    async def send_signal(
        self,
        execution_id: str,
        signal_type: str,
        payload: Any = None,
    ) -> SignalResult:
        """Send a signal to a running execution."""
        body: dict[str, Any] = {"signal_type": signal_type}
        if payload is not None:
            body["payload"] = payload
        resp = await self._request("POST", f"/v0/executions/{execution_id}/signal", json=body)
        return SignalResult(**resp.json())

    async def events(
        self,
        execution_id: str,
        after_sequence: int = 0,
    ) -> AsyncIterator[Event]:
        """Stream execution events live, terminating when the execution does.

        Yields each Event as it arrives. The iterator ends after the first
        ``execution.{completed,failed,cancelled}`` event. For a completed
        execution it replays the backlog and exits at the terminal event.

        For a finite snapshot of an event log (no streaming), use
        :meth:`get_events` instead.
        """
        params: dict[str, Any] = {}
        if after_sequence:
            params["after_sequence"] = after_sequence
        async for sse in self._stream_sse(f"/v0/executions/{execution_id}/stream", params):
            event = Event(**json.loads(sse.data))
            yield event
            if event.type in _TERMINAL_EVENT_TYPES:
                return

    async def run(
        self,
        agent_id: str,
        input: Any = None,
        labels: dict[str, str] | None = None,
    ) -> AsyncIterator[Event]:
        """Create an execution and stream its events until completion.

        Yields each Event including the terminal event. Use ``client.get(id)``
        afterward to fetch the final output.
        """
        execution = await self.create(agent_id, input=input, labels=labels)
        async for event in self.events(execution.id):
            yield event
            if event.type in _TERMINAL_EVENT_TYPES:
                return

    async def run_until_complete(
        self,
        agent_id: str,
        input: Any = None,
        labels: dict[str, str] | None = None,
        on_event: Callable[[Event], Awaitable[None] | None] | None = None,
    ) -> Execution:
        """Convenience: create + stream + return final Execution.

        Calls ``on_event`` for each event if provided. Sync or async callbacks
        both work.
        """
        execution = await self.create(agent_id, input=input, labels=labels)
        async for event in self.events(execution.id):
            if on_event is not None:
                result = on_event(event)
                if asyncio.iscoroutine(result):
                    await result
            if event.type in _TERMINAL_EVENT_TYPES:
                break
        return await self.get(execution.id)

    async def submit_intent(
        self,
        execution_id: str,
        session_id: str,
        intent_type: str,
        *,
        tool_id: str = "",
        arguments: Any = None,
        idempotency_key: str = "",
        signal_type: str = "",
        output: Any = None,
        error: str = "",
        remote: bool = False,
    ) -> IntentResult:
        intent: dict[str, Any] = {"type": intent_type}
        if tool_id:
            intent["tool_id"] = tool_id
        if arguments is not None:
            intent["arguments"] = arguments
        if idempotency_key:
            intent["idempotency_key"] = idempotency_key
        if signal_type:
            intent["signal_type"] = signal_type
        if output is not None:
            intent["output"] = output
        if error:
            intent["error"] = error
        if remote:
            intent["remote"] = True
        body = {"execution_id": execution_id, "session_id": session_id, "intent": intent}
        resp = await self._request("POST", "/v0/agents/intent", json=body, idempotent=True)
        return IntentResult(**resp.json())

    async def report_step_result(
        self,
        execution_id: str,
        session_id: str,
        step_id: str,
        success: bool,
        data: Any = None,
        error: str = "",
    ) -> None:
        body: dict[str, Any] = {
            "execution_id": execution_id,
            "session_id": session_id,
            "step_id": step_id,
            "success": success,
        }
        if data is not None:
            body["data"] = data
        if error:
            body["error"] = error
        await self._request("POST", "/v0/agents/step-result", json=body, idempotent=True)

    async def submit_job_result(
        self,
        runner_id: str,
        job_id: str,
        execution_id: str,
        step_id: str,
        success: bool,
        *,
        data: Any = None,
        error: str = "",
        retryable: bool = False,
        started_at: str | None = None,
        completed_at: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "job_id": job_id,
            "execution_id": execution_id,
            "step_id": step_id,
            "success": success,
        }
        if data is not None:
            body["data"] = data
        if error:
            body["error"] = error
        if retryable:
            body["retryable"] = retryable
        if started_at is not None:
            body["started_at"] = started_at
        if completed_at is not None:
            body["completed_at"] = completed_at
        resp = await self._request(
            "POST",
            f"/v0/runners/{runner_id}/results",
            json=body,
            idempotent=True,
        )
        return resp.json()

    async def step_started(self, step_id: str, execution_id: str, runner_id: str) -> None:
        await self._request(
            "POST",
            f"/v0/runners/steps/{step_id}/started",
            json={"execution_id": execution_id, "runner_id": runner_id},
        )

    async def list_tools(self, prefix: str = "") -> list[dict[str, Any]]:
        """Fetch tool schemas from the kernel directory.

        Each entry: {id, description, input_schema, runner_id, registered_at}.
        Empty prefix returns all currently advertised tools.
        """
        params: dict[str, Any] = {}
        if prefix:
            params["prefix"] = prefix
        resp = await self._request("GET", "/v0/tools", params=params)
        data = resp.json()
        return data if isinstance(data, list) else []

    async def publish_tools(
        self,
        runner_id: str,
        schemas: list[dict[str, Any]],
    ) -> None:
        """Publish a runner's full tool schema set to the kernel.

        Replaces (not merges) any previous publication for this runner_id.
        Atomically updates both the schema cache and the capability index.
        Each schema entry must have at least: id, description, input_schema.
        """
        body = {"tools": schemas}
        await self._request(
            "POST",
            f"/v0/runners/{runner_id}/tools",
            json=body,
            idempotent=True,
        )

    async def update_capabilities(self, runner_id: str, tools: list[str]) -> None:
        await self._request(
            "POST",
            f"/v0/runners/{runner_id}/capabilities",
            json={"tools": tools},
        )

    async def unregister_runner(self, runner_id: str) -> None:
        await self._request("DELETE", f"/v0/runners/{runner_id}")

    async def agent_stream(
        self,
        agent_id: str,
        consumer_id: str,
    ) -> AsyncIterator[SSEEvent]:
        params = {"agent_id": agent_id, "consumer_id": consumer_id}
        async for event in self._stream_sse("/v0/agents/stream", params):
            yield event

    async def runner_stream(
        self,
        runner_id: str,
        consumer_id: str,
        capabilities: list[str] | None = None,
    ) -> AsyncIterator[SSEEvent]:
        params: dict[str, Any] = {"runner_id": runner_id, "consumer_id": consumer_id}
        if capabilities:
            params["capabilities"] = ",".join(capabilities)
        async for event in self._stream_sse("/v0/runners/stream", params):
            yield event

    async def _stream_sse(
        self,
        path: str,
        params: dict[str, Any],
    ) -> AsyncIterator[SSEEvent]:
        async with self._http.stream(
            "GET",
            path,
            params=params,
            headers={"Accept": "text/event-stream"},
            timeout=None,
        ) as response:
            response.raise_for_status()
            async for event in async_parse_sse(response.aiter_lines()):
                yield event

    def _retry_delay(self, attempt: int) -> float:
        return min(self.retry_base_delay * (2**attempt), self.retry_max_delay)

    async def _request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        idempotent: bool = False,
    ) -> httpx.Response:
        retryable = method == "GET" or idempotent
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self._http.request(method, path, json=json, params=params)
                if resp.status_code == 429:
                    if attempt >= self.max_retries:
                        raise api_error(resp)
                    retry_after = float(resp.headers.get("Retry-After", "1") or "1")
                    await asyncio.sleep(retry_after)
                    continue
                if resp.status_code == 403:
                    body = resp.json() if resp.content else {}
                    raise PolicyError(
                        body.get("error", "Forbidden"),
                        rule_id=body.get("rule_id", ""),
                    )
                if resp.status_code >= 500 and retryable:
                    last_error = api_error(resp)
                    if attempt >= self.max_retries:
                        break
                    delay = self._retry_delay(attempt)
                    logger.warning(
                        "Server error %d, retrying in %.1fs (attempt %d/%d)",
                        resp.status_code,
                        delay,
                        attempt + 1,
                        self.max_retries + 1,
                    )
                    await asyncio.sleep(delay)
                    continue
                if resp.status_code >= 400:
                    raise api_error(resp)
                return resp
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_error = NetworkError(str(e))
                if not retryable:
                    raise last_error from e
                if attempt >= self.max_retries:
                    break
                delay = self._retry_delay(attempt)
                logger.warning(
                    "Connection error, retrying in %.1fs (attempt %d/%d)",
                    delay,
                    attempt + 1,
                    self.max_retries + 1,
                )
                await asyncio.sleep(delay)
            except (APIError, PolicyError):
                raise
            except Exception as e:
                raise RebunoError(str(e)) from e
        raise last_error or RebunoError("max retries exceeded")
