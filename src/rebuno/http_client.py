from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from rebuno.execution import _get_current

logger = logging.getLogger("rebuno.http")


class RebunoTransport(httpx.AsyncBaseTransport):
    """An httpx transport that records LLM calls as durable Rebuno steps.

    Wrap a real transport (defaults to ``httpx.AsyncHTTPTransport``) and use it
    in an ``httpx.AsyncClient``. ``model_field`` names the request-body field
    holding the model id, used as the step ``target``.
    """

    def __init__(self, inner: httpx.AsyncBaseTransport | None = None, *, model_field: str = "model"):
        self._inner = inner or httpx.AsyncHTTPTransport()
        self._model_field = model_field

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        ctx = _get_current()
        if ctx is None:
            return await self._inner.handle_async_request(request)

        payload = _json_body(request)
        if payload is None:
            # Non-JSON body (file uploads, form posts): not an LLM call we can
            # identify, pass through untouched.
            return await self._inner.handle_async_request(request)
        if payload.get("stream"):
            # Streaming responses are not recorded yet; pass through so the call
            # still works, but it is not durable. See module TODO.
            logger.warning("rebuno: streaming LLM call is not durable; passing through")
            return await self._inner.handle_async_request(request)

        target = str(payload.get(self._model_field) or "")

        async def forward() -> dict[str, Any]:
            resp = await self._inner.handle_async_request(request)
            await resp.aread()
            return {
                "status": resp.status_code,
                "headers": {"content-type": resp.headers.get("content-type", "application/json")},
                "body": resp.text,
            }

        record = await ctx.invoke_llm(target, payload, run=forward)
        return _response_from_record(request, record)

    async def aclose(self) -> None:
        await self._inner.aclose()


def http_client(*, model_field: str = "model", **kwargs: Any) -> httpx.AsyncClient:
    """Return an ``httpx.AsyncClient`` that records LLM calls as durable steps.

    Pass it to an async LLM client::

        llm = AsyncOpenAI(http_client=rebuno.http_client())

    ``model_field`` names the request-body field holding the model id (the step
    ``target``); it defaults to ``"model"``. Extra keyword arguments are
    forwarded to ``httpx.AsyncClient`` (e.g. ``timeout``).
    """
    return httpx.AsyncClient(transport=RebunoTransport(model_field=model_field), **kwargs)


def _json_body(request: httpx.Request) -> dict[str, Any] | None:
    body = request.content
    if not body:
        return None
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _response_from_record(request: httpx.Request, record: Any) -> httpx.Response:
    """Rebuild an httpx.Response from a recorded provider response.

    Only the status, content-type, and body are reconstructed — hop-by-hop and
    length/encoding headers are deliberately dropped so a replayed body is never
    mismatched against a stale ``content-encoding`` or ``content-length``.
    """
    if not isinstance(record, dict):
        return httpx.Response(200, json=record, request=request)
    status = int(record.get("status", 200))
    headers = record.get("headers") or {"content-type": "application/json"}
    body = record.get("body", "")
    content = body.encode("utf-8") if isinstance(body, str) else json.dumps(body).encode("utf-8")
    return httpx.Response(status, headers=headers, content=content, request=request)
