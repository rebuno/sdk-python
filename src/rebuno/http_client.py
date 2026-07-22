from __future__ import annotations

import asyncio
import codecs
import contextlib
import json
import time
from typing import Any

import httpx

from rebuno.execution import ExecutionContext, _get_current

_DELTA_FLUSH_BYTES = 2000
_DELTA_FLUSH_INTERVAL = 0.05
_DELTA_MAX_CHARS = 6000


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

        target = str(payload.get(self._model_field) or "")

        step_id, dec = await ctx.begin_llm(target, payload)
        if dec.decision == "replay":
            return _replay_response(request, dec.result)

        resp = await self._inner.handle_async_request(request)
        if resp.status_code < 400 and _is_event_stream(resp.headers.get("content-type", "")):
            content_type = resp.headers.get("content-type", "text/event-stream")
            tee = _TeeStream(ctx, step_id, resp, content_type)
            return httpx.Response(
                resp.status_code, headers={"content-type": content_type}, stream=tee, request=request
            )

        # Whole response (including error statuses): read under a lease, record it,
        # and hand back a reconstructed response.
        try:
            async with ctx.lease():
                await resp.aread()
        except Exception as e:
            await ctx._fail_step_quietly(step_id, e)
            raise
        record = {
            "status": resp.status_code,
            "headers": {"content-type": resp.headers.get("content-type", "application/json")},
            "body": resp.text,
        }
        await ctx.record_llm(step_id, record)
        return _response_from_record(request, record)

    async def aclose(self) -> None:
        await self._inner.aclose()


class _TeeStream(httpx.AsyncByteStream):
    """Streams the provider's bytes to the caller while accumulating the whole and
    publishing live deltas, then records the assembled response as the step result
    when the stream ends.

    Recording fires once, from whichever comes first: the byte iterator reaching
    EOF, or the consumer closing the response. A mid-stream error fails the step.
    """

    def __init__(self, ctx: ExecutionContext, step_id: str, resp: httpx.Response, content_type: str):
        self._ctx = ctx
        self._step_id = step_id
        self._resp = resp
        self._content_type = content_type
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._chunks: list[str] = []
        self._pending = ""
        self._seq = 0
        self._done = False
        self._hb: asyncio.Task | None = None

    async def __aiter__(self):
        if self._hb is None:
            self._hb = self._ctx.start_heartbeat()  # renew the lease while streaming
        last_flush = time.monotonic()
        try:
            async for raw in self._resp.aiter_raw():
                # Accumulate before yielding: a consumer that breaks right after
                # receiving a chunk never resumes us, so recording after the yield
                # would drop that chunk from the result.
                text = self._decoder.decode(raw)  # incremental: never splits a UTF-8 char
                if text:
                    self._chunks.append(text)
                    self._pending += text
                yield raw  # live to the caller
                now = time.monotonic()
                if len(self._pending) >= _DELTA_FLUSH_BYTES or (now - last_flush) >= _DELTA_FLUSH_INTERVAL:
                    await self._flush()
                    last_flush = now
        except Exception as e:
            await self._finish(error=e)
            raise
        await self._finish()

    async def aclose(self) -> None:
        # A consumer may close without draining to EOF, so __aiter__'s tail may not
        # run; record here too. _finish is idempotent.
        try:
            await self._finish()
        finally:
            await self._resp.aclose()

    async def _finish(self, *, error: Exception | None = None) -> None:
        """Record the assembled response, or fail the step, exactly once."""
        if self._done:
            return
        self._done = True
        try:
            if error is not None:
                await self._ctx._fail_step_quietly(self._step_id, error)
                return
            tail = self._decoder.decode(b"", final=True)
            if tail:
                self._chunks.append(tail)
                self._pending += tail
            if self._pending:
                await self._flush()
            record = {
                "status": self._resp.status_code,
                "headers": {"content-type": self._content_type},
                "body": "".join(self._chunks),
            }
            await self._ctx.record_llm(self._step_id, record)
        finally:
            await self._stop_heartbeat()

    async def _flush(self) -> None:
        for i in range(0, len(self._pending), _DELTA_MAX_CHARS):
            await self._ctx.publish_llm_delta(self._step_id, self._seq, self._pending[i : i + _DELTA_MAX_CHARS])
            self._seq += 1
        self._pending = ""

    async def _stop_heartbeat(self) -> None:
        if self._hb is not None:
            hb, self._hb = self._hb, None
            hb.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await hb


class _BytesStream(httpx.AsyncByteStream):
    """Replays fixed bytes as a stream, so a replayed streamed call still yields a
    streaming response the provider SDK can iterate."""

    def __init__(self, data: bytes, chunk_size: int = 4096):
        self._data = data
        self._chunk = chunk_size

    async def __aiter__(self):
        for i in range(0, len(self._data), self._chunk):
            yield self._data[i : i + self._chunk]

    async def aclose(self) -> None:
        pass


def http_client(*, model_field: str = "model", **kwargs: Any) -> httpx.AsyncClient:
    """Return an ``httpx.AsyncClient`` that records LLM calls as durable steps.

    Pass it to an async LLM client::

        llm = AsyncOpenAI(http_client=rebuno.http_client())

    ``model_field`` names the request-body field holding the model id (the step
    ``target``); it defaults to ``"model"``. Extra keyword arguments are
    forwarded to ``httpx.AsyncClient`` (e.g. ``timeout``).
    """
    return httpx.AsyncClient(transport=RebunoTransport(model_field=model_field), **kwargs)


def _is_event_stream(content_type: str) -> bool:
    """True for a Server-Sent-Events content type."""
    return content_type.split(";", 1)[0].strip().lower() == "text/event-stream"


def _json_body(request: httpx.Request) -> dict[str, Any] | None:
    body = request.content
    if not body:
        return None
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _replay_response(request: httpx.Request, record: Any) -> httpx.Response:
    """Rebuild a replayed response — as a stream when the recorded response was an
    event stream (so a replayed streamed call still yields a stream), otherwise as
    a whole response."""
    if isinstance(record, dict):
        headers = record.get("headers") or {}
        if _is_event_stream(str(headers.get("content-type", ""))):
            return _stream_response_from_record(request, record)
    return _response_from_record(request, record)


def _response_from_record(request: httpx.Request, record: Any) -> httpx.Response:
    """Rebuild an httpx.Response from a recorded provider response.

    Only the status, content-type, and body are reconstructed — hop-by-hop and
    length/encoding headers are deliberately dropped so a replayed body is never
    mismatched against a stale ``content-encoding`` or ``content-length``.
    """
    status, headers, content = _record_parts(record)
    return httpx.Response(status, headers=headers, content=content, request=request)


def _stream_response_from_record(request: httpx.Request, record: Any) -> httpx.Response:
    """Like :func:`_response_from_record`, but delivers the recorded body as a
    stream so a replayed streaming call still yields a streaming response."""
    status, headers, content = _record_parts(record)
    return httpx.Response(status, headers=headers, stream=_BytesStream(content), request=request)


def _record_parts(record: Any) -> tuple[int, dict[str, str], bytes]:
    if not isinstance(record, dict):
        return 200, {"content-type": "application/json"}, json.dumps(record).encode("utf-8")
    status = int(record.get("status", 200))
    headers = record.get("headers") or {"content-type": "application/json"}
    body = record.get("body", "")
    content = body.encode("utf-8") if isinstance(body, str) else json.dumps(body).encode("utf-8")
    return status, headers, content
