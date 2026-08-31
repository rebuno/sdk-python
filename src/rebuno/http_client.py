from __future__ import annotations

import codecs
import json
import time
from typing import Any

import httpx2

from rebuno.errors import (
    REFUSAL_TYPE,
    Blocked,
    LeaseSuperseded,
    PolicyError,
    RateLimited,
    RebunoError,
    Terminated,
    refusal_message,
)
from rebuno.execution import ExecutionContext, _get_current

# The status and decision name each refusal is returned as.
_REFUSALS: dict[type[RebunoError], tuple[int, str]] = {
    Blocked: (403, "blocked"),
    PolicyError: (403, "denied"),
    RateLimited: (429, "rate_limited"),
    Terminated: (403, "execution_terminal"),
    LeaseSuperseded: (409, "lease_superseded"),
}

_DELTA_FLUSH_BYTES = 2000
_DELTA_FLUSH_INTERVAL = 0.05
_DELTA_MAX_CHARS = (
    1750  # the kernel caps a delta at 7000 bytes; UTF-8 runs to 4 bytes a char
)


class RebunoTransport(httpx2.AsyncBaseTransport):
    """An httpx2 transport that records LLM calls as durable Rebuno steps.

    Wrap a real transport (defaults to ``httpx2.AsyncHTTPTransport``) and use it
    in an ``httpx2.AsyncClient``. The request body's ``model`` is the step
    ``target``.
    """

    def __init__(self, inner: httpx2.AsyncBaseTransport | None = None):
        self._inner = inner or httpx2.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        ctx = _get_current()
        if ctx is None:
            return await self._inner.handle_async_request(request)

        payload = _json_body(request)
        if payload is None:
            # Non-JSON body (file uploads, form posts): not an LLM call we can
            # identify, pass through untouched.
            return await self._inner.handle_async_request(request)

        target = str(payload.get("model") or "")

        try:
            step_id, dec = await ctx.begin_llm(target, payload)
        except tuple(_REFUSALS) as e:
            return _refusal_response(request, e)
        if dec.decision == "replay":
            return _replay_response(request, dec.result)

        resp = await self._inner.handle_async_request(request)
        if resp.status_code < 400 and _is_event_stream(
            resp.headers.get("content-type", "")
        ):
            content_type = resp.headers.get("content-type", "text/event-stream")
            tee = _TeeStream(ctx, step_id, resp, content_type)
            return httpx2.Response(
                resp.status_code,
                headers={"content-type": content_type},
                stream=tee,
                request=request,
            )

        # Whole response (including error statuses): read it, record it, and hand
        # back a reconstructed response.
        try:
            await resp.aread()
        except Exception as e:
            await ctx._fail_step_quietly(step_id, e)
            raise
        record = {
            "status": resp.status_code,
            "headers": {
                "content-type": resp.headers.get("content-type", "application/json")
            },
            "body": resp.text,
        }
        await ctx.record_llm(step_id, record)
        return _response_from_record(request, record)

    async def aclose(self) -> None:
        await self._inner.aclose()


class _TeeStream(httpx2.AsyncByteStream):
    """Streams the provider's bytes to the caller while accumulating the whole and
    publishing live deltas, then records the assembled response as the step result
    when the stream ends.

    Recording fires once, from whichever comes first: the byte iterator reaching
    EOF, or the consumer closing the response. A mid-stream error fails the step.
    """

    def __init__(
        self,
        ctx: ExecutionContext,
        step_id: str,
        resp: httpx2.Response,
        content_type: str,
    ):
        self._ctx = ctx
        self._step_id = step_id
        self._resp = resp
        self._content_type = content_type
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._chunks: list[str] = []
        self._pending = ""
        self._seq = 0
        self._done = False

    async def __aiter__(self):
        last_flush = time.monotonic()
        try:
            async for raw in self._resp.aiter_raw():
                # Accumulate before yielding: a consumer that breaks right after
                # receiving a chunk never resumes us, so recording after the yield
                # would drop that chunk from the result.
                text = self._decoder.decode(
                    raw
                )  # incremental: never splits a UTF-8 char
                if text:
                    self._chunks.append(text)
                    self._pending += text
                yield raw  # live to the caller
                now = time.monotonic()
                if (
                    len(self._pending) >= _DELTA_FLUSH_BYTES
                    or (now - last_flush) >= _DELTA_FLUSH_INTERVAL
                ):
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

    async def _flush(self) -> None:
        for i in range(0, len(self._pending), _DELTA_MAX_CHARS):
            await self._ctx.publish_llm_delta(
                self._step_id, self._seq, self._pending[i : i + _DELTA_MAX_CHARS]
            )
            self._seq += 1
        self._pending = ""


class _BytesStream(httpx2.AsyncByteStream):
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


def http_client(**kwargs: Any) -> httpx2.AsyncClient:
    """Return an ``httpx2.AsyncClient`` that records LLM calls as durable steps.

    Pass it to an async LLM client::

        llm = AsyncOpenAI(http_client=rebuno.http_client())

    Keyword arguments are forwarded to ``httpx2.AsyncClient`` (e.g. ``timeout``).
    """
    return httpx2.AsyncClient(transport=RebunoTransport(), **kwargs)


def _refusal_response(request: httpx2.Request, e: RebunoError) -> httpx2.Response:
    """A refused decision as an HTTP error carrying the refusal marker."""
    status, decision = _REFUSALS[type(e)]
    # Exception.__str__ skips APIError's display formatting.
    reason = Exception.__str__(e) if isinstance(e, (PolicyError, RateLimited)) else ""
    message = refusal_message(decision, reason)
    return httpx2.Response(
        status,
        json={"error": {"type": REFUSAL_TYPE, "message": message}},
        request=request,
    )


def _is_event_stream(content_type: str) -> bool:
    """True for a Server-Sent-Events content type."""
    return content_type.split(";", 1)[0].strip().lower() == "text/event-stream"


def _json_body(request: httpx2.Request) -> dict[str, Any] | None:
    body = request.content
    if not body:
        return None
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _replay_response(request: httpx2.Request, record: Any) -> httpx2.Response:
    """Rebuild a replayed response — as a stream when the recorded response was an
    event stream (so a replayed streamed call still yields a stream), otherwise as
    a whole response."""
    if isinstance(record, dict):
        headers = record.get("headers") or {}
        if _is_event_stream(str(headers.get("content-type", ""))):
            return _stream_response_from_record(request, record)
    return _response_from_record(request, record)


def _response_from_record(request: httpx2.Request, record: Any) -> httpx2.Response:
    """Rebuild an httpx2.Response from a recorded provider response.

    Only the status, content-type, and body are reconstructed — hop-by-hop and
    length/encoding headers are deliberately dropped so a replayed body is never
    mismatched against a stale ``content-encoding`` or ``content-length``.
    """
    status, headers, content = _record_parts(record)
    return httpx2.Response(status, headers=headers, content=content, request=request)


def _stream_response_from_record(
    request: httpx2.Request, record: Any
) -> httpx2.Response:
    """Like :func:`_response_from_record`, but delivers the recorded body as a
    stream so a replayed streaming call still yields a streaming response."""
    status, headers, content = _record_parts(record)
    return httpx2.Response(
        status, headers=headers, stream=_BytesStream(content), request=request
    )


def _record_parts(record: Any) -> tuple[int, dict[str, str], bytes]:
    if not isinstance(record, dict):
        return (
            200,
            {"content-type": "application/json"},
            json.dumps(record).encode("utf-8"),
        )
    status = int(record.get("status", 200))
    headers = record.get("headers") or {"content-type": "application/json"}
    body = record.get("body", "")
    content = (
        body.encode("utf-8")
        if isinstance(body, str)
        else json.dumps(body).encode("utf-8")
    )
    return status, headers, content
