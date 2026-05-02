from __future__ import annotations

import random
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx

from rebuno.errors import (
    APIError,
    ConflictError,
    NotFoundError,
    UnauthorizedError,
    ValidationError,
)


@dataclass
class SSEEvent:
    type: str
    data: str
    id: str = ""


_STATUS_TO_ERROR: dict[int, type[APIError]] = {
    400: ValidationError,
    401: UnauthorizedError,
    404: NotFoundError,
    409: ConflictError,
}


def api_error(resp: httpx.Response) -> APIError:
    try:
        body = resp.json()
    except Exception:
        body = {}
    message = (body.get("error") if body else None) or (resp.text or "Unknown error")
    code = body.get("code", "UNKNOWN") if body else "UNKNOWN"
    details = body.get("details") if body else None
    cls = _STATUS_TO_ERROR.get(resp.status_code, APIError)
    return cls(message=message, code=code, status_code=resp.status_code, details=details)


def jittered_backoff(base_delay: float, attempt: int, max_delay: float) -> float:
    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
    return delay * (0.5 + random.random() * 0.5)


async def async_parse_sse(lines: AsyncIterator[str]) -> AsyncIterator[SSEEvent]:
    event_type = ""
    event_id = ""
    data_lines: list[str] = []

    def flush() -> SSEEvent | None:
        if event_type and data_lines:
            return SSEEvent(type=event_type, data="\n".join(data_lines), id=event_id)
        return None

    async for raw_line in lines:
        line = raw_line.rstrip("\n\r")
        if line.startswith("event:"):
            event_type = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].removeprefix(" "))
        elif line.startswith("id:"):
            event_id = line[3:].strip()
        elif line.startswith(("retry:", ":")):
            pass
        elif line == "":
            event = flush()
            event_type = ""
            event_id = ""
            data_lines = []
            if event is not None:
                yield event

    trailing = flush()
    if trailing is not None:
        yield trailing
