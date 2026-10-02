from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any


async def offload(fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Await ``fn``'s result, running a synchronous ``fn`` in a worker thread."""

    if inspect.iscoroutinefunction(fn):
        return await fn(*args, **kwargs)
    result = await asyncio.to_thread(fn, *args, **kwargs)
    return await result if inspect.isawaitable(result) else result
