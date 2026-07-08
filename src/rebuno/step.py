from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rebuno.execution import _get_current


async def step(
    name: str,
    fn: Callable[..., Any],
    args: dict[str, Any] | None = None,
    idempotency: str = "safe_to_retry",
) -> Any:
    """Record non-deterministic local work as a durable step.

    The result is recorded under ``name`` so it replays identically on resume.
    Use for anything that influences which effects run (current time, random
    choices, fresh ids). ``idempotency`` mirrors ``@tool``'s: ``safe_to_retry``
    (default) for reads/non-determinism; ``at_most_once`` for side effects that
    must not be re-run on resume.

    ``args`` is the JSON-recorded payload used for step identity/hashing. It is
    passed to ``fn`` as ``fn(**args)`` when the step runs; pass ``None`` (the
    default) when ``fn`` takes no arguments.
    """
    ctx = _get_current()
    if ctx is None:
        raise RuntimeError(f"rebuno.step('{name}') called outside an active execution.")
    payload = args or {}
    return await ctx.invoke_tool(name, payload, idempotency=idempotency, run=lambda: fn(**payload))
