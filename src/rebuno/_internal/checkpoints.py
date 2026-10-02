from __future__ import annotations

import logging
from typing import Any

from rebuno._internal.call import offload
from rebuno.errors import (
    Blocked,
    LeaseSuperseded,
    PolicyError,
    RateLimited,
    RebunoError,
    Terminated,
)
from rebuno.types import StepResource

logger = logging.getLogger("rebuno.resource")


async def capture(ctx: Any, due: list[StepResource]) -> dict[str, Any]:
    captures, failures = [], []
    for r in due:
        try:
            managed = ctx._resources.get(r.key)
            if managed is None:
                raise RebunoError("resource() was not called in this dispatch")
            ref = await offload(managed["driver"].checkpoint, managed["handle"])
            captures.append(
                {"key": r.key, "generation": r.generation, "checkpoint_ref": ref}
            )
        except (Blocked, Terminated, PolicyError, RateLimited, LeaseSuperseded):
            raise
        except Exception as e:
            logger.warning("checkpoint of resource %r failed", r.key, exc_info=True)
            failures.append(
                {
                    "key": r.key,
                    "generation": r.generation,
                    "error": str(e) or type(e).__name__,
                }
            )
    records = {}
    if captures:
        records["captures"] = captures
    if failures:
        records["capture_failures"] = failures
    return records


async def publish(ctx: Any, due: list[StepResource]) -> None:
    records = await capture(ctx, due)
    await ctx._on_owner_loop(
        ctx._kernel.publish_checkpoints(ctx.id, lease=ctx._lease, **records)
    )


async def checkpoint_on_completion(ctx: Any) -> None:
    if not ctx._resources:
        return
    try:
        async with ctx._exclusive():
            due = []
            for managed in ctx._resources.values():
                view = await ctx._on_owner_loop(
                    ctx._kernel.register_resource(
                        ctx.id, lease=ctx._lease, **managed["registration"]
                    )
                )
                if view.on_completion and not view.covered:
                    due.append(StepResource(key=view.key, generation=view.generation))
            if due:
                await publish(ctx, due)
    except (Blocked, Terminated, PolicyError, RateLimited, LeaseSuperseded):
        raise
    except Exception:
        logger.warning("checkpoint on completion failed", exc_info=True)
