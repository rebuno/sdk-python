from __future__ import annotations

from typing import Any

from rebuno._internal.call import offload
from rebuno._internal.checkpoints import publish
from rebuno.execution import execution
from rebuno.types import StepResource


class CheckpointPolicy:
    """Capture after every N affecting steps and optionally on completion."""

    def __init__(self, every_steps: int = 1, on_completion: bool = True):
        if every_steps < 1:
            raise ValueError("every_steps must be positive")
        self.every_steps = every_steps
        self.on_completion = on_completion


async def resource(
    key: str,
    *,
    driver: Any,
    checkpoints: CheckpointPolicy | None = None,
) -> Any:
    """Register an external resource with the current execution and return its handle."""
    ctx = execution()
    async with ctx._exclusive():
        if key in ctx._resources:
            return ctx._resources[key]["handle"]
        policy = checkpoints or CheckpointPolicy()
        registration = {
            "key": key,
            "driver_id": driver.driver_id,
            "configuration": getattr(driver, "configuration", None),
            "coverage_reuse": bool(getattr(driver, "coverage_reuse", False)),
            "every_steps": policy.every_steps,
            "on_completion": policy.on_completion,
        }
        view = await ctx._on_owner_loop(
            ctx._kernel.register_resource(ctx.id, lease=ctx._lease, **registration)
        )
        if view.binding is not None:
            handle = await offload(driver.open, view.binding)
        else:
            handle, binding = await offload(driver.create, view.checkpoint_ref or None)
            await ctx._on_owner_loop(
                ctx._kernel.bind_resource(
                    ctx.id, key, lease=ctx._lease, binding=binding
                )
            )
        ctx._resources[key] = {
            "driver": driver,
            "handle": handle,
            "registration": registration,
        }
        if not view.covered:
            await publish(ctx, [StepResource(key=key, generation=view.generation)])
        return handle
