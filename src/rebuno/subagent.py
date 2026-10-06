from __future__ import annotations

from typing import Any

from rebuno.client import Client
from rebuno.execution import _current_step, _get_current
from rebuno.types import SpawnedBy


async def subagent(
    agent_id: str, input: Any = None, *, client: Client | None = None
) -> Any:
    """Run ``agent_id`` as a subagent of the calling tool and return its output.

    Return the result from the tool body: the kernel records the subagent's
    outcome as the tool's step. Once every in-flight call of the execution
    waits, the execution suspends, and the handler reruns after they settle.

    ``client`` needs the ``executions:write`` scope. Defaults to ``Client()``.
    """
    ctx = _get_current()
    step_id = _current_step.get()
    if ctx is None or step_id is None:
        raise RuntimeError(f"subagent('{agent_id}') called outside a tool body.")
    spawned_by = SpawnedBy(execution_id=ctx.id, step_id=step_id)
    if client is not None:
        await client.create(agent_id, input, spawned_by=spawned_by)
    else:
        async with Client() as owned:
            await owned.create(agent_id, input, spawned_by=spawned_by)
    return await ctx.await_subagent(step_id)
