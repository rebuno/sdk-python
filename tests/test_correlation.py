"""CorrelationMap: per-execution future map for SSE-driven results."""

from __future__ import annotations

import asyncio

from rebuno._internal.correlation import CorrelationMap


async def test_resolves_pre_existing_subscription():
    cm = CorrelationMap()
    fut = cm.future("result", "step-1")
    cm.resolve("result", "step-1", {"status": "succeeded", "result": 42})
    value = await fut
    assert value["result"] == 42


async def test_resolves_late_subscriber():
    cm = CorrelationMap()
    cm.resolve("signal", "approval", {"approved": True})
    fut = cm.future("signal", "approval")
    value = await fut
    assert value["approved"] is True


async def test_keys_are_independent():
    cm = CorrelationMap()
    f1 = cm.future("result", "step-1")
    f2 = cm.future("result", "step-2")
    cm.resolve("result", "step-1", "first")
    assert (await f1) == "first"
    assert not f2.done()


async def test_kinds_are_independent():
    cm = CorrelationMap()
    f_result = cm.future("result", "step-1")
    f_signal = cm.future("signal", "step-1")
    cm.resolve("result", "step-1", "result-value")
    assert (await f_result) == "result-value"
    assert not f_signal.done()


async def test_double_resolve_is_noop():
    cm = CorrelationMap()
    fut = cm.future("result", "step-1")
    cm.resolve("result", "step-1", "first")
    cm.resolve("result", "step-1", "second")
    assert (await fut) == "first"


async def test_cancel_all_marks_pending_futures_cancelled():
    cm = CorrelationMap()
    f1 = cm.future("result", "step-1")
    f2 = cm.future("signal", "x")
    cm.cancel_all()
    assert f1.cancelled()
    assert f2.cancelled()


async def test_concurrent_wait_and_resolve():
    cm = CorrelationMap()

    async def waiter():
        return await cm.future("result", "step-1")

    async def resolver():
        await asyncio.sleep(0)  # let waiter subscribe
        cm.resolve("result", "step-1", "ok")

    waiter_task = asyncio.create_task(waiter())
    await resolver()
    assert (await waiter_task) == "ok"
