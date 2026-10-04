import asyncio
import contextlib

import pytest

from rebuno import (
    Blocked,
    CheckpointPolicy,
    CheckpointUnavailable,
    LeaseSuperseded,
    Terminated,
    ToolError,
    resource,
    step,
    tool,
    wrap_tool,
)
from rebuno._internal.checkpoints import checkpoint_on_completion
from rebuno._kernel import DispatchLease
from rebuno.execution import ExecutionContext, _reset_current, _set_current
from rebuno.types import Resource, StepDecision, StepResource


class FakeKernel:
    def __init__(self, view=None, decisions=()):
        self.view = view or Resource(key="workspace")
        self.decisions = list(decisions)
        self.calls = []

    async def register_resource(self, execution_id, *, lease, **registration):
        self.calls.append(("register", registration))
        if not self.view.every_steps and registration["every_steps"]:
            self.view = self.view.model_copy(
                update={
                    "every_steps": registration["every_steps"],
                    "on_completion": registration["on_completion"],
                }
            )
        return self.view

    async def bind_resource(self, execution_id, key, *, lease, binding):
        self.calls.append(("bind", binding))
        self.view = self.view.model_copy(update={"binding": binding})

    async def publish_checkpoints(self, execution_id, *, lease, **records):
        self.calls.append(("publish", records))
        if records.get("captures"):
            self.view = self.view.model_copy(update={"covered": True})

    async def submit_step(self, execution_id, *, lease, **step):
        self.calls.append(("submit", step))
        return self.decisions.pop(0)

    async def complete_step(self, execution_id, step_id, *, lease, **outcome):
        self.calls.append(("complete", outcome))

    async def fail_step(self, execution_id, step_id, *, lease, **outcome):
        self.calls.append(("fail", outcome))

    def named(self, name):
        return [payload for call, payload in self.calls if call == name]


class Driver:
    driver_id = "test.v1"
    configuration = {"size": "small"}

    def __init__(self):
        self.calls = []
        self.fail_checkpoint = False
        self.checkpoint_count = 0

    async def create(self, checkpoint_ref=None):
        self.calls.append(("create", checkpoint_ref))
        return "handle", {"id": "sbx-new"}

    async def open(self, binding):
        self.calls.append(("open", binding))
        return "handle"

    async def checkpoint(self, handle):
        if self.fail_checkpoint:
            raise RuntimeError("snapshot failed")
        self.calls.append(("checkpoint", handle))
        self.checkpoint_count += 1
        return f"snap-{self.checkpoint_count}"


@contextlib.contextmanager
def running(kernel):
    ctx = ExecutionContext(
        kernel=kernel,
        execution_id="e1",
        lease=DispatchLease("d1", 1, 120.0),
        agent_id="a",
        input=None,
    )
    token = _set_current(ctx)
    try:
        yield ctx
    finally:
        _reset_current(token)


async def test_new_execution_creates_binds_and_captures_a_baseline():
    kernel, driver = FakeKernel(), Driver()
    with running(kernel):
        assert (
            await resource("workspace", driver=driver, checkpoints=CheckpointPolicy())
            == "handle"
        )
        assert (
            await resource("workspace", driver=driver, checkpoints=CheckpointPolicy())
            == "handle"
        )

        assert driver.calls == [("create", None), ("checkpoint", "handle")]
        assert kernel.named("bind") == [{"id": "sbx-new"}]
        assert kernel.named("publish") == [
            {
                "captures": [
                    {
                        "key": "workspace",
                        "generation": 0,
                        "checkpoint_ref": "snap-1",
                    }
                ]
            }
        ]
        assert len(kernel.named("register")) == 1


@pytest.mark.parametrize("supports_checkpoints", [True, False])
async def test_resource_without_policy_reuses_its_binding_without_captures(
    supports_checkpoints,
):
    kernel = FakeKernel(decisions=[proceed("s1", due=False)])
    driver = Driver()
    if not supports_checkpoints:
        driver.checkpoint = None
    with running(kernel) as ctx:
        await resource("workspace", driver=driver)
        assert await step("write", lambda: "ok", resources=["workspace"]) == "ok"
        await checkpoint_on_completion(ctx)
    with running(kernel) as ctx:
        await resource("workspace", driver=driver)
        await checkpoint_on_completion(ctx)

    assert driver.calls == [("create", None), ("open", {"id": "sbx-new"})]
    assert kernel.named("publish") == []
    assert all(
        call["every_steps"] == 0 and call["on_completion"] is False
        for call in kernel.named("register")
    )


async def test_checkpoint_policy_enables_captures_on_an_existing_binding():
    kernel, driver = FakeKernel(), Driver()
    with running(kernel):
        await resource("workspace", driver=driver)
    with running(kernel):
        await resource(
            "workspace", driver=driver, checkpoints=CheckpointPolicy(every_steps=5)
        )

    assert driver.calls == [
        ("create", None),
        ("open", {"id": "sbx-new"}),
        ("checkpoint", "handle"),
    ]
    assert kernel.view.every_steps == 5
    assert len(kernel.named("publish")) == 1


async def test_checkpoint_policy_requires_a_checkpoint_method():
    kernel, driver = FakeKernel(), Driver()
    driver.checkpoint = None
    with running(kernel), pytest.raises(ValueError, match="driver.checkpoint"):
        await resource("workspace", driver=driver, checkpoints=CheckpointPolicy())
    assert driver.calls == []


async def test_later_dispatch_opens_the_recorded_binding():
    view = Resource(key="workspace", binding={"id": "sbx-1"}, generation=3)
    kernel, driver = FakeKernel(view), Driver()
    with running(kernel):
        await resource("workspace", driver=driver, checkpoints=CheckpointPolicy())

        assert driver.calls == [
            ("open", {"id": "sbx-1"}),
            ("checkpoint", "handle"),
        ]
        assert kernel.named("bind") == []


async def test_fork_creates_its_resource_from_the_selected_checkpoint():
    view = Resource(key="workspace", checkpoint_ref="snap-5", covered=True)
    kernel, driver = FakeKernel(view), Driver()
    with running(kernel):
        await resource("workspace", driver=driver, checkpoints=CheckpointPolicy())

        assert driver.calls == [("create", "snap-5")]
        assert kernel.named("bind") == [{"id": "sbx-new"}]
        assert kernel.named("publish") == []


async def test_missing_checkpoint_stops_resource_initialization():
    class Gone(Driver):
        async def create(self, checkpoint_ref=None):
            self.calls.append(("create", checkpoint_ref))
            raise CheckpointUnavailable("expired")

    kernel = FakeKernel(Resource(key="workspace", checkpoint_ref="snap-5"))
    driver = Gone()
    with running(kernel):
        with pytest.raises(CheckpointUnavailable, match="expired"):
            await resource("workspace", driver=driver, checkpoints=CheckpointPolicy())
        assert driver.calls == [("create", "snap-5")]
        assert kernel.named("bind") == []
        assert kernel.named("publish") == []


def proceed(step_id, *, due):
    return StepDecision(
        decision="proceed",
        step_id=step_id,
        resources=[StepResource(key="workspace", generation=4, due=due)],
    )


async def test_due_capture_is_recorded_with_the_step_outcome():
    kernel = FakeKernel(
        Resource(key="workspace", covered=True),
        [proceed("s1", due=False), proceed("s2", due=True)],
    )
    driver = Driver()
    with running(kernel):
        await resource("workspace", driver=driver, checkpoints=CheckpointPolicy())

        @tool("write", resources=["workspace"])
        async def write() -> str:
            return "ok"

        await write()
        await write()

        assert [s["resources"] for s in kernel.named("submit")] == [["workspace"]] * 2
        assert kernel.named("complete") == [
            {"result": "ok"},
            {
                "result": "ok",
                "captures": [
                    {
                        "key": "workspace",
                        "generation": 4,
                        "checkpoint_ref": "snap-1",
                    }
                ],
            },
        ]


async def test_failed_capture_keeps_the_step_outcome():
    kernel = FakeKernel(
        Resource(key="workspace", covered=True), [proceed("s1", due=True)]
    )
    driver = Driver()
    with running(kernel):
        await resource("workspace", driver=driver, checkpoints=CheckpointPolicy())
        driver.fail_checkpoint = True

        assert await step("setup", lambda: "done", resources=["workspace"]) == "done"

        assert kernel.named("submit")[0]["resources"] == ["workspace"]
        assert kernel.named("complete") == [
            {
                "result": "done",
                "capture_failures": [
                    {"key": "workspace", "generation": 4, "error": "snapshot failed"}
                ],
            }
        ]


async def test_tools_and_local_steps_default_to_no_resource_changes():
    kernel = FakeKernel(
        Resource(key="workspace", covered=True),
        [StepDecision(decision="proceed", step_id=f"s{i}") for i in range(5)],
    )
    driver = Driver()
    with running(kernel):
        await resource("workspace", driver=driver, checkpoints=CheckpointPolicy())

        @tool("read")
        async def read():
            return "read"

        assert await read() == "read"
        assert await wrap_tool("lookup", lambda args: "found")() == "found"
        assert await step("now", lambda: 42) == 42
        await wrap_tool("write", lambda args: "ok", resources=["workspace"])()
        await step("update", lambda: "ok", resources=["workspace"])

        assert [s.get("resources", []) for s in kernel.named("submit")] == [
            [],
            [],
            [],
            ["workspace"],
            ["workspace"],
        ]
        assert driver.checkpoint_count == 0


async def test_completion_captures_only_resources_without_coverage():
    kernel, driver = FakeKernel(Resource(key="workspace", covered=True)), Driver()
    with running(kernel) as ctx:
        await resource(
            "workspace", driver=driver, checkpoints=CheckpointPolicy(every_steps=5)
        )

        await checkpoint_on_completion(ctx)
        assert kernel.named("publish") == []

        kernel.view = Resource(key="workspace", generation=7)
        await checkpoint_on_completion(ctx)
        assert driver.calls[-1] == ("checkpoint", "handle")
        assert len(kernel.named("publish")) == 1


async def test_replayed_tool_does_not_run_or_capture():
    kernel = FakeKernel(
        Resource(key="workspace", covered=True),
        [StepDecision(decision="replay", step_id="s1", result="recorded")],
    )
    driver = Driver()
    with running(kernel):
        await resource("workspace", driver=driver, checkpoints=CheckpointPolicy())

        def body():
            pytest.fail("a replay must not invoke the tool body")

        assert await step("write", body, resources=["workspace"]) == "recorded"
        assert driver.calls == [("create", None)]
        assert kernel.named("complete") == []


async def test_failed_tool_captures_its_partial_changes():
    kernel = FakeKernel(
        Resource(key="workspace", covered=True), [proceed("s1", due=True)]
    )
    driver = Driver()
    with running(kernel):
        await resource("workspace", driver=driver, checkpoints=CheckpointPolicy())

        def body():
            raise ValueError("partial write")

        with pytest.raises(ToolError, match="partial write"):
            await step("write", body, resources=["workspace"])
        assert kernel.named("complete") == []
        outcome = kernel.named("fail")[0]
        assert outcome["error"] == {"message": "partial write"}
        assert outcome["captures"][0]["checkpoint_ref"] == "snap-1"


@pytest.mark.parametrize("signal", [Blocked, Terminated, LeaseSuperseded])
async def test_capture_control_flow_stops_outcome_writes(signal):
    class ControlFlowDriver(Driver):
        async def checkpoint(self, handle):
            raise signal

    kernel = FakeKernel(
        Resource(key="workspace", covered=True), [proceed("s1", due=True)]
    )
    with running(kernel):
        await resource(
            "workspace", driver=ControlFlowDriver(), checkpoints=CheckpointPolicy()
        )
        with pytest.raises(signal):
            await step("write", lambda: "ok", resources=["workspace"])
        assert kernel.named("complete") == []
        assert kernel.named("fail") == []


async def test_concurrent_tools_capture_before_the_next_mutation():
    snapshots, values = [], []

    class StateDriver(Driver):
        async def checkpoint(self, handle):
            snapshots.append(values.copy())
            return await super().checkpoint(handle)

    kernel = FakeKernel(
        Resource(key="workspace", covered=True),
        [proceed("s1", due=True), proceed("s2", due=True)],
    )
    with running(kernel):
        await resource(
            "workspace", driver=StateDriver(), checkpoints=CheckpointPolicy()
        )

        @tool("write", resources=["workspace"])
        async def write(value):
            values.append(value)
            await asyncio.sleep(0)
            return value

        assert await asyncio.gather(write(1), write(2)) == [1, 2]
        assert snapshots == [[1], [1, 2]]


async def test_resource_calls_from_another_loop_use_the_owner_for_kernel_io():
    owner = asyncio.get_running_loop()

    class LoopKernel(FakeKernel):
        async def register_resource(self, *args, **kwargs):
            assert asyncio.get_running_loop() is owner
            return await super().register_resource(*args, **kwargs)

        async def publish_checkpoints(self, *args, **kwargs):
            assert asyncio.get_running_loop() is owner
            return await super().publish_checkpoints(*args, **kwargs)

    kernel, driver = LoopKernel(), Driver()
    with running(kernel):
        handle = await asyncio.to_thread(
            lambda: asyncio.run(
                resource("workspace", driver=driver, checkpoints=CheckpointPolicy())
            )
        )
        assert handle == "handle"
        assert kernel.named("publish")[0]["captures"][0]["checkpoint_ref"] == "snap-1"
        assert (
            await asyncio.wait_for(
                resource("workspace", driver=driver, checkpoints=CheckpointPolicy()), 2
            )
            == handle
        )
