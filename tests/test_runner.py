"""Runner: capability collection and job dispatch into the @tool registry."""

from __future__ import annotations

import pytest
from conftest import make_job

from rebuno import Runner, tool


def _runner() -> Runner:
    return Runner("r-1", kernel_url="http://test", api_key="")


def test_runner_capabilities_default_to_registered_tools():
    @tool("a.one")
    async def one() -> int:
        return 1

    @tool("a.two")
    async def two() -> int:
        return 2

    r = _runner()
    caps = r._capabilities()
    assert set(caps) == {"a.one", "a.two"}


def test_runner_explicit_capabilities_override_registry():
    @tool("a.skipped")
    async def skipped():
        return None

    r = Runner("r-1", kernel_url="http://test", capabilities=["other.tool"])
    assert r._capabilities() == ["other.tool"]


def test_runner_capabilities_dedupe_preserves_order():
    @tool("a.one")
    async def one():
        return None

    @tool("a.two")
    async def two():
        return None

    r = _runner()
    caps = r._capabilities()
    assert caps == ["a.one", "a.two"]
    assert len(set(caps)) == len(caps)


async def test_dispatch_calls_registered_tool_function_directly():
    """Runner._dispatch bypasses the @tool wrapper (which would gate via
    kernel intent) and calls the raw function. This is correct: the runner
    IS the kernel-side execution."""
    captured = {}

    @tool("compute.echo")
    async def echo(value: str) -> str:
        captured["value"] = value
        return value.upper()

    r = _runner()
    result = await r._dispatch("compute.echo", {"value": "hi"})
    assert result == "HI"
    assert captured["value"] == "hi"


async def test_dispatch_supports_sync_tool_functions():
    @tool("compute.sync")
    def sync_op(x: int) -> int:
        return x * 2

    r = _runner()
    assert await r._dispatch("compute.sync", {"x": 21}) == 42


async def test_dispatch_unknown_tool_raises_runtime_error():
    r = _runner()
    with pytest.raises(RuntimeError, match="No handler registered"):
        await r._dispatch("nonexistent.tool", {})


async def test_dispatch_with_non_dict_arguments_passes_empty_kwargs():
    @tool("compute.noargs")
    async def noargs() -> str:
        return "ok"

    r = _runner()
    assert await r._dispatch("compute.noargs", None) == "ok"
    assert await r._dispatch("compute.noargs", "scalar") == "ok"


class _SpyRunnerClient:
    def __init__(self):
        self.results: list[dict] = []

    async def step_started(self, *args, **kwargs): ...

    async def submit_job_result(self, **kwargs):
        self.results.append(kwargs)


async def test_handle_job_success_submits_data():
    @tool("compute.add")
    async def add(a: int, b: int) -> int:
        return a + b

    r = _runner()
    spy = _SpyRunnerClient()
    r._client = spy  # type: ignore[assignment]
    job = make_job(tool_id="compute.add", arguments={"a": 2, "b": 3})

    await r._handle_job(job)

    assert len(spy.results) == 1
    assert spy.results[0]["success"] is True
    assert spy.results[0]["data"] == 5
    assert spy.results[0]["job_id"] == job.id


async def test_handle_job_failure_submits_error_with_retryable_flag():
    @tool("compute.bad")
    async def bad():
        raise ValueError("oops")

    r = _runner()
    spy = _SpyRunnerClient()
    r._client = spy  # type: ignore[assignment]

    await r._handle_job(make_job(tool_id="compute.bad", arguments={}))
    assert spy.results[0]["success"] is False
    assert "oops" in spy.results[0]["error"]
    assert spy.results[0]["retryable"] is False


async def test_handle_job_unknown_tool_reports_failure():
    r = _runner()
    spy = _SpyRunnerClient()
    r._client = spy  # type: ignore[assignment]

    await r._handle_job(make_job(tool_id="ghost.tool", arguments={}))
    assert spy.results[0]["success"] is False
    assert "ghost.tool" in spy.results[0]["error"]
