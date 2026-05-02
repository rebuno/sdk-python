"""remote.Tools: kernel-directory-driven tool discovery."""

from __future__ import annotations

import inspect

import pytest

from rebuno import remote
from rebuno.execution import _reset_current, _set_current
from rebuno.types import IntentResult

SAMPLE_SCHEMAS = [
    {
        "id": "github.create_pr",
        "description": "Create a pull request.",
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "title": {"type": "string"},
            },
            "required": ["owner", "repo", "title"],
        },
        "runner_id": "runner-1",
        "registered_at": "2026-05-01T12:00:00Z",
    },
    {
        "id": "github.issue_read",
        "description": "Read an issue.",
        "input_schema": {"type": "object"},
        "runner_id": "runner-1",
        "registered_at": "2026-05-01T12:00:00Z",
    },
]


class _StubClient:
    def __init__(self, schemas: list[dict]):
        self.schemas = schemas
        self.intents: list[dict] = []
        self.last_prefix: str | None = None

    async def list_tools(self, prefix: str = "") -> list[dict]:
        self.last_prefix = prefix
        return [s for s in self.schemas if s["id"].startswith(prefix)]

    async def submit_intent(self, **kwargs):
        self.intents.append(kwargs)
        return IntentResult(accepted=True, step_id="step-1")

    async def report_step_result(self, **kwargs): ...


def test_requires_non_empty_prefix():
    with pytest.raises(ValueError, match="non-empty prefix"):
        remote.Tools("")


def test_registers_in_module_registry():
    t = remote.Tools("github")
    assert t in remote.all_handles()


def test_tools_property_raises_before_connect():
    t = remote.Tools("github")
    with pytest.raises(RuntimeError, match="has not been connected"):
        _ = t.tools


async def test_connect_fetches_schemas_and_builds_callables():
    handle = remote.Tools("github")
    client = _StubClient(SAMPLE_SCHEMAS)

    await handle.connect(client)

    assert client.last_prefix == "github"
    assert len(handle.tools) == 2
    names = {fn.__name__ for fn in handle.tools}
    assert names == {"create_pr", "issue_read"}


async def test_wrapped_callable_has_synthesized_signature():
    handle = remote.Tools("github")
    await handle.connect(_StubClient(SAMPLE_SCHEMAS))

    create_pr = next(fn for fn in handle.tools if fn.__name__ == "create_pr")
    sig = inspect.signature(create_pr)
    assert list(sig.parameters) == ["owner", "repo", "title"]
    for name in ("owner", "repo", "title"):
        assert sig.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert sig.parameters[name].default is inspect.Parameter.empty
    assert create_pr.__doc__ == "Create a pull request."
    assert create_pr.__rebuno_tool_id__ == "github.create_pr"
    assert create_pr.__rebuno_remote__ is True


async def test_connect_is_idempotent():
    handle = remote.Tools("github")
    client = _StubClient(SAMPLE_SCHEMAS)

    await handle.connect(client)
    first = handle.tools
    await handle.connect(client)  # should not refetch / rebuild
    assert handle.tools is first


async def test_connect_with_no_matching_tools_yields_empty_list():
    handle = remote.Tools("nonexistent")
    await handle.connect(_StubClient(SAMPLE_SCHEMAS))
    assert handle.tools == []


async def test_calling_remote_tool_outside_execution_raises():
    handle = remote.Tools("github")
    await handle.connect(_StubClient(SAMPLE_SCHEMAS))

    create_pr = next(fn for fn in handle.tools if fn.__name__ == "create_pr")
    with pytest.raises(RuntimeError, match="outside an active execution"):
        await create_pr(owner="o", repo="r", title="t")


async def test_calling_remote_tool_submits_intent_and_awaits_result():
    handle = remote.Tools("github")
    client = _StubClient(SAMPLE_SCHEMAS)
    await handle.connect(client)

    create_pr = next(fn for fn in handle.tools if fn.__name__ == "create_pr")

    # Build a minimal ExecutionState-shaped object so the wrapper can call
    # _client.submit_intent and _wait correctly.
    class _State:
        id = "exec-1"
        session_id = "sess-1"
        _client = client

        async def _wait(self, kind, key, timeout_msg):
            return {"status": "succeeded", "result": {"pr_url": "..."}}

    token = _set_current(_State())  # type: ignore[arg-type]
    try:
        result = await create_pr(owner="o", repo="r", title="t")
    finally:
        _reset_current(token)

    assert result == {"pr_url": "..."}
    assert len(client.intents) == 1
    assert client.intents[0]["intent_type"] == "invoke_tool"
    assert client.intents[0]["tool_id"] == "github.create_pr"
    assert client.intents[0]["remote"] is True
    assert client.intents[0]["arguments"] == {"owner": "o", "repo": "r", "title": "t"}


async def test_connect_all_resolves_every_registered_handle():
    a = remote.Tools("github")
    b = remote.Tools("compute")

    schemas = SAMPLE_SCHEMAS + [
        {
            "id": "compute.heavy",
            "description": "Heavy.",
            "input_schema": {"type": "object"},
        }
    ]
    await remote.connect_all(_StubClient(schemas))

    assert {fn.__rebuno_tool_id__ for fn in a.tools} == {"github.create_pr", "github.issue_read"}
    assert {fn.__rebuno_tool_id__ for fn in b.tools} == {"compute.heavy"}


async def test_connect_all_failure_doesnt_abort_others():
    class _PartlyFailingClient(_StubClient):
        async def list_tools(self, prefix: str = ""):
            if prefix == "broken":
                raise RuntimeError("network down")
            return await super().list_tools(prefix)

    bad = remote.Tools("broken")
    good = remote.Tools("github")
    await remote.connect_all(_PartlyFailingClient(SAMPLE_SCHEMAS))

    # bad is left unconnected (handle.tools raises) but good resolved
    assert good._connected
    assert not bad._connected
