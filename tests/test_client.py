import httpx
import pytest

from rebuno.client import Client
from rebuno.errors import APIError, StepIDMismatch
from rebuno.types import Execution


def make_client(handler):
    transport = httpx.MockTransport(handler)
    c = Client(base_url="http://k", api_key="tok")
    c._http = httpx.AsyncClient(transport=transport, base_url="http://k", headers={"Authorization": "Bearer tok"})
    return c


async def test_create_execution():
    def handler(req):
        assert req.method == "POST"
        assert req.url.path == "/v0/executions"
        assert req.headers["Authorization"] == "Bearer tok"
        return httpx.Response(201, json={"id": "e1", "agent_id": "a", "status": "pending"})

    c = make_client(handler)
    exec = await c.create("a", input={"x": 1})
    assert isinstance(exec, Execution)
    assert exec.id == "e1"


async def test_cancel_and_approvals():
    def handler(req):
        if req.url.path.endswith("/cancel"):
            return httpx.Response(204)
        if "/grant" in req.url.path:
            return httpx.Response(204)
        return httpx.Response(200, json=[])

    c = make_client(handler)
    await c.cancel("e1")
    await c.grant_approval("ap1", decided_by="me")
    assert await c.list_approvals() == []


async def test_conflict_maps_to_api_error():
    def handler(req):
        return httpx.Response(409, json={"code": "conflict", "message": "already exists"})

    c = make_client(handler)
    with pytest.raises(APIError) as exc_info:
        await c.cancel("e1")
    assert exc_info.value.code == "conflict"


async def test_step_id_divergence_maps_to_step_id_mismatch():
    def handler(req):
        return httpx.Response(409, json={"code": "step_id_divergence", "message": "mismatch"})

    c = make_client(handler)
    with pytest.raises(StepIDMismatch):
        await c.cancel("e1")
