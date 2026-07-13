from __future__ import annotations

import os
from typing import Any

import httpx

from rebuno.errors import NetworkError, error_from_response
from rebuno.types import Approval, Event, Execution, Step

USER_AGENT = "rebuno-python-sdk"


class Client:
    """Async HTTP client for client/admin kernel routes (Bearer auth).

    Defaults to env vars REBUNO_URL and REBUNO_API_KEY.
    """

    def __init__(self, base_url: str | None = None, api_key: str | None = None, *, timeout: float = 35.0):
        url = base_url or os.environ.get("REBUNO_URL", "")
        if not url:
            raise ValueError("Client base_url is required (set REBUNO_URL or pass base_url=)")
        self.base_url = url.rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("REBUNO_API_KEY", "")
        headers = {"User-Agent": USER_AGENT}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self._http = httpx.AsyncClient(base_url=self.base_url, headers=headers, timeout=timeout)

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> Client:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            resp = await self._http.request(method, path, **kwargs)
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            raise NetworkError(str(e)) from e
        if resp.status_code >= 400:
            raise self._error(resp)
        return resp

    @staticmethod
    def _error(resp: httpx.Response) -> Exception:
        try:
            data = resp.json()
        except Exception:
            data = {}
        code = data.get("code", "internal_error")
        message = data.get("message", resp.text or "request failed")
        return error_from_response(code, message, resp.status_code, rule_id=data.get("rule_id", ""))

    async def create(self, agent_id: str, input: Any = None, *, agent_version: str = "") -> Execution:
        body: dict[str, Any] = {"agent_id": agent_id}
        if input is not None:
            body["input"] = input
        if agent_version:
            body["agent_version"] = agent_version
        resp = await self._request("POST", "/v0/executions", json=body)
        return Execution.model_validate(resp.json())

    async def get(self, execution_id: str) -> Execution:
        resp = await self._request("GET", f"/v0/executions/{execution_id}")
        return Execution.model_validate(resp.json())

    async def events(self, execution_id: str, *, after_seq: int = 0, limit: int = 100) -> list[Event]:
        params: dict[str, Any] = {"limit": limit}
        if after_seq:
            params["after_seq"] = after_seq
        resp = await self._request("GET", f"/v0/executions/{execution_id}/events", params=params)
        data = resp.json()
        return [Event.model_validate(e) for e in (data or [])]

    async def cancel(self, execution_id: str) -> None:
        await self._request("POST", f"/v0/executions/{execution_id}/cancel")

    async def get_step(self, execution_id: str, step_id: str) -> Step:
        resp = await self._request("GET", f"/v0/executions/{execution_id}/steps/{step_id}")
        return Step.model_validate(resp.json())

    async def list_steps(self, execution_id: str, *, status: str = "") -> list[Step]:
        params: dict[str, Any] = {}
        if status:
            params["status"] = status
        resp = await self._request("GET", f"/v0/executions/{execution_id}/steps", params=params)
        data = resp.json()
        return [Step.model_validate(s) for s in (data or [])]

    async def list_approvals(self, *, status: str = "pending") -> list[Approval]:
        resp = await self._request("GET", "/v0/approvals", params={"status": status})
        data = resp.json()
        return [Approval.model_validate(a) for a in (data or [])]

    async def get_approval(self, approval_id: str) -> Approval:
        resp = await self._request("GET", f"/v0/approvals/{approval_id}")
        return Approval.model_validate(resp.json())

    async def grant_approval(self, approval_id: str, *, decided_by: str, rationale: str = "") -> None:
        body = {"decided_by": decided_by, "rationale": rationale}
        await self._request("POST", f"/v0/approvals/{approval_id}/grant", json=body)

    async def deny_approval(self, approval_id: str, *, decided_by: str, rationale: str = "") -> None:
        body = {"decided_by": decided_by, "rationale": rationale}
        await self._request("POST", f"/v0/approvals/{approval_id}/deny", json=body)
