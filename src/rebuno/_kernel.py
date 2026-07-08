from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

import httpx

from rebuno.errors import NotFoundError, error_from_response
from rebuno.identity import canonical_json
from rebuno.types import Execution, Step, StepDecision


def _json_field(value: str) -> bytes:
    return json.dumps(value).encode("utf-8")


class KernelClient:
    """Agent-side kernel client. Signs every request body with the agent secret."""

    def __init__(self, *, agent_id: str, secret: str, http: httpx.AsyncClient):
        self._agent_id = agent_id
        self._secret = secret.encode("utf-8")
        self._http = http

    def _sign(self, body: bytes) -> str:
        return "sha256=" + hmac.new(self._secret, body, hashlib.sha256).hexdigest()

    def _headers(self, body: bytes, extra: dict[str, str] | None = None) -> dict[str, str]:
        h = {
            "Content-Type": "application/json",
            "Rebuno-Agent-Id": self._agent_id,
            "Rebuno-Signature": self._sign(body),
        }
        if extra:
            h.update(extra)
        return h

    async def _send(self, method: str, path: str, body: bytes, extra: dict[str, str] | None = None) -> httpx.Response:
        resp = await self._http.request(method, path, content=body, headers=self._headers(body, extra))
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

    async def get_execution(self, execution_id: str) -> Execution:
        resp = await self._send("GET", f"/v0/executions/{execution_id}", b"")
        return Execution.model_validate(resp.json())

    async def get_step(self, execution_id: str, step_id: str) -> Step | None:
        try:
            resp = await self._send("GET", f"/v0/executions/{execution_id}/steps/{step_id}", b"")
        except NotFoundError:
            return None
        return Step.model_validate(resp.json())

    async def list_terminal_steps(self, execution_id: str) -> list[Step]:
        """Fetch an execution's terminal steps in one read, for replay hydration."""
        resp = await self._send("GET", f"/v0/executions/{execution_id}/steps?status=terminal", b"")
        return [Step.model_validate(s) for s in resp.json()]

    async def submit_step(
        self, execution_id: str, *, kind: str, target: str, args: Any, idempotency: str, step_id: str
    ) -> StepDecision:
        body = (
            b'{"kind":'
            + _json_field(kind)
            + b',"target":'
            + _json_field(target)
            + b',"args":'
            + canonical_json(args)
            + b',"idempotency":'
            + _json_field(idempotency)
            + b"}"
        )
        resp = await self._send("POST", f"/v0/executions/{execution_id}/steps", body, {"Rebuno-Step-Id": step_id})
        return StepDecision.model_validate(resp.json())

    async def complete_step(self, execution_id: str, step_id: str, *, result: Any) -> None:
        body = json.dumps({"result": result}).encode("utf-8")
        await self._send("POST", f"/v0/executions/{execution_id}/steps/{step_id}/complete", body)

    async def fail_step(self, execution_id: str, step_id: str, *, error: Any) -> None:
        body = json.dumps({"error": error}).encode("utf-8")
        await self._send("POST", f"/v0/executions/{execution_id}/steps/{step_id}/fail", body)

    async def complete_execution(self, execution_id: str, *, output: Any) -> None:
        body = json.dumps({"output": output}).encode("utf-8")
        await self._send("POST", f"/v0/executions/{execution_id}/complete", body)

    async def fail_execution(self, execution_id: str, *, error: str) -> None:
        body = json.dumps({"error": error}).encode("utf-8")
        await self._send("POST", f"/v0/executions/{execution_id}/fail", body)
