from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from typing import Any

import httpx2

from rebuno.errors import NotFoundError, error_from_response
from rebuno.types import Execution, Step, StepDecision


@dataclass(frozen=True, slots=True)
class DispatchLease:
    """The delivery attempt a webhook arrived under.

    Every mutation sends it back, so the kernel refuses a handler whose dispatch
    was reclaimed and re-delivered to a newer attempt.
    """

    dispatch_id: str
    attempt: int

    def supersedes(self, other: DispatchLease) -> bool:
        """Whether this delivery replaces one already being handled. Attempts only
        order within a dispatch; a different dispatch is always fresh work."""
        return self.dispatch_id != other.dispatch_id or self.attempt > other.attempt

    def headers(self) -> dict[str, str]:
        return {
            "Rebuno-Dispatch-Id": self.dispatch_id,
            "Rebuno-Dispatch-Attempt": str(self.attempt),
        }


class KernelClient:
    """Agent-side kernel client. Signs every request body with the agent secret."""

    def __init__(self, *, agent_id: str, secret: str, http: httpx2.AsyncClient):
        self._agent_id = agent_id
        self._secret = secret.encode("utf-8")
        self._http = http

    def _sign(self, body: bytes) -> str:
        return "sha256=" + hmac.new(self._secret, body, hashlib.sha256).hexdigest()

    def _headers(
        self, body: bytes, extra: dict[str, str] | None = None
    ) -> dict[str, str]:
        h = {
            "Content-Type": "application/json",
            "Rebuno-Agent-Id": self._agent_id,
            "Rebuno-Signature": self._sign(body),
        }
        if extra:
            h.update(extra)
        return h

    async def _send(
        self, method: str, path: str, body: bytes, extra: dict[str, str] | None = None
    ) -> httpx2.Response:
        resp = await self._http.request(
            method, path, content=body, headers=self._headers(body, extra)
        )
        if resp.status_code >= 400:
            raise self._error(resp)
        return resp

    @staticmethod
    def _error(resp: httpx2.Response) -> Exception:
        try:
            data = resp.json()
        except Exception:
            data = {}
        code = data.get("code", "internal_error")
        message = data.get("message", resp.text or "request failed")
        return error_from_response(
            code, message, resp.status_code, rule_id=data.get("rule_id", "")
        )

    async def get_execution(self, execution_id: str) -> Execution:
        resp = await self._send("GET", f"/v0/executions/{execution_id}", b"")
        return Execution.model_validate(resp.json())

    async def get_step(self, execution_id: str, step_id: str) -> Step | None:
        try:
            resp = await self._send(
                "GET", f"/v0/executions/{execution_id}/steps/{step_id}", b""
            )
        except NotFoundError:
            return None
        return Step.model_validate(resp.json())

    async def submit_step(
        self,
        execution_id: str,
        *,
        lease: DispatchLease,
        kind: str,
        target: str,
        args: Any,
        idempotency: str,
    ) -> StepDecision:
        body = json.dumps(
            {"kind": kind, "target": target, "args": args, "idempotency": idempotency}
        ).encode("utf-8")
        resp = await self._send(
            "POST", f"/v0/executions/{execution_id}/steps", body, lease.headers()
        )
        return StepDecision.model_validate(resp.json())

    async def complete_step(
        self, execution_id: str, step_id: str, *, lease: DispatchLease, result: Any
    ) -> None:
        body = json.dumps({"result": result}).encode("utf-8")
        await self._send(
            "POST",
            f"/v0/executions/{execution_id}/steps/{step_id}/complete",
            body,
            lease.headers(),
        )

    async def fail_step(
        self, execution_id: str, step_id: str, *, lease: DispatchLease, error: Any
    ) -> None:
        body = json.dumps({"error": error}).encode("utf-8")
        await self._send(
            "POST",
            f"/v0/executions/{execution_id}/steps/{step_id}/fail",
            body,
            lease.headers(),
        )

    async def stream_delta(
        self, execution_id: str, step_id: str, *, seq: int, data: str
    ) -> None:
        body = json.dumps({"seq": seq, "data": data}).encode("utf-8")
        await self._send(
            "POST", f"/v0/executions/{execution_id}/steps/{step_id}/stream", body
        )

    async def heartbeat(self, execution_id: str, *, lease: DispatchLease) -> None:
        """Renew the dispatch lease while a long effect body runs (empty signed body)."""
        await self._send(
            "POST", f"/v0/executions/{execution_id}/heartbeat", b"", lease.headers()
        )

    async def complete_execution(
        self, execution_id: str, *, lease: DispatchLease, output: Any
    ) -> None:
        body = json.dumps({"output": output}).encode("utf-8")
        await self._send(
            "POST", f"/v0/executions/{execution_id}/complete", body, lease.headers()
        )

    async def fail_execution(
        self, execution_id: str, *, lease: DispatchLease, error: str
    ) -> None:
        body = json.dumps({"error": error}).encode("utf-8")
        await self._send(
            "POST", f"/v0/executions/{execution_id}/fail", body, lease.headers()
        )
