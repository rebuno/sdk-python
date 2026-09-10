from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any

import httpx2

from rebuno.errors import NotFoundError, error_from_response
from rebuno.types import Execution, Step, StepDecision

MAX_HEARTBEAT_INTERVAL = 30.0


@dataclass(frozen=True, slots=True)
class DispatchLease:
    """The delivery attempt a webhook arrived under.

    Every mutation sends it back, so the kernel refuses a handler whose dispatch
    was reclaimed and re-delivered to a newer attempt.
    """

    dispatch_id: str
    attempt: int
    timeout: float

    @property
    def heartbeat_interval(self) -> float:
        return min(self.timeout / 3, MAX_HEARTBEAT_INTERVAL)

    def headers(self) -> dict[str, str]:
        return {
            "Rebuno-Dispatch-Id": self.dispatch_id,
            "Rebuno-Dispatch-Attempt": str(self.attempt),
        }


class KernelClient:
    """Agent-side kernel client. Signs each request with the agent secret."""

    def __init__(self, *, agent_id: str, secret: str, http: httpx2.AsyncClient):
        self._agent_id = agent_id
        self._secret = secret.encode("utf-8")
        self._http = http

    def _sign(self, request: httpx2.Request) -> str:
        fields = [
            "rebuno-request-v1",
            request.method,
            request.url.raw_path.decode("ascii"),
            request.headers["Rebuno-Timestamp"],
            request.headers.get("Rebuno-Dispatch-Id", ""),
            request.headers.get("Rebuno-Dispatch-Attempt", ""),
        ]
        message = ("\n".join(fields) + "\n").encode() + request.content
        return "v1=" + hmac.new(self._secret, message, hashlib.sha256).hexdigest()

    async def _send(
        self, method: str, path: str, body: bytes, extra: dict[str, str] | None = None
    ) -> httpx2.Response:
        request = self._http.build_request(
            method,
            path,
            content=body,
            headers={"Content-Type": "application/json", **(extra or {})},
        )
        request.headers["Rebuno-Agent-Id"] = self._agent_id
        request.headers["Rebuno-Timestamp"] = str(int(time.time()))
        request.headers["Rebuno-Signature"] = self._sign(request)
        resp = await self._http.send(request, follow_redirects=False)
        if resp.status_code >= 300:
            raise error_from_response(resp)
        return resp

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
