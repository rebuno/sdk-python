from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict


class ExecutionStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore")


class Execution(_Model):
    id: str
    agent_id: str = ""
    agent_version: str = ""
    input: Any = None
    status: ExecutionStatus = ExecutionStatus.PENDING
    output: Any = None
    failure_reason: str = ""


class Step(_Model):
    step_id: str
    execution_id: str = ""
    kind: str = ""
    target: str = ""
    status: str = ""
    idempotency: str = ""
    args: Any = None
    result: Any = None
    error: Any = None


class StepDecision(_Model):
    decision: str
    result: Any = None
    error: Any = None
    approval_id: str | None = None
    reason: str = ""


class Event(_Model):
    execution_id: str = ""
    event_seq: int = 0
    type: str = ""
    payload: Any = None
    occurred_at: str = ""


class Approval(_Model):
    id: str
    step_id: str = ""
    execution_id: str = ""
    status: str = ""
    message: str = ""
    decided_by: str = ""
    rationale: str = ""
