from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel


class ExecutionStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Execution(BaseModel):
    id: str
    agent_id: str = ""
    input: Any = None
    status: ExecutionStatus = ExecutionStatus.PENDING
    output: Any = None
    failure_reason: str = ""
    session: str = ""
    parent_execution_id: str | None = None
    forked_from: str | None = None
    fork_seq: int = 0
    restoration: dict[str, dict[str, Any]] | None = None


class Step(BaseModel):
    step_id: str
    execution_id: str = ""
    kind: str = ""
    target: str = ""
    args_hash: str = ""
    occurrence: int = 0
    status: str = ""
    idempotency: str = ""
    args: Any = None
    result: Any = None
    error: Any = None


class StepResource(BaseModel):
    key: str
    generation: int = 0
    due: bool = False


class StepDecision(BaseModel):
    decision: str
    step_id: str = ""
    result: Any = None
    error: Any = None
    approval_id: str | None = None
    reason: str = ""
    rule_id: str = ""
    resources: list[StepResource] = []


class Resource(BaseModel):
    key: str
    generation: int = 0
    binding: Any = None
    checkpoint_ref: str = ""
    covered: bool = False
    every_steps: int = 1
    on_completion: bool = True


class Event(BaseModel):
    execution_id: str = ""
    event_seq: int = 0
    type: str = ""
    payload: Any = None
    occurred_at: str = ""


class Approval(BaseModel):
    id: str
    step_id: str = ""
    execution_id: str = ""
    status: str = ""
    message: str = ""
    decided_by: str = ""
    rationale: str = ""
