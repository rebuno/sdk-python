"""Rebuno Python SDK.

Public surface:

  Agent     — webhook-driven consumer of executions for one agent_id
  Client    — HTTP client for creating/inspecting executions and approvals
  tool        — decorator registering an async function as a Rebuno tool
  wrap_tool   — route an arbitrary (non-decorator) tool through Rebuno
  step        — record non-deterministic local work as a durable step
  http_client — an httpx2 client that records LLM calls as durable steps
  raise_for_refusal — turn a refused LLM call's provider error back into Blocked/PolicyError/...
  execution   — ambient accessor for the current ExecutionContext
"""

from rebuno import types
from rebuno.agent import Agent
from rebuno.client import Client
from rebuno.errors import (
    APIError,
    Blocked,
    ForbiddenError,
    NetworkError,
    NotFoundError,
    PolicyError,
    RateLimited,
    RebunoError,
    Terminated,
    ToolError,
    UnauthorizedError,
    ValidationError,
    failure_reason,
    raise_for_refusal,
)
from rebuno.execution import execution
from rebuno.http_client import RebunoTransport, http_client
from rebuno.step import step
from rebuno.tool import tool, wrap_tool

__all__ = [
    "Agent",
    "Client",
    "tool",
    "wrap_tool",
    "step",
    "http_client",
    "RebunoTransport",
    "execution",
    "types",
    "RebunoError",
    "APIError",
    "PolicyError",
    "ToolError",
    "NetworkError",
    "NotFoundError",
    "UnauthorizedError",
    "ForbiddenError",
    "ValidationError",
    "RateLimited",
    "Blocked",
    "Terminated",
    "failure_reason",
    "raise_for_refusal",
]
