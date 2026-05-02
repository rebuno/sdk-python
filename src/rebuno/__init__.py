"""Rebuno Python SDK.

Public surface:

  Agent     — long-lived process consuming executions for one agent_id
  Client    — HTTP client for external services and tool-side kernel calls
  Runner    — long-lived process executing tools assigned by the kernel
  tool      — decorator that registers a function as a Rebuno tool
  MCPServer — MCP integration
  remote    — kernel-mediated remote tool discovery (remote.Tools(prefix))
  execution — ambient accessor for the current ExecutionState

"""

from rebuno import remote, types
from rebuno.agent import Agent
from rebuno.client import Client
from rebuno.errors import (
    APIError,
    ConflictError,
    NetworkError,
    NotFoundError,
    PolicyError,
    RebunoError,
    ToolError,
    UnauthorizedError,
    ValidationError,
)
from rebuno.execution import execution
from rebuno.mcp import MCPServer
from rebuno.runner import Runner
from rebuno.tool import tool

__all__ = [
    "Agent",
    "Client",
    "Runner",
    "tool",
    "MCPServer",
    "remote",
    "execution",
    "types",
    "RebunoError",
    "APIError",
    "PolicyError",
    "ToolError",
    "NetworkError",
    "ConflictError",
    "NotFoundError",
    "UnauthorizedError",
    "ValidationError",
]
