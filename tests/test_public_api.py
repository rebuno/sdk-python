"""Public-API export contract. Detailed behavior lives in the per-module test files."""

from __future__ import annotations

import rebuno


def test_public_api_exports():
    for name in (
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
    ):
        assert hasattr(rebuno, name), f"rebuno.{name} missing"


def test_types_module_reexports():
    assert hasattr(rebuno.types, "Execution")
    assert hasattr(rebuno.types, "Event")
    assert hasattr(rebuno.types, "ClaimResult")
    assert hasattr(rebuno.types, "Job")
