from __future__ import annotations

from typing import Any


class RebunoError(Exception):
    """Base exception for all Rebuno SDK errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.details = details or {}


class NetworkError(RebunoError):
    """Raised when a network-level error occurs (connection refused, timeout, etc.)."""

    def __repr__(self) -> str:
        return f"NetworkError({str(self)!r})"


class APIError(RebunoError):
    """Raised when the API returns an error response."""

    def __init__(
        self,
        message: str,
        code: str,
        status_code: int,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, details)
        self.code = code
        self.status_code = status_code

    def __str__(self) -> str:
        return f"[{self.code}] {super().__str__()} (HTTP {self.status_code})"

    def __repr__(self) -> str:
        return f"APIError(status_code={self.status_code!r}, code={self.code!r}, message={super().__str__()!r})"


class ValidationError(APIError):
    """Raised when request validation fails (400)."""


class UnauthorizedError(APIError):
    """Raised when authentication fails (401)."""


class ForbiddenError(APIError):
    """Raised when a decision is refused (403 forbidden)."""


class NotFoundError(APIError):
    """Raised when a resource is not found (404)."""


class ConflictError(APIError):
    """Raised when there is a resource conflict (409)."""


class PolicyError(APIError):
    """Raised when an action is denied by policy."""

    def __init__(self, message: str, rule_id: str = ""):
        super().__init__(message, code="policy_denied", status_code=403)
        self.rule_id = rule_id

    def __repr__(self) -> str:
        return f"PolicyError(reason={Exception.__str__(self)!r})"


class ToolError(RebunoError):
    def __init__(
        self,
        message: str,
        tool_id: str = "",
        step_id: str = "",
        retryable: bool = False,
    ):
        super().__init__(message)
        self.tool_id = tool_id
        self.step_id = step_id
        self.retryable = retryable

    def __repr__(self) -> str:
        return f"ToolError(tool_id={self.tool_id!r}, step_id={self.step_id!r})"


class StepIDMismatch(APIError):
    """Kernel rejected the SDK-computed step id (409 step_id_divergence).

    Signals the agent's effect sequence diverged from a prior dispatch
    (non-determinism not wrapped in rebuno.step) or canonicalization drift.
    """


class RateLimited(RebunoError):
    """A step was rejected because a policy rate limit was exceeded."""

    def __init__(self, reason: str = "rate_limit_exceeded"):
        super().__init__(reason)
        self.reason = reason


class Blocked(RebunoError):
    """Internal control-flow signal: a step is awaiting human approval.

    Raised inside a tool call to unwind the dispatch cleanly; the agent's
    webhook handler catches it and returns 200 (the execution is already
    'blocked' in the kernel). Not normally seen by user code.
    """

    def __init__(self, approval_id: str | None = None):
        super().__init__("execution blocked awaiting approval")
        self.approval_id = approval_id


class Terminated(RebunoError):
    """Internal control-flow signal: the execution is terminal (e.g. cancelled).

    Raised inside a kernel call so the dispatch unwinds; the handler returns 200.
    """


_ERROR_BY_CODE: dict[str, type[APIError]] = {
    "not_found": NotFoundError,
    "validation_error": ValidationError,
    "unauthorized": UnauthorizedError,
    "forbidden": ForbiddenError,
    "conflict": APIError,
    "step_id_divergence": StepIDMismatch,
}


def error_from_response(code: str, message: str, status_code: int, *, rule_id: str = "") -> RebunoError:
    """Translate a kernel error envelope ({"code", "message"}) into the matching SDK exception.

    Shared by Client and KernelClient so the two HTTP clients can't map the same
    error code to different exception types.
    """
    if code == "policy_denied":
        return PolicyError(message, rule_id=rule_id)
    cls = _ERROR_BY_CODE.get(code, APIError)
    return cls(message, code=code, status_code=status_code)
