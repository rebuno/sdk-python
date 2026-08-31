from __future__ import annotations

import re
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

    def __str__(self) -> str:
        message = super().__str__()
        return f"{self.tool_id}: {message}" if self.tool_id else message

    def __repr__(self) -> str:
        return f"ToolError(tool_id={self.tool_id!r}, step_id={self.step_id!r})"


class RateLimited(RebunoError):
    """A step was rejected because a policy rate limit was exceeded."""

    def __init__(self, reason: str = "rate_limit_exceeded"):
        super().__init__(reason)
        self.reason = reason


class Blocked(RebunoError):
    """Internal control-flow signal: the kernel suspended the step.

    Raised inside a tool call to unwind the dispatch cleanly; the agent's
    webhook handler catches it and returns 200. Not normally seen by user code.
    """

    def __init__(self) -> None:
        super().__init__("step blocked")


class Terminated(RebunoError):
    """Internal control-flow signal: the execution is terminal (e.g. cancelled).

    Raised inside a kernel call so the dispatch unwinds; the handler returns 200.
    """


class LeaseSuperseded(APIError):
    """Internal control-flow signal: a newer delivery attempt owns this dispatch.

    The kernel refuses every mutation from the superseded attempt. The handler
    stops where it stands and returns 200, leaving the execution to the attempt
    that replaced it.
    """

    def __init__(
        self,
        message: str = "dispatch lease superseded",
        code: str = "lease_superseded",
        status_code: int = 409,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, code, status_code, details)


REFUSAL_TYPE = "rebuno_refusal"

_REFUSAL_RE = re.compile(rf"{REFUSAL_TYPE}: (\w+)(?: reason=(.*))?")

_TOKEN_RE = re.compile(r"[a-z0-9_]+")

_DEFAULT_REASON = {"denied": "policy_denied", "rate_limited": "rate_limit_exceeded"}


def refusal_message(decision: str, reason: str = "") -> str:
    """The marker a refused LLM call carries in its HTTP error body."""
    msg = f"{REFUSAL_TYPE}: {decision}"
    if reason:
        msg += f" reason={reason}"
    return msg


def raise_for_refusal(exc: BaseException) -> None:
    """Re-raise a Rebuno refusal carried in a provider error as its control-flow error.

    A step the kernel refuses (approval pending, policy denial, rate limit) reaches
    an LLM call as an HTTP error. Call this on the error the provider raised to get
    ``Blocked``, ``PolicyError``, ``RateLimited``, ``Terminated`` or
    ``LeaseSuperseded`` back, so the dispatch unwinds. Returns silently for any
    other exception.
    """
    for e in _causes(exc):
        m = _REFUSAL_RE.search(str(e))
        if not m:
            continue
        decision = m.group(1)
        reason = (m.group(2) or "").rstrip("'\"} \n") or _DEFAULT_REASON.get(
            decision, decision
        )
        if decision in ("blocked", "execution_blocked"):
            raise Blocked from exc
        if decision == "execution_terminal":
            raise Terminated(reason) from exc
        if decision == "lease_superseded":
            raise LeaseSuperseded from exc
        if decision == "denied":
            raise PolicyError(reason) from exc
        if decision == "rate_limited":
            raise RateLimited(reason) from exc
        return


def _causes(exc: BaseException | None, limit: int = 10):
    """``exc`` and the exceptions it was raised from."""
    for _ in range(limit):
        if exc is None:
            return
        yield exc
        exc = exc.__cause__ or exc.__context__


_ERROR_BY_CODE: dict[str, type[APIError]] = {
    "not_found": NotFoundError,
    "validation_error": ValidationError,
    "unauthorized": UnauthorizedError,
    "forbidden": ForbiddenError,
    "conflict": APIError,
    "lease_superseded": LeaseSuperseded,
}


def error_from_response(
    code: str, message: str, status_code: int, *, rule_id: str = ""
) -> RebunoError:
    """Translate a kernel error envelope ({"code", "message"}) into the matching SDK exception.

    Shared by Client and KernelClient so the two HTTP clients can't map the same
    error code to different exception types.
    """
    if code == "policy_denied":
        return PolicyError(message, rule_id=rule_id)
    cls = _ERROR_BY_CODE.get(code, APIError)
    return cls(message, code=code, status_code=status_code)


def failure_reason(exc: BaseException) -> str:
    """The text an execution's ``failure_reason`` records for ``exc``.

    Everything before the first colon is a stable token: a kernel reason
    (``policy_denied``, ``execution_token_budget_exceeded``, ``approval_timeout``,
    ``rate_limit_exceeded``, ``rate_limiter_unavailable``) or one of
    ``tool_error``, ``agent_error``, ``input_invalid``. A rule's own prose reason
    is not a token, so it follows ``policy_denied:``.
    """
    if isinstance(exc, PolicyError):
        # Exception.__str__ skips APIError's display formatting.
        reason = Exception.__str__(exc)
        return reason if _TOKEN_RE.fullmatch(reason) else f"policy_denied: {reason}"
    if isinstance(exc, RateLimited):
        return str(exc)
    if isinstance(exc, ToolError):
        return f"tool_error: {exc}"
    detail = Exception.__str__(exc) if isinstance(exc, APIError) else str(exc)
    return f"agent_error: {type(exc).__name__}: {detail}"
