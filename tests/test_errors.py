from rebuno.errors import (
    APIError,
    Blocked,
    ForbiddenError,
    NotFoundError,
    PolicyError,
    RateLimited,
    RebunoError,
    Terminated,
    ToolError,
    UnauthorizedError,
    ValidationError,
    error_from_response,
)


def test_hierarchy():
    assert issubclass(APIError, RebunoError)
    assert issubclass(NotFoundError, APIError)
    for cls in (Blocked, Terminated, RateLimited, ToolError, PolicyError):
        assert issubclass(cls, RebunoError)


def test_policy_error_reason():
    p = PolicyError("nope", rule_id="r1")
    assert p.rule_id == "r1"
    assert "nope" in str(p)


def test_error_from_response_maps_known_codes():
    assert isinstance(error_from_response("not_found", "x", 404), NotFoundError)
    assert isinstance(
        error_from_response("validation_error", "x", 400), ValidationError
    )
    assert isinstance(error_from_response("unauthorized", "x", 401), UnauthorizedError)
    assert isinstance(error_from_response("forbidden", "x", 403), ForbiddenError)
    assert isinstance(error_from_response("conflict", "x", 409), APIError)


def test_error_from_response_policy_denied_carries_rule_id():
    err = error_from_response("policy_denied", "nope", 403, rule_id="r1")
    assert isinstance(err, PolicyError)
    assert err.rule_id == "r1"


def test_error_from_response_unknown_code_falls_back_to_api_error():
    err = error_from_response("something_new", "weird", 500)
    assert isinstance(err, APIError)
    assert not isinstance(err, (NotFoundError, ValidationError, UnauthorizedError))
    assert err.code == "something_new"
    assert err.status_code == 500


def test_refusal_reason_stops_at_the_marker_line():
    from rebuno.errors import PolicyError, raise_for_refusal, refusal_message

    body = {
        "error": {
            "type": "rebuno_refusal",
            "message": refusal_message("denied", "budget_gone"),
        }
    }
    provider = Exception(f"Error code: 403 - {body}\n\nRequest ID: req_abc123")
    try:
        raise_for_refusal(provider)
    except PolicyError as e:
        assert Exception.__str__(e) == "budget_gone"
    else:
        raise AssertionError("expected PolicyError")
