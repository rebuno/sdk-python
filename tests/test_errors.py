import httpx2
import pytest

from rebuno.errors import (
    APIError,
    ConflictError,
    ForbiddenError,
    NotFoundError,
    PolicyError,
    UnauthorizedError,
    ValidationError,
    error_from_response,
)


def response(status, **body):
    return httpx2.Response(status, json=body)


@pytest.mark.parametrize(
    ("status", "code", "cls"),
    [
        (400, "validation_error", ValidationError),
        (401, "unauthorized", UnauthorizedError),
        (403, "forbidden", ForbiddenError),
        (404, "not_found", NotFoundError),
        (409, "conflict", ConflictError),
    ],
)
def test_error_from_response_maps_known_codes(status, code, cls):
    assert isinstance(
        error_from_response(response(status, code=code, message="x")), cls
    )


def test_error_from_response_policy_denied_carries_rule_id():
    err = error_from_response(
        response(403, code="policy_denied", message="nope", rule_id="r1")
    )
    assert isinstance(err, PolicyError)
    assert err.rule_id == "r1"


def test_error_from_response_unknown_code_falls_back_to_api_error():
    err = error_from_response(response(500, code="something_new", message="weird"))
    assert isinstance(err, APIError)
    assert not isinstance(err, (NotFoundError, ValidationError, UnauthorizedError))
    assert err.code == "something_new"
    assert err.status_code == 500


def test_error_from_response_without_an_envelope():
    err = error_from_response(httpx2.Response(502, text="upstream is down"))
    assert isinstance(err, APIError)
    assert err.code == "internal_error"
    assert Exception.__str__(err) == "upstream is down"


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
