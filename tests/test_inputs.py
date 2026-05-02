"""InputBinder: handler signature → kwargs from claim.input."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from rebuno._internal.inputs import InputBinder


async def _fn_raw(input):
    return None


async def _fn_raw_dict(input: dict):
    return None


def test_raw_shape_unannotated():
    binder = InputBinder(_fn_raw)
    assert binder.shape == "raw"
    assert binder.bind({"a": 1}) == {"input": {"a": 1}}


def test_raw_shape_dict_annotated():
    binder = InputBinder(_fn_raw_dict)
    assert binder.shape == "raw"
    assert binder.bind({"a": 1}) == {"input": {"a": 1}}


def test_raw_shape_passes_non_dict_inputs_through():
    binder = InputBinder(_fn_raw)
    assert binder.bind("scalar") == {"input": "scalar"}
    assert binder.bind(None) == {"input": None}


async def _fn_kwargs(prompt: str, repo_url: str = ""):
    return None


async def _fn_required_only(a: int, b: int):
    return None


def test_kwargs_shape():
    binder = InputBinder(_fn_kwargs)
    assert binder.shape == "kwargs"


def test_kwargs_unpacks_required_and_optional():
    binder = InputBinder(_fn_kwargs)
    assert binder.bind({"prompt": "hi", "repo_url": "x"}) == {"prompt": "hi", "repo_url": "x"}


def test_kwargs_omits_optional_when_absent():
    binder = InputBinder(_fn_kwargs)
    assert binder.bind({"prompt": "hi"}) == {"prompt": "hi"}


def test_kwargs_missing_required_raises_clear_message():
    binder = InputBinder(_fn_required_only)
    with pytest.raises(ValueError, match="missing required input fields: a, b"):
        binder.bind({})

    with pytest.raises(ValueError, match="missing required input fields: b"):
        binder.bind({"a": 1})


def test_kwargs_ignores_extra_input_fields():
    binder = InputBinder(_fn_kwargs)
    bound = binder.bind({"prompt": "hi", "extra": "ignored"})
    assert bound == {"prompt": "hi"}


def test_kwargs_non_dict_input_treated_as_empty():
    binder = InputBinder(_fn_kwargs)
    with pytest.raises(ValueError, match="missing required"):
        binder.bind("scalar")


class _ModelIn(BaseModel):
    prompt: str
    repo_url: str = ""


async def _fn_model(input: _ModelIn):
    return None


def test_model_shape():
    binder = InputBinder(_fn_model)
    assert binder.shape == "model"
    assert binder.model is _ModelIn


def test_model_validates_and_constructs():
    binder = InputBinder(_fn_model)
    bound = binder.bind({"prompt": "hi", "repo_url": "x"})
    assert isinstance(bound["input"], _ModelIn)
    assert bound["input"].prompt == "hi"
    assert bound["input"].repo_url == "x"


def test_model_validation_error_wrapped():
    binder = InputBinder(_fn_model)
    with pytest.raises(ValueError, match="input validation failed"):
        binder.bind({})  # missing required prompt


def test_model_with_non_dict_input():
    binder = InputBinder(_fn_model)
    with pytest.raises(ValueError, match="input validation failed"):
        binder.bind("not-a-dict")
