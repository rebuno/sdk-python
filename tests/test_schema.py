"""fn_to_json_schema: Python signatures → JSON Schema for kernel publishing."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from rebuno._internal.schema import fn_to_json_schema


def test_simple_primitives():
    def fn(name: str, count: int) -> None: ...

    schema = fn_to_json_schema(fn)
    assert schema["type"] == "object"
    props = schema["properties"]
    assert props["name"]["type"] == "string"
    assert props["count"]["type"] == "integer"
    assert set(schema["required"]) == {"name", "count"}


def test_optional_via_default():
    def fn(name: str, age: int = 0) -> None: ...

    schema = fn_to_json_schema(fn)
    assert schema["required"] == ["name"]
    assert schema["properties"]["age"]["default"] == 0


def test_optional_type_with_none_default():
    def fn(name: str | None = None) -> None: ...

    schema = fn_to_json_schema(fn)
    # Pydantic produces an anyOf for Optional[X]
    assert "name" not in schema.get("required", [])


def test_annotated_field_description_preserved():
    def fn(
        repo: Annotated[str, Field(description="Repository name")],
        title: Annotated[str, Field(description="PR title")],
    ) -> None: ...

    schema = fn_to_json_schema(fn)
    assert schema["properties"]["repo"]["description"] == "Repository name"
    assert schema["properties"]["title"]["description"] == "PR title"


def test_no_args_returns_empty_object_schema():
    def fn() -> None: ...

    schema = fn_to_json_schema(fn)
    assert schema == {"type": "object", "properties": {}}


class _Address(BaseModel):
    street: str
    city: str


def _fn_with_nested(address: _Address) -> None: ...


def test_nested_pydantic_model():
    schema = fn_to_json_schema(_fn_with_nested)
    assert "address" in schema["properties"]
    # pydantic puts nested models in $defs / via $ref
    assert (
        "$defs" in schema
        or "$ref" in schema["properties"]["address"]
        or "properties" in schema["properties"]["address"]
    )


def test_unannotated_param_is_any():
    def fn(x) -> None: ...

    # Should not raise even without annotation
    schema = fn_to_json_schema(fn)
    assert "x" in schema["properties"]


def test_skips_self_param():
    class C:
        def method(self, x: int) -> None: ...

    schema = fn_to_json_schema(C.method)
    assert "self" not in schema["properties"]
    assert "x" in schema["properties"]


def test_skips_var_args():
    def fn(x: int, *args, **kwargs) -> None: ...

    schema = fn_to_json_schema(fn)
    assert set(schema["properties"]) == {"x"}


def test_no_title_field_in_output():
    def fn(x: int) -> None: ...

    schema = fn_to_json_schema(fn)
    assert "title" not in schema  # we strip pydantic's auto-generated title
