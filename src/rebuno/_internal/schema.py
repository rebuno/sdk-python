from __future__ import annotations

import inspect
import typing
from collections.abc import Callable
from typing import Any

from pydantic import create_model


def fn_to_json_schema(fn: Callable[..., Any]) -> dict[str, Any]:
    """Build a JSON Schema describing the function's keyword arguments.

    Returns a JSON Schema "object" with one property per parameter (excluding
    self/cls). Required-ness comes from whether the parameter has a default.
    """
    sig = inspect.signature(fn)
    try:
        hints = typing.get_type_hints(fn, include_extras=True)
    except Exception:
        hints = {}

    fields: dict[str, tuple[Any, Any]] = {}
    for name, p in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        ann = hints.get(name, p.annotation)
        if ann is inspect.Parameter.empty:
            ann = Any
        default = ... if p.default is inspect.Parameter.empty else p.default
        fields[name] = (ann, default)

    if not fields:
        return {"type": "object", "properties": {}}

    Model = create_model(_safe_name(fn), **fields)  # type: ignore[call-overload]
    schema = Model.model_json_schema()
    # Strip pydantic's auto-generated title; tool ID is the canonical name.
    schema.pop("title", None)
    return schema


def _safe_name(fn: Callable[..., Any]) -> str:
    name = getattr(fn, "__name__", "Tool")
    # pydantic create_model wants a valid Python identifier
    safe = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
    if not safe or not (safe[0].isalpha() or safe[0] == "_"):
        safe = "Tool_" + safe
    return safe + "Input"
