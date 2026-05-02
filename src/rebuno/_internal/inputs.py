from __future__ import annotations

import inspect
import typing
from collections.abc import Callable
from typing import Any

try:
    from pydantic import BaseModel
except ImportError:
    BaseModel = None  # type: ignore[assignment,misc]


class InputBinder:
    def __init__(self, fn: Callable[..., Any]):
        self.fn = fn
        sig = inspect.signature(fn)
        self.sig = sig

        # Resolve string annotations (PEP 563 / `from __future__ import annotations`).
        try:
            hints = typing.get_type_hints(fn)
        except Exception:
            hints = {}

        self.shape, self.model = self._classify(sig, hints)
        self.required = [
            name
            for name, p in sig.parameters.items()
            if p.default is inspect.Parameter.empty
            and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        ]

    @staticmethod
    def _classify(
        sig: inspect.Signature,
        hints: dict[str, Any],
    ) -> tuple[str, type | None]:
        params = [p for p in sig.parameters.values() if p.name != "self"]
        if len(params) == 1:
            p = params[0]
            ann = hints.get(p.name, p.annotation)
            if BaseModel is not None and isinstance(ann, type) and issubclass(ann, BaseModel):
                return "model", ann
            if ann is inspect.Parameter.empty or ann is Any or ann is dict:
                return "raw", None
        return "kwargs", None

    def bind(self, claim_input: Any) -> dict[str, Any]:
        """Convert claim.input into the kwargs the handler expects.

        Raises ValueError on validation failure (used to fail the execution
        with a clear message before the handler runs).
        """
        if self.shape == "raw":
            return {next(iter(self.sig.parameters)): claim_input}

        if self.shape == "model":
            assert self.model is not None
            data = claim_input if isinstance(claim_input, dict) else {}
            try:
                model = self.model(**data)
            except Exception as e:
                raise ValueError(f"input validation failed: {e}") from e
            return {next(iter(self.sig.parameters)): model}

        data = claim_input if isinstance(claim_input, dict) else {}
        missing = [name for name in self.required if name not in data]
        if missing:
            raise ValueError(f"missing required input fields: {', '.join(missing)}")
        accepted = {name: data[name] for name in self.sig.parameters if name in data}
        return accepted
