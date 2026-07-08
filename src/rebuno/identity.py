"""Deterministic step identity, byte-compatible with the Go kernel.

The kernel recomputes and validates the step ID (409 on mismatch), so the
canonical JSON encoding here must match Go's encoding/json output exactly:
sorted keys, no whitespace, number literals preserved, and Go-style string
escaping (escape <, >, &, U+2028, U+2029; raw UTF-8 otherwise).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _encode_string(s: str) -> str:
    out = ['"']
    for ch in s:
        o = ord(ch)
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif o < 0x20:
            out.append(f"\\u{o:04x}")
        elif ch == "<":
            out.append("\\u003c")
        elif ch == ">":
            out.append("\\u003e")
        elif ch == "&":
            out.append("\\u0026")
        elif o == 0x2028:
            out.append("\\u2028")
        elif o == 0x2029:
            out.append("\\u2029")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _encode(v: Any) -> str:
    if v is None:
        return "null"
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, str):
        return _encode_string(v)
    if isinstance(v, (int, float)):
        # json.dumps yields a JSON-valid number literal; the SDK both sends and
        # hashes this exact literal, and the kernel preserves it (json.Number).
        return json.dumps(v)
    if isinstance(v, dict):
        # Go sorts keys by UTF-8 byte order, which equals Python code-point order.
        items = sorted(v.items(), key=lambda kv: kv[0])
        return "{" + ",".join(_encode_string(str(k)) + ":" + _encode(val) for k, val in items) + "}"
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(_encode(el) for el in v) + "]"
    raise TypeError(f"cannot canonicalize value of type {type(v).__name__}")


def canonical_json(value: Any) -> bytes:
    """Canonical JSON bytes matching the kernel's CanonicalizeJSON."""
    return _encode(value).encode("utf-8")


def args_hash(args: Any) -> str:
    """hex(sha256(canonical_json(args)))."""
    return hashlib.sha256(canonical_json(args)).hexdigest()


def compute_step_id(execution_id: str, kind: str, target: str, args_hash_value: str, occurrence: int) -> str:
    """hex(sha256( Σ f"{len(bytes(field))}:" + bytes(field) ))."""
    fields = [execution_id, kind, target, args_hash_value, str(occurrence)]
    buf = bytearray()
    for f in fields:
        fb = f.encode("utf-8")
        buf += f"{len(fb)}:".encode()
        buf += fb
    return hashlib.sha256(bytes(buf)).hexdigest()
