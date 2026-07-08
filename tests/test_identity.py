import json
from pathlib import Path

import pytest

from rebuno.identity import args_hash, canonical_json, compute_step_id

VECTORS = json.loads((Path(__file__).parent / "fixtures" / "identity_vectors.json").read_text())


@pytest.mark.parametrize("v", VECTORS, ids=[v["target"] for v in VECTORS])
def test_matches_kernel(v):
    # The generator embeds raw JSON via json.RawMessage, so loading the fixture
    # yields already-decoded Python objects (including bare scalars such as the
    # string "plain string"). Use the value as-is rather than re-parsing it.
    args = v["args"]
    assert canonical_json(args).decode("utf-8") == v["canonical"]
    assert args_hash(args) == v["args_hash"]
    assert compute_step_id(v["execution_id"], v["kind"], v["target"], v["args_hash"], v["occurrence"]) == v["step_id"]


def test_key_ordering_is_value_independent():
    assert args_hash({"a": 1, "b": 2}) == args_hash({"b": 2, "a": 1})


def test_bool_not_treated_as_int():
    assert canonical_json({"x": True}) == b'{"x":true}'
    assert canonical_json({"x": 1}) == b'{"x":1}'
