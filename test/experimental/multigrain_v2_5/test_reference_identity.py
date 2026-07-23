from __future__ import annotations

import json
from pathlib import Path

import pytest

from .reference_semantics.identity_oracle import (
    canonical_encode,
    expand_entity,
    expand_grain_id,
    filter_grain_id,
    map_grain_id,
    reduce_grain_id,
    relate_entity,
    relate_grain_id,
    source_entity,
    source_grain_id,
)
from .reference_semantics.models import ItemRef, PortId, RoleItems
from .reference_semantics.semantic_cases import RUN_SALT, SOURCE_PORT


GOLDEN = Path(__file__).with_name("golden") / "identity_vectors.json"


def test_canonical_scalar_framing_is_exact():
    assert canonical_encode(None).hex() == "6e"
    assert canonical_encode(False).hex() == "6200"
    assert canonical_encode(True).hex() == "6201"
    assert canonical_encode(0).hex() == "69000000000000000000"
    assert canonical_encode(1).hex() == "6900000000000000000101"
    assert canonical_encode(-1).hex() == "6901000000000000000101"
    assert canonical_encode(b"A").hex() == "79000000000000000141"
    assert canonical_encode("A").hex() == "73000000000000000141"
    assert canonical_encode(()).hex() == "740000000000000000"


def test_canonical_encoding_is_type_sensitive_and_closed():
    assert canonical_encode(True) != canonical_encode(1)
    assert canonical_encode(b"x") != canonical_encode("x")
    with pytest.raises(TypeError):
        canonical_encode([1, 2])
    with pytest.raises(TypeError):
        canonical_encode({"x": 1})
    with pytest.raises(TypeError):
        canonical_encode(1.0)


@pytest.mark.parametrize("bad_salt", [b"", b"x" * 15, b"x" * 17, "not-bytes"])
def test_run_salt_is_exactly_sixteen_bytes(bad_salt):
    with pytest.raises(ValueError, match="16 bytes"):
        source_entity(bad_salt, SOURCE_PORT, 0)  # type: ignore[arg-type]


def _actual_vectors() -> dict[str, str]:
    source = source_entity(RUN_SALT, SOURCE_PORT, 7)
    source_item = ItemRef(SOURCE_PORT, source)
    unary_inputs = (RoleItems("primary", (source_item,)),)
    relation_inputs = (
        RoleItems("left", (source_item,)),
        RoleItems("right", (ItemRef(PortId(1, 0), source),)),
    )
    return {
        "source_entity": source.hex(),
        "source_grain": source_grain_id(
            RUN_SALT,
            SOURCE_PORT,
            7,
        ).hex(),
        "map_grain": map_grain_id(RUN_SALT, 10, unary_inputs).hex(),
        "filter_grain": filter_grain_id(
            RUN_SALT,
            11,
            unary_inputs,
        ).hex(),
        "expand_grain": expand_grain_id(
            RUN_SALT,
            12,
            unary_inputs,
        ).hex(),
        "expand_entity_0": expand_entity(
            RUN_SALT,
            12,
            source,
            0,
        ).hex(),
        "expand_entity_1": expand_entity(
            RUN_SALT,
            12,
            source,
            1,
        ).hex(),
        "reduce_grain": reduce_grain_id(
            RUN_SALT,
            13,
            source_item,
        ).hex(),
        "relate_entity": relate_entity(
            RUN_SALT,
            14,
            relation_inputs,
        ).hex(),
        "relate_grain": relate_grain_id(
            RUN_SALT,
            14,
            relation_inputs,
        ).hex(),
    }


def test_identity_vectors_match_frozen_golden():
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert _actual_vectors() == expected


def test_role_order_is_semantic_not_sorted_by_name():
    source = source_entity(RUN_SALT, SOURCE_PORT, 0)
    left = ItemRef(PortId(1, 0), source)
    right = ItemRef(PortId(2, 0), source)
    left_right = (
        RoleItems("left", (left,)),
        RoleItems("right", (right,)),
    )
    right_left = (
        RoleItems("right", (right,)),
        RoleItems("left", (left,)),
    )
    assert relate_grain_id(RUN_SALT, 20, left_right) != relate_grain_id(
        RUN_SALT,
        20,
        right_left,
    )
