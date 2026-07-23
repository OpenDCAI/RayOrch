from __future__ import annotations

import json
from pathlib import Path

import pytest

from rayorch.experimental.multigrain_v2_5.grain import (
    ItemRef,
    PortId,
    RoleItems,
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

from .reference_semantics import identity_oracle
from .reference_semantics.models import (
    ItemRef as OracleItemRef,
    PortId as OraclePortId,
    RoleItems as OracleRoleItems,
)
from .reference_semantics.semantic_cases import RUN_SALT


GOLDEN = Path(__file__).with_name("golden") / "identity_vectors.json"


def _production_vectors() -> dict[str, str]:
    source_port = PortId(0, 0)
    source = source_entity(RUN_SALT, source_port, 7)
    source_item = ItemRef(source_port, source)
    unary_inputs = (RoleItems("primary", (source_item,)),)
    relation_inputs = (
        RoleItems("left", (source_item,)),
        RoleItems("right", (ItemRef(PortId(1, 0), source),)),
    )
    return {
        "source_entity": source.hex(),
        "source_grain": source_grain_id(RUN_SALT, source_port, 7).hex(),
        "map_grain": map_grain_id(RUN_SALT, 10, unary_inputs).hex(),
        "filter_grain": filter_grain_id(RUN_SALT, 11, unary_inputs).hex(),
        "expand_grain": expand_grain_id(RUN_SALT, 12, unary_inputs).hex(),
        "expand_entity_0": expand_entity(RUN_SALT, 12, source, 0).hex(),
        "expand_entity_1": expand_entity(RUN_SALT, 12, source, 1).hex(),
        "reduce_grain": reduce_grain_id(RUN_SALT, 13, source_item).hex(),
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


def _oracle_vectors() -> dict[str, str]:
    source_port = OraclePortId(0, 0)
    source = identity_oracle.source_entity(RUN_SALT, source_port, 7)
    source_item = OracleItemRef(source_port, source)
    unary_inputs = (OracleRoleItems("primary", (source_item,)),)
    relation_inputs = (
        OracleRoleItems("left", (source_item,)),
        OracleRoleItems(
            "right",
            (OracleItemRef(OraclePortId(1, 0), source),),
        ),
    )
    return {
        "source_entity": source.hex(),
        "source_grain": identity_oracle.source_grain_id(
            RUN_SALT,
            source_port,
            7,
        ).hex(),
        "map_grain": identity_oracle.map_grain_id(
            RUN_SALT,
            10,
            unary_inputs,
        ).hex(),
        "filter_grain": identity_oracle.filter_grain_id(
            RUN_SALT,
            11,
            unary_inputs,
        ).hex(),
        "expand_grain": identity_oracle.expand_grain_id(
            RUN_SALT,
            12,
            unary_inputs,
        ).hex(),
        "expand_entity_0": identity_oracle.expand_entity(
            RUN_SALT,
            12,
            source,
            0,
        ).hex(),
        "expand_entity_1": identity_oracle.expand_entity(
            RUN_SALT,
            12,
            source,
            1,
        ).hex(),
        "reduce_grain": identity_oracle.reduce_grain_id(
            RUN_SALT,
            13,
            source_item,
        ).hex(),
        "relate_entity": identity_oracle.relate_entity(
            RUN_SALT,
            14,
            relation_inputs,
        ).hex(),
        "relate_grain": identity_oracle.relate_grain_id(
            RUN_SALT,
            14,
            relation_inputs,
        ).hex(),
    }


def test_production_and_oracle_match_frozen_identity_vectors():
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert _production_vectors() == golden
    assert _oracle_vectors() == golden


@pytest.mark.parametrize(
    "value",
    [None, False, True, 0, 1, -1, b"x", "雪", (), (1, "x")],
)
def test_production_and_oracle_canonical_bytes_match(value):
    assert canonical_encode(value) == identity_oracle.canonical_encode(value)


def test_production_encoder_rejects_mutable_identity_containers():
    with pytest.raises(TypeError):
        canonical_encode([1])
