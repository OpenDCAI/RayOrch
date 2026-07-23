from __future__ import annotations

import pytest

from rayorch.experimental.multigrain_v2.identity import (
    BatchId,
    IdentityKey,
    MicrobatchAttempt,
    RowId,
    RowRef,
    derive_expand_domain,
    derive_expand_entity,
    derive_input_port_id,
    derive_input_salt,
    derive_node_id,
    derive_port_id,
    derive_source_domain,
    derive_source_position_entity,
)


def test_entity_identity_excludes_batch_attempt_port_and_output_slot() -> None:
    salt = derive_input_salt("m", "P", "rows", 0)
    source_domain = derive_source_domain(salt)
    parent = derive_source_position_entity(source_domain, 5)
    node = derive_node_id("m", "P", "expand", "Expand")
    child = derive_expand_entity(node, source_domain, parent, 2)

    batch_a = BatchId(b"a")
    batch_b = BatchId(b"b")
    domain = derive_expand_domain(node)
    assert IdentityKey(batch_a, domain, child).entity == IdentityKey(
        batch_b, domain, child
    ).entity
    assert derive_port_id(node, 0) != derive_port_id(node, 1)
    assert child == derive_expand_entity(node, source_domain, parent, 2)
    assert MicrobatchAttempt(0) != MicrobatchAttempt(1)


def test_row_refs_are_physical_locators_not_entity_identity() -> None:
    input_port = derive_input_port_id(
        derive_input_salt("m", "P", "rows", 0)
    )
    first = RowRef(BatchId(b"batch"), input_port, RowId(0))
    second = RowRef(BatchId(b"batch"), input_port, RowId(1))
    assert first != second


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_non_negative_integer_wrappers_reject_invalid_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        RowId(value)  # type: ignore[arg-type]
    with pytest.raises((TypeError, ValueError)):
        MicrobatchAttempt(value)  # type: ignore[arg-type]


def test_derivation_ordinals_and_slots_reject_bool() -> None:
    node = derive_node_id("m", "P", "expand", "Expand")
    salt = derive_input_salt("m", "P", "rows", 0)
    domain = derive_source_domain(salt)
    parent = derive_source_position_entity(domain, 0)
    with pytest.raises(ValueError):
        derive_port_id(node, True)
    with pytest.raises(ValueError):
        derive_input_salt("m", "P", "rows", True)
    with pytest.raises(ValueError):
        derive_source_position_entity(domain, True)
    with pytest.raises(ValueError):
        derive_expand_entity(node, domain, parent, True)
