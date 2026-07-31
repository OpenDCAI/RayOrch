"""Identity and lifecycle invariants for V3."""

from __future__ import annotations

import pytest

from rayorch.experimental.multigrain_v3.model import (
    GrainPhase,
    GrainRecord,
    InputBinding,
    ItemRef,
    PortId,
    source_entity,
    stage_grain_id,
)


SALT = bytes(range(16))


def test_stage_grain_identity_ignores_dispatch_and_rebatching():
    """Logical identity depends on Stage/input bindings, not attempts."""

    entity = source_entity(SALT, 7)
    item = ItemRef(PortId(0, 0), entity)
    inputs = (InputBinding("primary", (item,)),)
    first = stage_grain_id(SALT, 3, inputs)
    second = stage_grain_id(SALT, 3, inputs)
    assert first == second

    record = GrainRecord(first, 3, inputs, (PortId(3, 0),))
    token1 = record.reserve(1, 10)
    assert record.phase is GrainPhase.IN_FLIGHT
    assert record.release(token1)
    token2 = record.reserve(1, 11)
    assert token2.grain == token1.grain
    assert token2.generation == token1.generation + 1


def test_run_salt_must_be_exactly_16_bytes():
    """Identity encoding rejects unstable run-salt representations."""

    with pytest.raises(ValueError, match="16 bytes"):
        source_entity(b"short", 0)

