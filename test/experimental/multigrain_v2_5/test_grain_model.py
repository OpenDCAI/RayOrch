from __future__ import annotations

from dataclasses import fields

import pytest

from rayorch.experimental.multigrain_v2_5.grain import (
    ConsumerIndex,
    Emission,
    EntityId,
    Failed,
    GrainFailure,
    GrainId,
    GrainInvariantError,
    GrainPhase,
    GrainRecord,
    GrainTable,
    ItemRef,
    PortId,
    PortIndex,
    ProducerIndex,
    RoleItems,
    Success,
    Suppressed,
    ValueIndex,
)


def _ids():
    entity = EntityId(bytes.fromhex("01" * 16))
    item = ItemRef(PortId(0, 0), entity)
    grain = GrainId(bytes.fromhex("02" * 16))
    output = ItemRef(PortId(1, 0), entity)
    return item, grain, output


def _ready() -> GrainRecord:
    item, grain, output = _ids()
    return GrainRecord(
        id=grain,
        node=1,
        inputs=(RoleItems("primary", (item,)),),
        output_slots=(output,),
    )


def test_grain_record_field_allowlist_is_exact():
    assert [field.name for field in fields(GrainRecord)] == [
        "id",
        "node",
        "inputs",
        "output_slots",
        "outcome",
        "phase",
        "generation",
        "active",
        "infra_failures",
    ]


def test_grain_lifecycle_has_only_three_legal_states():
    record = _ready()
    assert record.phase is GrainPhase.READY

    token = record.reserve(arena=3, dispatch=7)
    assert record.phase is GrainPhase.IN_FLIGHT
    assert record.active == token

    assert record.retry_infrastructure_failure(token)
    assert record.phase is GrainPhase.READY
    assert record.infra_failures == 1

    second = record.reserve(arena=3, dispatch=8)
    outcome = Success(((Emission(record.output_slots[0], 0),),))
    assert record.seal(outcome, token=second)
    assert record.phase is GrainPhase.SEALED
    assert record.outcome == outcome


def test_stale_attempt_cannot_seal_or_retry_current_generation():
    record = _ready()
    stale = record.reserve(arena=1, dispatch=1)
    assert record.retry_infrastructure_failure(stale)
    current = record.reserve(arena=1, dispatch=2)

    assert not record.retry_infrastructure_failure(stale)
    assert not record.seal(Failed(GrainFailure("x", "stale", ())), token=stale)
    assert record.active == current


def test_indexes_distinguish_expected_slots_from_actual_emissions():
    record = _ready()
    producer = ProducerIndex()
    consumers = ConsumerIndex()
    ports = PortIndex()
    values = ValueIndex()

    producer.register(record)
    consumers.register(record)
    ports.register(Success(((),)))

    assert producer.get(record.output_slots[0]) == record.id
    assert consumers.get(record.inputs[0].items[0]) == (record.id,)
    assert ports.items(record.output_slots[0].port) == ()

    emitted = Success(((Emission(record.output_slots[0], 0),),))
    ports.register(emitted)
    values.put(record.output_slots[0], ("block", 3))
    assert ports.items(record.output_slots[0].port) == record.output_slots
    assert values.get(record.output_slots[0]) == ("block", 3)


@pytest.mark.parametrize(
    "phase_outcome",
    [
        (GrainPhase.READY, Success(((),))),
        (GrainPhase.SEALED, None),
        (GrainPhase.IN_FLIGHT, None),
    ],
)
def test_illegal_grain_states_are_rejected(phase_outcome):
    item, grain, output = _ids()
    phase, outcome = phase_outcome
    with pytest.raises(GrainInvariantError):
        GrainRecord(
            grain,
            1,
            (RoleItems("primary", (item,)),),
            (output,),
            outcome=outcome,
            phase=phase,
        )


def test_grain_table_ensure_is_idempotent_and_detects_classification_conflict():
    template = _ready()
    table = GrainTable()
    assert table.ensure_executable(template) is template
    assert table.ensure_executable(_ready()) is template
    assert len(table) == 1

    suppressed = GrainRecord.sealed(
        id=template.id,
        node=template.node,
        inputs=template.inputs,
        output_slots=template.output_slots,
        outcome=Suppressed((GrainId(bytes.fromhex("03" * 16)),)),
    )
    with pytest.raises(GrainInvariantError, match="classification"):
        table.ensure_suppressed(suppressed)
