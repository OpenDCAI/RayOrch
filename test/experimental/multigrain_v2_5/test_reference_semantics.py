from __future__ import annotations

import json
from pathlib import Path

from .reference_semantics.interpreter import (
    ReferenceInterpreter,
    SourcePositionOracle,
)
from .reference_semantics.models import (
    Failed,
    GrainId,
    ItemRef,
    MemberSettlement,
    PortId,
    Primitive,
    RoleItems,
    Success,
    Suppressed,
)
from .reference_semantics.semantic_cases import (
    EXPAND_PORT_A,
    EXPAND_PORT_B,
    FILTER_PORT,
    MAP_PORT,
    REDUCE_PORT,
    RELATE_PORT,
    RUN_SALT,
    SOURCE_PORT,
    snapshot_record,
)


GOLDEN = Path(__file__).with_name("golden") / "semantic_snapshot.json"


def _source_item(interpreter: ReferenceInterpreter, position: int = 0) -> ItemRef:
    record = interpreter.source(SOURCE_PORT, position)
    assert isinstance(record.outcome, Success)
    return record.outcome.emissions_by_port[0][0].item


def _child_inputs(parent: ItemRef) -> tuple[RoleItems, ...]:
    return (RoleItems("parent", (parent,)),)


def _member_items(expand_record, port_index: int, members_port: PortId):
    assert isinstance(expand_record.outcome, Success)
    return tuple(
        ItemRef(members_port, emission.item.entity)
        for emission in expand_record.outcome.emissions_by_port[port_index]
    )


def test_compiled_source_kind_and_run_global_position_oracle():
    assert Primitive.SOURCE.value == "source"
    positions = SourcePositionOracle()
    other_port = PortId(1, 0)

    assert [positions.allocate(SOURCE_PORT) for _ in range(4)] == [0, 1, 2, 3]
    assert positions.allocate(other_port) == 0
    assert positions.peek(SOURCE_PORT) == 4

    interpreter = ReferenceInterpreter(RUN_SALT)
    first = interpreter.source(SOURCE_PORT, 0)
    second_arena = interpreter.source(SOURCE_PORT, 2)
    assert first.node == SOURCE_PORT.node
    assert first.id != second_arena.id


def test_source_success_and_failed_before_output():
    interpreter = ReferenceInterpreter(RUN_SALT)
    success = interpreter.source(SOURCE_PORT, 0)
    failed = interpreter.source(SOURCE_PORT, 1, failure="decode")

    assert isinstance(success.outcome, Success)
    assert len(success.output_slots) == 1
    assert success.outcome.emissions_by_port[0][0].item == success.output_slots[0]

    assert isinstance(failed.outcome, Failed)
    assert len(failed.output_slots) == 1
    assert failed.outcome.failure.direct_causes == ()


def test_map_success_failure_and_suppression():
    interpreter = ReferenceInterpreter(RUN_SALT)
    item = _source_item(interpreter)
    inputs = (RoleItems("primary", (item,)),)
    upstream_failure = GrainId(bytes.fromhex("11" * 16))

    success = interpreter.map(20, inputs, (MAP_PORT,))
    failed = interpreter.map(20, inputs, (MAP_PORT,), failure="bad")
    suppressed = interpreter.map(
        20,
        inputs,
        (MAP_PORT,),
        suppressed_by=(upstream_failure,),
    )

    assert isinstance(success.outcome, Success)
    assert success.outcome.emissions_by_port[0][0].item.entity == item.entity
    assert isinstance(failed.outcome, Failed)
    assert failed.outcome.failure.direct_causes == (item,)
    assert suppressed.outcome == Suppressed((upstream_failure,))
    assert success.id == failed.id == suppressed.id


def test_filter_true_false_error_and_suppression():
    interpreter = ReferenceInterpreter(RUN_SALT)
    item = _source_item(interpreter)
    inputs = (RoleItems("target", (item,)),)
    cause = GrainId(bytes.fromhex("22" * 16))

    kept = interpreter.filter(30, inputs, (FILTER_PORT,), keep=True)
    dropped = interpreter.filter(30, inputs, (FILTER_PORT,), keep=False)
    failed = interpreter.filter(
        30,
        inputs,
        (FILTER_PORT,),
        failure="predicate",
    )
    suppressed = interpreter.filter(
        30,
        inputs,
        (FILTER_PORT,),
        suppressed_by=(cause,),
    )

    assert len(kept.outcome.emissions_by_port[0]) == 1  # type: ignore[union-attr]
    assert dropped.outcome == Success(((),))
    assert isinstance(failed.outcome, Failed)
    assert suppressed.outcome == Suppressed((cause,))


def test_expand_nonempty_empty_and_multiple_ports_share_layout():
    interpreter = ReferenceInterpreter(RUN_SALT)
    parent = _source_item(interpreter)

    nonempty = interpreter.expand(
        10,
        _child_inputs(parent),
        (EXPAND_PORT_A, EXPAND_PORT_B),
        3,
    )
    empty = interpreter.expand(
        10,
        _child_inputs(parent),
        (EXPAND_PORT_A, EXPAND_PORT_B),
        0,
    )

    assert nonempty.output_slots == ()
    assert isinstance(nonempty.outcome, Success)
    port_a, port_b = nonempty.outcome.emissions_by_port
    assert [emission.ordinal for emission in port_a] == [0, 1, 2]
    assert [emission.ordinal for emission in port_b] == [0, 1, 2]
    assert [emission.item.entity for emission in port_a] == [
        emission.item.entity for emission in port_b
    ]
    assert empty.outcome == Success(((), ()))


def test_reduce_normal_empty_all_filtered_and_partial_filtered():
    interpreter = ReferenceInterpreter(RUN_SALT)
    anchor = _source_item(interpreter)
    expanded = interpreter.expand(
        10,
        _child_inputs(anchor),
        (EXPAND_PORT_A,),
        3,
    )
    members = _member_items(expanded, 0, MAP_PORT)

    normal = interpreter.reduce(
        40,
        anchor,
        MAP_PORT,
        (REDUCE_PORT,),
        expected=3,
        settlements=tuple(
            MemberSettlement.present(index, item)
            for index, item in enumerate(members)
        ),
    )
    empty = interpreter.reduce(
        40,
        anchor,
        MAP_PORT,
        (REDUCE_PORT,),
        expected=0,
    )
    all_filtered = interpreter.reduce(
        40,
        anchor,
        MAP_PORT,
        (REDUCE_PORT,),
        expected=3,
        settlements=tuple(MemberSettlement.dropped(i) for i in range(3)),
    )
    partial = interpreter.reduce(
        40,
        anchor,
        MAP_PORT,
        (REDUCE_PORT,),
        expected=3,
        settlements=(
            MemberSettlement.present(0, members[0]),
            MemberSettlement.dropped(1),
            MemberSettlement.present(2, members[2]),
        ),
    )

    assert isinstance(normal.outcome, Success)
    assert normal.inputs[1].items == members
    assert isinstance(empty.outcome, Success)
    assert empty.inputs[1].items == ()
    assert isinstance(all_filtered.outcome, Success)
    assert all_filtered.inputs[1].items == ()
    assert partial.inputs[1].items == (members[0], members[2])


def test_known_n_failure_has_canonical_members_and_causes():
    interpreter = ReferenceInterpreter(RUN_SALT)
    anchor = _source_item(interpreter)
    expanded = interpreter.expand(
        10,
        _child_inputs(anchor),
        (EXPAND_PORT_A,),
        4,
    )
    members = _member_items(expanded, 0, MAP_PORT)
    receipt_1 = GrainId(bytes.fromhex("31" * 16))
    receipt_3 = GrainId(bytes.fromhex("33" * 16))
    completion_order = (
        MemberSettlement.suppressed(3, members[3], receipt_3),
        MemberSettlement.dropped(2),
        MemberSettlement.failed(1, members[1], receipt_1),
        MemberSettlement.present(0, members[0]),
    )

    reduced = interpreter.reduce(
        40,
        anchor,
        MAP_PORT,
        (REDUCE_PORT,),
        expected=4,
        settlements=completion_order,
    )

    assert reduced.inputs[1].items == (members[0], members[1], members[3])
    assert reduced.outcome == Suppressed((receipt_1, receipt_3))


def test_origin_expand_failed_before_output_is_anchor_only_suppression():
    interpreter = ReferenceInterpreter(RUN_SALT)
    anchor = _source_item(interpreter)
    origin = interpreter.expand(
        10,
        _child_inputs(anchor),
        (EXPAND_PORT_A,),
        None,
        failure="fanout",
    )

    reduced = interpreter.reduce(
        40,
        anchor,
        MAP_PORT,
        (REDUCE_PORT,),
        expected=None,
        origin_failure=origin.id,
    )

    assert reduced.inputs == (RoleItems("anchor", (anchor,)),)
    assert reduced.outcome == Suppressed((origin.id,))


def test_relate_int_key_1_to_1_m_to_n_unmatched_and_sealing():
    interpreter = ReferenceInterpreter(RUN_SALT)
    left_port = PortId(51, 0)
    right_port = PortId(52, 0)
    left = tuple(
        ItemRef(left_port, _source_item(interpreter, position).entity)
        for position in range(3)
    )
    right = tuple(
        ItemRef(right_port, _source_item(interpreter, position + 3).entity)
        for position in range(3)
    )
    roles = {
        "left": ((left[0], 1), (left[1], 2), (left[2], 2)),
        "right": ((right[0], 1), (right[1], 2), (right[2], 9)),
    }

    open_result = interpreter.relate(
        50,
        roles,
        (RELATE_PORT,),
        sealed_roles=frozenset({"left"}),
    )
    sealed = interpreter.relate(
        50,
        roles,
        (RELATE_PORT,),
        sealed_roles=frozenset(roles),
    )

    assert not open_result.complete
    assert open_result.unmatched == ()
    assert sealed.complete
    assert len(sealed.records) == 3  # key 1: 1x1; key 2: 2x1
    assert sealed.unmatched == (right[2],)


def _golden_snapshot():
    interpreter = ReferenceInterpreter(RUN_SALT)
    source = interpreter.source(SOURCE_PORT, 7)
    anchor = source.output_slots[0]
    expand = interpreter.expand(
        10,
        _child_inputs(anchor),
        (EXPAND_PORT_A, EXPAND_PORT_B),
        3,
    )
    members = _member_items(expand, 0, FILTER_PORT)
    reduce_record = interpreter.reduce(
        40,
        anchor,
        FILTER_PORT,
        (REDUCE_PORT,),
        expected=3,
        settlements=(
            MemberSettlement.present(0, members[0]),
            MemberSettlement.dropped(1),
            MemberSettlement.present(2, members[2]),
        ),
    )
    return {
        "source": snapshot_record(source),
        "expand": snapshot_record(expand),
        "reduce": snapshot_record(reduce_record),
    }


def test_semantic_snapshot_matches_golden():
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert _golden_snapshot() == expected
