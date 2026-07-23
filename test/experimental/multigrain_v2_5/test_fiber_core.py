from __future__ import annotations

from rayorch.experimental.multigrain_v2_5.graph import (
    BindingReceipt,
    PlanAction,
    admit_source,
    expand_success,
    plan_expand,
    plan_reduce,
    register_expand_origins,
)
from rayorch.experimental.multigrain_v2_5.grain import (
    ExpandOriginIndex,
    FiberBarrier,
    FiberId,
    FiberState,
    GrainId,
    ItemRef,
    Success,
    Suppressed,
)

from .reference_semantics.semantic_cases import RUN_SALT
from .support import expand_node, reduce_node, source_node


def _expanded(count: int = 3):
    source_spec = source_node(0)
    source = admit_source(source_spec, RUN_SALT, 0)
    anchor = source.output_slots[0]
    expand_spec = expand_node(1, anchor.port, outputs=2)
    decision = plan_expand(
        expand_spec,
        RUN_SALT,
        (BindingReceipt.present("parent", anchor),),
    )
    assert decision.grain is not None
    outcome = expand_success(
        expand_spec,
        decision.grain,
        RUN_SALT,
        cardinality=count,
    )
    return anchor, expand_spec, decision.grain, outcome


def _members(outcome: Success, port: int, members_port):
    return tuple(
        ItemRef(members_port, emission.item.entity)
        for emission in outcome.emissions_by_port[port]
    )


def test_expand_origin_index_is_shared_across_output_ports():
    anchor, _, record, outcome = _expanded(3)
    index = ExpandOriginIndex()
    register_expand_origins(index, record, outcome)

    for port in outcome.emissions_by_port:
        for emission in port:
            origin = index.get(emission.item.entity)
            assert origin is not None
            assert origin.anchor == anchor
            assert origin.ordinal == emission.ordinal
            assert origin.origin == record.id


def test_fiber_barrier_normal_empty_all_filtered_and_partial():
    anchor, _, origin, outcome = _expanded(3)
    reduce_spec = reduce_node(4, anchor.port, outcome.emissions_by_port[0][0].item.port)
    members_port = reduce_spec.reduce_members
    assert members_port is not None
    members = _members(outcome, 0, members_port)

    normal = FiberBarrier(FiberId(4, anchor), origin.id)
    normal.set_expected(3)
    for ordinal, item in enumerate(members):
        normal.settle_present(ordinal, item)
    normal_decision = plan_reduce(
        reduce_spec,
        RUN_SALT,
        BindingReceipt.present("anchor", anchor),
        normal,
    )
    assert normal.state is FiberState.READY
    assert normal_decision.action is PlanAction.ENSURE_EXECUTABLE
    assert normal_decision.grain is not None
    assert normal_decision.grain.inputs[1].items == members

    empty = FiberBarrier(FiberId(4, anchor), origin.id)
    empty.set_expected(0)
    empty_decision = plan_reduce(
        reduce_spec,
        RUN_SALT,
        BindingReceipt.present("anchor", anchor),
        empty,
    )
    assert empty_decision.action is PlanAction.ENSURE_EXECUTABLE
    assert empty_decision.grain.inputs[1].items == ()  # type: ignore[union-attr]

    filtered = FiberBarrier(FiberId(4, anchor), origin.id)
    filtered.set_expected(3)
    for ordinal in range(3):
        filtered.settle_dropped(ordinal)
    filtered_decision = plan_reduce(
        reduce_spec,
        RUN_SALT,
        BindingReceipt.present("anchor", anchor),
        filtered,
    )
    assert filtered_decision.grain.inputs[1].items == ()  # type: ignore[union-attr]

    partial = FiberBarrier(FiberId(4, anchor), origin.id)
    partial.set_expected(3)
    partial.settle_present(0, members[0])
    partial.settle_dropped(1)
    partial.settle_present(2, members[2])
    partial_decision = plan_reduce(
        reduce_spec,
        RUN_SALT,
        BindingReceipt.present("anchor", anchor),
        partial,
    )
    assert partial_decision.grain.inputs[1].items == (  # type: ignore[union-attr]
        members[0],
        members[2],
    )


def test_known_n_failure_waits_for_closure_and_canonicalizes_causes():
    anchor, _, origin, outcome = _expanded(4)
    reduce_spec = reduce_node(4, anchor.port, outcome.emissions_by_port[0][0].item.port)
    members_port = reduce_spec.reduce_members
    assert members_port is not None
    members = _members(outcome, 0, members_port)
    first = GrainId(bytes.fromhex("11" * 16))
    third = GrainId(bytes.fromhex("33" * 16))
    barrier = FiberBarrier(FiberId(4, anchor), origin.id)
    barrier.set_expected(4)
    barrier.settle_failed(3, members[3], third)
    barrier.settle_present(0, members[0])

    waiting = plan_reduce(
        reduce_spec,
        RUN_SALT,
        BindingReceipt.present("anchor", anchor),
        barrier,
    )
    assert waiting.action is PlanAction.WAIT

    barrier.settle_failed(1, members[1], first)
    barrier.settle_dropped(2)
    suppressed = plan_reduce(
        reduce_spec,
        RUN_SALT,
        BindingReceipt.present("anchor", anchor),
        barrier,
    )
    assert suppressed.action is PlanAction.ENSURE_SUPPRESSED
    assert suppressed.grain is not None
    assert suppressed.grain.inputs[1].items == (
        members[0],
        members[1],
        members[3],
    )
    assert suppressed.grain.outcome == Suppressed((first, third))


def test_unknown_n_origin_failure_creates_anchor_only_suppression():
    anchor, _, origin, outcome = _expanded(1)
    reduce_spec = reduce_node(4, anchor.port, outcome.emissions_by_port[0][0].item.port)
    barrier = FiberBarrier(FiberId(4, anchor), origin.id)
    barrier.block_origin(origin.id)

    decision = plan_reduce(
        reduce_spec,
        RUN_SALT,
        BindingReceipt.present("anchor", anchor),
        barrier,
    )
    assert decision.action is PlanAction.ENSURE_SUPPRESSED
    assert decision.grain is not None
    assert len(decision.grain.inputs) == 1
    assert decision.grain.outcome == Suppressed((origin.id,))
