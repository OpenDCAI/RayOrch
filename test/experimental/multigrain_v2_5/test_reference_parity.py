from __future__ import annotations

from typing import Any

from rayorch.experimental.multigrain_v2_5.graph import (
    BindingReceipt,
    admit_source,
    expand_success,
    failed_outcome,
    filter_success,
    map_success,
    plan_expand,
    plan_filter,
    plan_map,
    plan_reduce,
)
from rayorch.experimental.multigrain_v2_5.grain import (
    FiberBarrier,
    FiberId,
    GrainId,
    ItemRef,
    RoleItems,
)

from .reference_semantics.interpreter import ReferenceInterpreter
from .reference_semantics.models import (
    GrainId as OracleGrainId,
    ItemRef as OracleItemRef,
    MemberSettlement,
    PortId as OraclePortId,
    RoleItems as OracleRoleItems,
)
from .reference_semantics.semantic_cases import RUN_SALT
from .support import (
    expand_node,
    filter_node,
    map_node,
    reduce_node,
    source_node,
)


def _item_snapshot(item: Any) -> tuple[int, int, str]:
    return item.port.node, item.port.slot, item.entity.raw.hex()


def _cause_snapshot(cause: Any) -> tuple[str, Any]:
    if hasattr(cause, "port"):
        return "item", _item_snapshot(cause)
    return "grain", cause.raw.hex()


def _snapshot(record: Any) -> dict[str, Any]:
    outcome = record.outcome
    name = type(outcome).__name__
    if name == "Success":
        result = (
            "success",
            tuple(
                tuple(
                    (_item_snapshot(emission.item), emission.ordinal)
                    for emission in emissions
                )
                for emissions in outcome.emissions_by_port
            ),
        )
    elif name == "Failed":
        result = (
            "failed",
            outcome.failure.kind,
            outcome.failure.message,
            tuple(
                _cause_snapshot(cause)
                for cause in outcome.failure.direct_causes
            ),
        )
    else:
        result = (
            "suppressed",
            tuple(
                _cause_snapshot(cause) for cause in outcome.direct_causes
            ),
        )
    return {
        "id": record.id.raw.hex(),
        "node": record.node,
        "inputs": tuple(
            (
                role.role,
                tuple(_item_snapshot(item) for item in role.items),
            )
            for role in record.inputs
        ),
        "slots": tuple(_item_snapshot(item) for item in record.output_slots),
        "outcome": result,
    }


def _oracle_item(item: ItemRef) -> OracleItemRef:
    from .reference_semantics.models import EntityId as OracleEntityId

    return OracleItemRef(
        OraclePortId(item.port.node, item.port.slot),
        OracleEntityId(item.entity.raw),
    )


def test_source_map_filter_expand_and_reduce_reference_parity():
    reference = ReferenceInterpreter(RUN_SALT)
    source_spec = source_node(0)
    production_source = admit_source(source_spec, RUN_SALT, 5)
    oracle_source = reference.source(OraclePortId(0, 0), 5)
    assert _snapshot(production_source) == _snapshot(oracle_source)

    anchor = production_source.output_slots[0]
    oracle_anchor = oracle_source.output_slots[0]
    expand_spec = expand_node(1, anchor.port, outputs=2)
    expand_plan = plan_expand(
        expand_spec,
        RUN_SALT,
        (BindingReceipt.present("parent", anchor),),
    )
    assert expand_plan.grain is not None
    expand_plan.grain.seal(
        expand_success(
            expand_spec,
            expand_plan.grain,
            RUN_SALT,
            cardinality=4,
        )
    )
    oracle_expand = reference.expand(
        1,
        (OracleRoleItems("parent", (oracle_anchor,)),),
        (OraclePortId(1, 0), OraclePortId(1, 1)),
        4,
    )
    assert _snapshot(expand_plan.grain) == _snapshot(oracle_expand)

    map_spec = map_node(2, expand_spec.output_ports[0])
    filter_spec = filter_node(3, map_spec.output_ports[0])
    production_members: list[ItemRef] = []
    oracle_members: list[OracleItemRef] = []
    production_settlements: list[tuple[str, ItemRef | None]] = []
    oracle_settlements: list[MemberSettlement] = []
    keeps = (True, False, True, True)

    for ordinal, emission in enumerate(
        expand_plan.grain.outcome.emissions_by_port[0]  # type: ignore[union-attr]
    ):
        map_plan = plan_map(
            map_spec,
            RUN_SALT,
            (BindingReceipt.present("primary", emission.item),),
        )
        assert map_plan.grain is not None
        map_plan.grain.seal(map_success(map_spec, map_plan.grain))
        oracle_map = reference.map(
            2,
            (
                OracleRoleItems(
                    "primary",
                    (_oracle_item(emission.item),),
                ),
            ),
            (OraclePortId(2, 0),),
        )
        assert _snapshot(map_plan.grain) == _snapshot(oracle_map)

        mapped_item = map_plan.grain.output_slots[0]
        filter_plan = plan_filter(
            filter_spec,
            RUN_SALT,
            (BindingReceipt.present("target", mapped_item),),
        )
        assert filter_plan.grain is not None
        filter_plan.grain.seal(
            filter_success(
                filter_spec,
                filter_plan.grain,
                keep=keeps[ordinal],
            )
        )
        oracle_filter = reference.filter(
            3,
            (
                OracleRoleItems(
                    "target",
                    (_oracle_item(mapped_item),),
                ),
            ),
            (OraclePortId(3, 0),),
            keep=keeps[ordinal],
        )
        assert _snapshot(filter_plan.grain) == _snapshot(oracle_filter)

        final_item = filter_plan.grain.output_slots[0]
        if keeps[ordinal]:
            production_members.append(final_item)
            oracle_final = _oracle_item(final_item)
            oracle_members.append(oracle_final)
            production_settlements.append(("present", final_item))
            oracle_settlements.append(
                MemberSettlement.present(ordinal, oracle_final)
            )
        else:
            production_settlements.append(("dropped", None))
            oracle_settlements.append(MemberSettlement.dropped(ordinal))

    reduce_spec = reduce_node(4, anchor.port, filter_spec.output_ports[0])
    barrier = FiberBarrier(FiberId(4, anchor), expand_plan.grain.id)
    barrier.set_expected(4)
    for ordinal, (kind, item) in enumerate(production_settlements):
        if kind == "present":
            assert item is not None
            barrier.settle_present(ordinal, item)
        else:
            barrier.settle_dropped(ordinal)
    reduce_plan = plan_reduce(
        reduce_spec,
        RUN_SALT,
        BindingReceipt.present("anchor", anchor),
        barrier,
    )
    assert reduce_plan.grain is not None
    from rayorch.experimental.multigrain_v2_5.graph import reduce_success

    reduce_plan.grain.seal(reduce_success(reduce_spec, reduce_plan.grain))
    oracle_reduce = reference.reduce(
        4,
        oracle_anchor,
        OraclePortId(3, 0),
        (OraclePortId(4, 0),),
        expected=4,
        settlements=tuple(oracle_settlements),
    )
    assert _snapshot(reduce_plan.grain) == _snapshot(oracle_reduce)


def test_failure_and_suppressed_reduce_reference_parity():
    reference = ReferenceInterpreter(RUN_SALT)
    source_spec = source_node(0)
    source = admit_source(source_spec, RUN_SALT, 9)
    anchor = source.output_slots[0]
    oracle_anchor = reference.source(OraclePortId(0, 0), 9).output_slots[0]

    expand_spec = expand_node(1, anchor.port)
    expand_plan = plan_expand(
        expand_spec,
        RUN_SALT,
        (BindingReceipt.present("parent", anchor),),
    )
    assert expand_plan.grain is not None
    expand_plan.grain.seal(
        expand_success(
            expand_spec,
            expand_plan.grain,
            RUN_SALT,
            cardinality=2,
        )
    )
    emissions = expand_plan.grain.outcome.emissions_by_port[0]  # type: ignore[union-attr]
    members_port = expand_spec.output_ports[0]
    first_item = emissions[0].item
    failed_item = emissions[1].item
    receipt = GrainId(bytes.fromhex("ab" * 16))
    oracle_receipt = OracleGrainId(receipt.raw)

    barrier = FiberBarrier(FiberId(3, anchor), expand_plan.grain.id)
    barrier.set_expected(2)
    barrier.settle_present(0, first_item)
    barrier.settle_failed(1, failed_item, receipt)
    reduce_spec = reduce_node(3, anchor.port, members_port)
    production_reduce = plan_reduce(
        reduce_spec,
        RUN_SALT,
        BindingReceipt.present("anchor", anchor),
        barrier,
    ).grain
    assert production_reduce is not None

    oracle_reduce = reference.reduce(
        3,
        oracle_anchor,
        OraclePortId(1, 0),
        (OraclePortId(3, 0),),
        expected=2,
        settlements=(
            MemberSettlement.present(0, _oracle_item(first_item)),
            MemberSettlement.failed(
                1,
                _oracle_item(failed_item),
                oracle_receipt,
            ),
        ),
    )
    assert _snapshot(production_reduce) == _snapshot(oracle_reduce)

    production_origin_failure = plan_expand(
        expand_spec,
        RUN_SALT,
        (BindingReceipt.present("parent", anchor),),
    ).grain
    assert production_origin_failure is not None
    production_origin_failure.seal(
        failed_outcome(
            production_origin_failure,
            kind="udf",
            message="fanout",
        )
    )
    blocked = FiberBarrier(
        FiberId(3, anchor),
        production_origin_failure.id,
    )
    blocked.block_origin(production_origin_failure.id)
    blocked_reduce = plan_reduce(
        reduce_spec,
        RUN_SALT,
        BindingReceipt.present("anchor", anchor),
        blocked,
    ).grain
    assert blocked_reduce is not None

    oracle_origin = reference.expand(
        1,
        (OracleRoleItems("parent", (oracle_anchor,)),),
        (OraclePortId(1, 0),),
        None,
        failure="fanout",
    )
    oracle_blocked = reference.reduce(
        3,
        oracle_anchor,
        OraclePortId(1, 0),
        (OraclePortId(3, 0),),
        expected=None,
        origin_failure=oracle_origin.id,
    )
    assert _snapshot(blocked_reduce) == _snapshot(oracle_blocked)
