from __future__ import annotations

import pytest

from rayorch.experimental.multigrain_v2_5.graph import (
    BindingReceipt,
    PlanAction,
    PlannerContractError,
    admit_source,
    ensure_decision,
    expand_success,
    filter_success,
    map_success,
    plan_expand,
    plan_filter,
    plan_map,
)
from rayorch.experimental.multigrain_v2_5.grain import (
    EntityId,
    Failed,
    GrainFailure,
    GrainId,
    GrainPhase,
    GrainTable,
    ItemRef,
    PortId,
    Success,
    Suppressed,
)

from .reference_semantics.interpreter import ReferenceInterpreter
from .reference_semantics.models import PortId as OraclePortId
from .reference_semantics.semantic_cases import RUN_SALT
from .support import expand_node, filter_node, map_node, source_node


def _source():
    node = source_node(0)
    record = admit_source(node, RUN_SALT, 0)
    return node, record, record.output_slots[0]


def test_source_admission_matches_reference_semantics():
    node = source_node(0)
    production = admit_source(node, RUN_SALT, 7)
    reference = ReferenceInterpreter(RUN_SALT).source(
        OraclePortId(0, 0),
        7,
    )

    assert production.id.raw == reference.id.raw
    assert production.output_slots[0].entity.raw == (
        reference.output_slots[0].entity.raw
    )
    assert production.node == reference.node


def test_planner_normal_absence_short_circuits_other_roles():
    _, _, item = _source()
    control_item = ItemRef(PortId(1, 0), item.entity)
    node = map_node(2, item.port, ("control", control_item.port))
    decision = plan_map(
        node,
        RUN_SALT,
        (
            BindingReceipt.absent("primary", item),
            BindingReceipt.failed(
                "control",
                control_item,
                GrainId(bytes.fromhex("11" * 16)),
            ),
        ),
    )
    assert decision.action is PlanAction.NORMAL_ABSENCE
    assert decision.grain is None


def test_planner_waits_for_all_required_roles_before_suppression():
    _, _, item = _source()
    control_item = ItemRef(PortId(1, 0), item.entity)
    node = map_node(2, item.port, ("control", control_item.port))
    decision = plan_map(
        node,
        RUN_SALT,
        (
            BindingReceipt.failed(
                "primary",
                item,
                GrainId(bytes.fromhex("21" * 16)),
            ),
            BindingReceipt.pending("control", control_item),
        ),
    )
    assert decision.action is PlanAction.WAIT


def test_aligned_normal_missing_is_contract_violation():
    _, _, item = _source()
    control_item = ItemRef(PortId(1, 0), item.entity)
    node = map_node(2, item.port, ("control", control_item.port))
    with pytest.raises(PlannerContractError, match="normally absent"):
        plan_map(
            node,
            RUN_SALT,
            (
                BindingReceipt.present("primary", item),
                BindingReceipt.absent("control", control_item),
            ),
        )


def test_suppression_causes_follow_compiled_role_order():
    _, _, item = _source()
    control_item = ItemRef(PortId(1, 0), item.entity)
    node = map_node(2, item.port, ("control", control_item.port))
    primary_cause = GrainId(bytes.fromhex("31" * 16))
    control_cause = GrainId(bytes.fromhex("30" * 16))

    decision = plan_map(
        node,
        RUN_SALT,
        (
            BindingReceipt.failed("primary", item, primary_cause),
            BindingReceipt.suppressed(
                "control",
                control_item,
                control_cause,
            ),
        ),
    )

    assert decision.action is PlanAction.ENSURE_SUPPRESSED
    assert decision.grain is not None
    assert decision.grain.outcome == Suppressed(
        (primary_cause, control_cause)
    )


def test_all_present_roles_create_one_executable_grain():
    _, _, item = _source()
    control_item = ItemRef(PortId(1, 0), item.entity)
    node = map_node(2, item.port, ("control", control_item.port))
    decision = plan_map(
        node,
        RUN_SALT,
        (
            BindingReceipt.present("primary", item),
            BindingReceipt.present("control", control_item),
        ),
    )
    assert decision.action is PlanAction.ENSURE_EXECUTABLE
    assert decision.grain is not None
    assert decision.grain.phase is GrainPhase.READY
    assert decision.grain.output_slots[0].entity == item.entity


def test_misaligned_secondary_entity_aborts_planning():
    _, _, item = _source()
    control_item = ItemRef(
        PortId(1, 0),
        EntityId(bytes.fromhex("99" * 16)),
    )
    node = map_node(2, item.port, ("control", control_item.port))
    with pytest.raises(PlannerContractError, match="not aligned"):
        plan_map(
            node,
            RUN_SALT,
            (
                BindingReceipt.present("primary", item),
                BindingReceipt.present("control", control_item),
            ),
        )


@pytest.mark.parametrize("terminal", [None, "success", "failed"])
def test_ensure_executable_is_idempotent_for_every_lifecycle(terminal):
    _, _, item = _source()
    node = map_node(2, item.port)
    decision = plan_map(
        node,
        RUN_SALT,
        (BindingReceipt.present("primary", item),),
    )
    table = GrainTable()
    existing = ensure_decision(table, decision)
    assert existing is not None

    if terminal is None:
        existing.reserve(arena=1, dispatch=1)
    elif terminal == "success":
        existing.seal(map_success(node, existing))
    else:
        existing.seal(Failed(GrainFailure("udf", "bad", (item,))))

    repeated = ensure_decision(table, decision)
    assert repeated is existing
    assert len(table) == 1


def test_filter_and_expand_success_cardinality_contracts():
    _, _, item = _source()
    filter_spec = filter_node(2, item.port, outputs=2)
    filter_plan = plan_filter(
        filter_spec,
        RUN_SALT,
        (BindingReceipt.present("target", item),),
    )
    assert filter_plan.grain is not None
    assert filter_success(
        filter_spec,
        filter_plan.grain,
        keep=False,
    ) == Success(((), ()))

    expand_spec = expand_node(3, item.port, outputs=2)
    expand_plan = plan_expand(
        expand_spec,
        RUN_SALT,
        (BindingReceipt.present("parent", item),),
    )
    assert expand_plan.grain is not None
    outcome = expand_success(
        expand_spec,
        expand_plan.grain,
        RUN_SALT,
        cardinality=3,
    )
    assert [len(port) for port in outcome.emissions_by_port] == [3, 3]
    assert [
        emission.item.entity for emission in outcome.emissions_by_port[0]
    ] == [
        emission.item.entity for emission in outcome.emissions_by_port[1]
    ]
