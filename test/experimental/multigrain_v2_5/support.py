from __future__ import annotations

from rayorch.experimental.multigrain_v2_5.graph import (
    ExecutionOptions,
    InputBinding,
    NodeSpec,
    Primitive,
    UdfRecipe,
)
from rayorch.experimental.multigrain_v2_5.grain import PortId


RECIPE = UdfRecipe("tests.noop")
EXECUTION = ExecutionOptions()


def source_node(node_id: int) -> NodeSpec:
    return NodeSpec(
        id=node_id,
        kind=Primitive.SOURCE,
        inputs=(),
        output_ports=(PortId(node_id, 0),),
        udf_recipe=None,
        execution=None,
    )


def map_node(
    node_id: int,
    primary: PortId,
    *secondary: tuple[str, PortId],
    outputs: int = 1,
) -> NodeSpec:
    return NodeSpec(
        id=node_id,
        kind=Primitive.MAP,
        inputs=(InputBinding("primary", primary),)
        + tuple(InputBinding(role, port) for role, port in secondary),
        output_ports=tuple(PortId(node_id, slot) for slot in range(outputs)),
        udf_recipe=RECIPE,
        execution=EXECUTION,
    )


def filter_node(
    node_id: int,
    target: PortId,
    *controls: tuple[str, PortId],
    outputs: int = 1,
) -> NodeSpec:
    return NodeSpec(
        id=node_id,
        kind=Primitive.FILTER,
        inputs=(InputBinding("target", target),)
        + tuple(InputBinding(role, port) for role, port in controls),
        output_ports=tuple(PortId(node_id, slot) for slot in range(outputs)),
        udf_recipe=RECIPE,
        execution=EXECUTION,
    )


def expand_node(
    node_id: int,
    parent: PortId,
    *,
    outputs: int = 1,
) -> NodeSpec:
    return NodeSpec(
        id=node_id,
        kind=Primitive.EXPAND,
        inputs=(InputBinding("parent", parent),),
        output_ports=tuple(PortId(node_id, slot) for slot in range(outputs)),
        udf_recipe=RECIPE,
        execution=EXECUTION,
    )


def reduce_node(
    node_id: int,
    anchor: PortId,
    members: PortId,
    *,
    outputs: int = 1,
) -> NodeSpec:
    return NodeSpec(
        id=node_id,
        kind=Primitive.REDUCE,
        inputs=(
            InputBinding("anchor", anchor),
            InputBinding("members", members),
        ),
        output_ports=tuple(PortId(node_id, slot) for slot in range(outputs)),
        udf_recipe=RECIPE,
        execution=EXECUTION,
        reduce_anchor=anchor,
        reduce_members=members,
    )
