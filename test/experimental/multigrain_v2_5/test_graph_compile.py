from __future__ import annotations

import pytest

from rayorch.experimental.multigrain_v2_5.api import CompileError
from rayorch.experimental.multigrain_v2_5.graph import (
    InputBinding,
    KeyProjection,
    NodeSpec,
    Primitive,
    compile_graph,
    plan_relate,
)
from rayorch.experimental.multigrain_v2_5.grain import PortId

from .support import (
    EXECUTION,
    RECIPE,
    expand_node,
    filter_node,
    map_node,
    reduce_node,
    source_node,
)


def test_source_expand_unary_reduce_path_compiles():
    source = source_node(0)
    expand = expand_node(1, source.output_ports[0])
    mapped = map_node(2, expand.output_ports[0])
    filtered = filter_node(3, mapped.output_ports[0])
    reduced = reduce_node(
        4,
        source.output_ports[0],
        filtered.output_ports[0],
    )

    graph = compile_graph((source, expand, mapped, filtered, reduced))
    assert graph.producer(source.output_ports[0]).kind is Primitive.SOURCE
    assert graph.producer(filtered.output_ports[0]) == filtered


def test_reduce_anchor_must_be_exact_origin_expand_input_port():
    anchor = source_node(0)
    aligned = source_node(1)
    expand = expand_node(2, anchor.output_ports[0])
    reduced = reduce_node(
        3,
        aligned.output_ports[0],
        expand.output_ports[0],
    )

    with pytest.raises(CompileError, match="exact input port"):
        compile_graph((anchor, aligned, expand, reduced))


def test_reduce_path_rejects_relate_or_second_expand():
    source = source_node(0)
    first = expand_node(1, source.output_ports[0])
    second = expand_node(2, first.output_ports[0])
    reduced = reduce_node(3, source.output_ports[0], second.output_ports[0])

    with pytest.raises(CompileError, match="exact input port"):
        compile_graph((source, first, second, reduced))


def test_source_schema_is_internal_and_strict():
    invalid = NodeSpec(
        id=0,
        kind=Primitive.SOURCE,
        inputs=(InputBinding("x", PortId(9, 0)),),
        output_ports=(PortId(0, 0),),
        udf_recipe=None,
        execution=None,
    )
    with pytest.raises(CompileError, match="Source"):
        compile_graph((invalid,))


def test_relate_schema_and_bounded_sealed_planner_are_retained():
    left = source_node(0)
    right = source_node(1)
    relate = NodeSpec(
        id=2,
        kind=Primitive.RELATE,
        inputs=(
            InputBinding("left", left.output_ports[0]),
            InputBinding("right", right.output_ports[0]),
        ),
        output_ports=(PortId(2, 0),),
        udf_recipe=RECIPE,
        execution=EXECUTION,
        relate_keys=(
            KeyProjection("left", "left_key"),
            KeyProjection("right", "right_key"),
        ),
    )

    graph = compile_graph((left, right, relate))
    assert graph.node(2).relate_keys == relate.relate_keys
    open_result = plan_relate(
        relate,
        bytes.fromhex("00" * 16),
        (("left", ()), ("right", ())),
        sealed_roles=frozenset({"left"}),
        max_cardinality=10,
    )
    assert not open_result.complete


def test_graph_must_be_topologically_ordered():
    source = source_node(0)
    mapped = map_node(1, source.output_ports[0])
    with pytest.raises(CompileError, match="topologically"):
        compile_graph((mapped, source))
