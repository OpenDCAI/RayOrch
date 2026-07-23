from __future__ import annotations

from dataclasses import replace

import pytest
from ray import cloudpickle

from rayorch.experimental.multigrain_v2.graph import (
    AggregateOf,
    ChildrenOf,
    CompiledGraph,
    CompiledInputGroup,
    CompiledNode,
    CompiledPort,
    ExpandSpec,
    FilterSpec,
    MapSpec,
    ModuleConfig,
    OperatorFactory,
    ReduceMemberSpec,
    ReduceSpec,
    RelateRoleSpec,
    RelateSpec,
    RelatedFrom,
    SameAs,
    SourceRelation,
    SubsetOf,
    ordered_input_ports,
    relation_of,
)
from rayorch.experimental.multigrain_v2.identity import (
    DomainId,
    NodeId,
    PortId,
    stable_hash,
)
from test.experimental.multigrain_v2.reference_semantics.graph_oracle import (
    project_relation,
)


def node_id(name: str) -> NodeId:
    return NodeId(stable_hash("test-node", name))


def port_id(name: str) -> PortId:
    return PortId(stable_hash("test-port", name))


def domain_id(name: str) -> DomainId:
    return DomainId(stable_hash("test-domain", name))


def graph_for(
    spec,
    *,
    input_ports: tuple[CompiledPort, ...],
    output_domains: tuple[DomainId, ...],
) -> tuple[CompiledGraph, tuple[PortId, ...]]:
    nid = node_id(type(spec).__name__)
    outputs = tuple(
        port_id(f"{type(spec).__name__}:{slot}")
        for slot in range(len(output_domains))
    )
    output_ports = tuple(
        CompiledPort(port, nid, slot, domain)
        for slot, (port, domain) in enumerate(zip(outputs, output_domains))
    )
    node = CompiledNode(
        nid,
        spec,
        None,
        ModuleConfig(),
        False,
        outputs,
    )
    inputs = tuple(
        CompiledInputGroup(
            f"in{index}",
            index,
            compiled.domain,
            compiled.id,
        )
        for index, compiled in enumerate(input_ports)
    )
    return (
        CompiledGraph(
            (node,),
            input_ports + output_ports,
            inputs,
            outputs,
            (nid,),
        ),
        outputs,
    )


def source_port(name: str, domain: str = "rows") -> CompiledPort:
    return CompiledPort(port_id(name), None, 0, domain_id(domain))


def plain_relation(relation) -> dict:
    if isinstance(relation, SameAs):
        return {
            "kind": "same_as",
            "roles": relation.role_names,
            "parents": relation.parents,
        }
    if isinstance(relation, SubsetOf):
        return {
            "kind": "subset_of",
            "target": relation.target,
            "controls": relation.controls,
        }
    if isinstance(relation, ChildrenOf):
        return {
            "kind": "children_of",
            "parent": relation.parent,
            "context": relation.context_ports,
        }
    if isinstance(relation, AggregateOf):
        return {
            "kind": "aggregate_of",
            "anchor": relation.anchor,
            "roles": relation.role_names,
            "members": relation.member_ports,
        }
    if isinstance(relation, RelatedFrom):
        return {
            "kind": "related_from",
            "roles": relation.role_names,
            "parents": relation.role_ports,
        }
    raise TypeError(type(relation).__name__)


def test_source_and_map_relations() -> None:
    primary = source_port("primary")
    aligned = source_port("aligned")
    spec = MapSpec(
        primary.id,
        ("primary", "aligned"),
        (primary.id, aligned.id),
    )
    graph, outputs = graph_for(
        spec,
        input_ports=(primary, aligned),
        output_domains=(primary.domain, primary.domain),
    )
    assert relation_of(graph, primary.id) == SourceRelation()
    expected = SameAs(("primary", "aligned"), (primary.id, aligned.id))
    assert tuple(relation_of(graph, output) for output in outputs) == (
        expected,
        expected,
    )
    assert ordered_input_ports(spec) == (primary.id, aligned.id)


def test_expand_reduce_and_relate_relations() -> None:
    parent = source_port("parent", "document")
    context = source_port("context", "document")
    child_domain = domain_id("child")
    expand = ExpandSpec(
        parent.id,
        ("parent", "context"),
        (parent.id, context.id),
    )
    graph, (output,) = graph_for(
        expand,
        input_ports=(parent, context),
        output_domains=(child_domain,),
    )
    assert relation_of(graph, output) == ChildrenOf(parent.id, (context.id,))

    member = source_port("member", "child")
    reduce = ReduceSpec(
        parent.id,
        (ReduceMemberSpec("pages", member.id),),
    )
    graph, (output,) = graph_for(
        reduce,
        input_ports=(parent, member),
        output_domains=(parent.domain,),
    )
    assert relation_of(graph, output) == AggregateOf(
        parent.id, ("pages",), (member.id,)
    )

    left = source_port("left", "left")
    right = source_port("right", "right")
    left_key = source_port("left_key", "left")
    right_key = source_port("right_key", "right")
    relate = RelateSpec(
        "key",
        (
            RelateRoleSpec("left", left.id, left_key.id),
            RelateRoleSpec("right", right.id, right_key.id),
        ),
    )
    graph, (output,) = graph_for(
        relate,
        input_ports=(left, right, left_key, right_key),
        output_domains=(domain_id("related"),),
    )
    assert relation_of(graph, output) == RelatedFrom(
        ("left", "right"), (left.id, right.id)
    )
    assert ordered_input_ports(relate) == (left.id, right.id)


def test_filter_relations_cover_predicate_mask_and_select_annotations() -> None:
    left = source_port("left")
    right = source_port("right")
    mask = source_port("mask")

    predicate = FilterSpec("predicate", ("value",), (left.id,), (left.id,))
    graph, (output,) = graph_for(
        predicate,
        input_ports=(left,),
        output_domains=(left.domain,),
    )
    assert relation_of(graph, output) == SubsetOf(left.id, ())

    by_mask = FilterSpec("mask", (), (mask.id,), (left.id,))
    graph, (output,) = graph_for(
        by_mask,
        input_ports=(left, mask),
        output_domains=(left.domain,),
    )
    assert relation_of(graph, output) == SubsetOf(left.id, (mask.id,))

    select = FilterSpec(
        "select",
        ("left", "right"),
        (left.id, right.id),
        (left.id, right.id),
        annotation_arity=1,
    )
    graph, outputs = graph_for(
        select,
        input_ports=(left, right),
        output_domains=(left.domain, right.domain, left.domain),
    )
    assert relation_of(graph, outputs[0]) == SubsetOf(left.id, (right.id,))
    assert relation_of(graph, outputs[1]) == SubsetOf(right.id, (left.id,))
    assert relation_of(graph, outputs[2]) == SubsetOf(left.id, (right.id,))


def test_all_operation_relations_match_independent_plain_data_oracle() -> None:
    left = source_port("oracle-left", "left")
    right = source_port("oracle-right", "right")
    specs = (
        (
            MapSpec(left.id, ("left", "right"), (left.id, right.id)),
            (left, right),
            (left.domain,),
            {
                "kind": "map",
                "roles": ("left", "right"),
                "inputs": (left.id, right.id),
            },
        ),
        (
            ExpandSpec(left.id, ("left", "right"), (left.id, right.id)),
            (left, right),
            (domain_id("oracle-child"),),
            {
                "kind": "expand",
                "parent": left.id,
                "inputs": (left.id, right.id),
            },
        ),
        (
            ReduceSpec(
                left.id,
                (ReduceMemberSpec("right", right.id),),
            ),
            (left, right),
            (left.domain,),
            {
                "kind": "reduce",
                "anchor": left.id,
                "members": (("right", right.id),),
            },
        ),
        (
            RelateSpec(
                "custom",
                (
                    RelateRoleSpec("left", left.id, None),
                    RelateRoleSpec("right", right.id, None),
                ),
            ),
            (left, right),
            (domain_id("oracle-related"),),
            {
                "kind": "relate",
                "roles": (
                    {"name": "left", "value": left.id},
                    {"name": "right", "value": right.id},
                ),
            },
        ),
        (
            FilterSpec(
                "select",
                ("left", "right"),
                (left.id, right.id),
                (left.id, right.id),
                annotation_arity=1,
            ),
            (left, right),
            (left.domain, right.domain, left.domain),
            {
                "kind": "filter",
                "controls": (left.id, right.id),
                "targets": (left.id, right.id),
                "annotation_arity": 1,
            },
        ),
    )
    for spec, inputs, domains, oracle_spec in specs:
        graph, outputs = graph_for(
            spec,
            input_ports=inputs,
            output_domains=domains,
        )
        for slot, output in enumerate(outputs):
            assert plain_relation(relation_of(graph, output)) == project_relation(
                oracle_spec, slot
            )


@pytest.mark.parametrize(
    "spec",
    [
        FilterSpec("predicate", ("x",), (port_id("c"),), (port_id("t"),)),
        FilterSpec("mask", (), (port_id("m"),), (port_id("t"),)),
    ],
)
def test_filter_ordered_inputs_are_deduplicated(spec: FilterSpec) -> None:
    assert ordered_input_ports(spec) == tuple(
        dict.fromkeys(spec.control_inputs + spec.targets)
    )


def test_ordered_inputs_preserve_duplicate_semantic_roles() -> None:
    shared = port_id("shared-role")
    assert ordered_input_ports(
        MapSpec(shared, ("left", "right"), (shared, shared))
    ) == (shared, shared)
    assert ordered_input_ports(
        FilterSpec(
            "select",
            ("left", "right"),
            (shared, shared),
            (shared, shared),
        )
    ) == (shared, shared)
    assert ordered_input_ports(
        ReduceSpec(
            shared,
            (
                ReduceMemberSpec("left", shared),
                ReduceMemberSpec("right", shared),
            ),
        )
    ) == (shared, shared, shared)
    assert ordered_input_ports(
        RelateSpec(
            "custom",
            (
                RelateRoleSpec("left", shared, None),
                RelateRoleSpec("right", shared, None),
            ),
        )
    ) == (shared, shared)


def test_graph_and_factory_are_cloudpickle_snapshots() -> None:
    class Udf:
        def __init__(self, value: int, *, unused: list[object]) -> None:
            self.value = value
            self.unused = unused

    mutable_args = [7]
    mutable_kwargs = {"unused": []}
    recipe = cloudpickle.loads(
        cloudpickle.dumps((Udf, tuple(mutable_args), dict(mutable_kwargs)))
    )
    factory = OperatorFactory(cloudpickle.dumps(recipe))
    mutable_args[0] = 99
    mutable_kwargs["unused"].append("changed")
    built = factory.build()
    assert built.value == 7
    assert built.unused == []

    source = source_port("snapshot")
    spec = MapSpec(source.id, ("value",), (source.id,))
    graph, _ = graph_for(
        spec,
        input_ports=(source,),
        output_domains=(source.domain,),
    )
    runtime_env = {"env_vars": {"PROFILE": "test"}}
    detached_env = cloudpickle.loads(cloudpickle.dumps(runtime_env))
    graph = replace(
        graph,
        nodes=(
            replace(
                graph.nodes[0],
                factory=factory,
                config=ModuleConfig(runtime_env=detached_env),
            ),
        ),
    )
    runtime_env["env_vars"]["PROFILE"] = "changed"
    restored = cloudpickle.loads(cloudpickle.dumps(graph))
    assert restored == graph
    assert restored.nodes[0].factory is not None
    assert restored.nodes[0].factory.build().value == 7
    assert restored.nodes[0].config.runtime_env == {
        "env_vars": {"PROFILE": "test"}
    }
    assert all(not hasattr(port, "relation") for port in restored.ports)


def test_compiled_graph_rejects_unknown_outputs() -> None:
    with pytest.raises(ValueError, match="unknown port"):
        CompiledGraph((), (), (), (port_id("missing"),), ())


def test_graph_rejects_output_slot_drift_and_filter_arity_drift() -> None:
    source = source_port("slot-source")
    spec = MapSpec(source.id, ("value",), (source.id,))
    graph, _ = graph_for(
        spec,
        input_ports=(source,),
        output_domains=(source.domain,),
    )
    bad_port = replace(graph.ports[-1], slot=1)
    with pytest.raises(ValueError, match="slots"):
        replace(graph, ports=graph.ports[:-1] + (bad_port,))

    select = FilterSpec(
        "select",
        ("left", "right"),
        (source.id, source.id),
        (source.id, source.id),
        annotation_arity=1,
    )
    with pytest.raises(ValueError, match="output count"):
        graph_for(
            select,
            input_ports=(source,),
            output_domains=(source.domain,),
        )


def test_graph_rejects_dangling_spec_and_producer_references() -> None:
    source = source_port("closed-source")
    spec = MapSpec(source.id, ("value",), (source.id,))
    graph, _ = graph_for(
        spec,
        input_ports=(source,),
        output_domains=(source.domain,),
    )

    unknown = port_id("unknown-dependency")
    bad_node = replace(
        graph.nodes[0],
        spec=MapSpec(unknown, ("value",), (unknown,)),
    )
    with pytest.raises(ValueError, match="unknown port"):
        replace(graph, nodes=(bad_node,))

    dangling = CompiledPort(
        port_id("dangling-produced"),
        graph.nodes[0].id,
        1,
        source.domain,
    )
    with pytest.raises(ValueError, match="producer"):
        replace(graph, ports=graph.ports + (dangling,))


def test_graph_rejects_two_input_groups_for_one_source_port() -> None:
    source = source_port("shared-source")
    first = CompiledInputGroup("first", 0, source.domain, source.id)
    second = CompiledInputGroup("second", 1, source.domain, source.id)
    with pytest.raises(ValueError, match="source ports must be unique"):
        CompiledGraph((), (source,), (first, second), (), ())
