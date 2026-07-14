"""Mandatory structural validation for multigrain execution graphs."""
from __future__ import annotations

from collections.abc import Sequence

from .graph import ExecutionGraph, NodeSpec
from .operations import (
    ExpandOp,
    FilterByMaskOp,
    FilterOp,
    KeyJoinSpec,
    MapOp,
    ReduceOp,
    RelateOp,
    RelationAdapterSpec,
    operator_factory,
)
from .refs import NodeOutputRef, PortRef, ref_label
from .relations import (
    AggregateOf,
    ChildrenOf,
    RelatedFrom,
    SameAs,
    SubsetOf,
)


class GraphValidationError(ValueError):
    """The passive graph violates a topology or relation invariant."""


def verify_graph(graph: ExecutionGraph) -> ExecutionGraph:
    if not graph.name:
        raise GraphValidationError("graph name must not be empty")
    if not graph.outputs:
        raise GraphValidationError("graph must expose at least one output")
    input_names = [spec.name for spec in graph.inputs]
    if len(set(input_names)) != len(input_names):
        raise GraphValidationError("graph input names must be unique")
    if any(not spec.name or not spec.grain for spec in graph.inputs):
        raise GraphValidationError("graph input names and grains must not be empty")

    grains: dict[PortRef, str] = {spec.ref: spec.grain for spec in graph.inputs}
    node_names: set[str] = set()
    for node in graph.nodes:
        if node.name in node_names:
            raise GraphValidationError(f"duplicate node name '{node.name}'")
        node_names.add(node.name)
        missing = [ref_label(ref) for ref in node.inputs if ref not in grains]
        if missing:
            raise GraphValidationError(
                f"node '{node.name}' has unresolved or non-topological inputs {missing}"
            )
        _verify_node(node, grains)
        for output in node.outputs:
            grains[output.ref] = output.grain

    missing_outputs = [ref_label(ref) for ref in graph.outputs if ref not in grains]
    if missing_outputs:
        raise GraphValidationError(
            f"graph outputs are not produced: {missing_outputs}"
        )
    return graph


def _verify_node(node: NodeSpec, upstream_grains: dict[PortRef, str]) -> None:
    if not node.name:
        raise GraphValidationError("node name must not be empty")
    if not node.inputs:
        raise GraphValidationError(f"node '{node.name}' has no inputs")
    if not node.outputs:
        raise GraphValidationError(f"node '{node.name}' has no outputs")
    output_names = [output.name for output in node.outputs]
    if len(set(output_names)) != len(output_names):
        raise GraphValidationError(f"node '{node.name}' has duplicate outputs")
    if any(output.ref.node != node.name for output in node.outputs):
        raise GraphValidationError(
            f"node '{node.name}' owns an output with a different producer"
        )
    if any(not output.name or not output.grain for output in node.outputs):
        raise GraphValidationError(
            f"node '{node.name}' output names and grains must not be empty"
        )
    factory = operator_factory(node.operation)
    if factory is not None and not factory.import_path:
        raise GraphValidationError(
            f"node '{node.name}' operator factory import_path must not be empty"
        )
    input_grains = tuple(upstream_grains[ref] for ref in node.inputs)
    if isinstance(node.operation, (MapOp, FilterOp, FilterByMaskOp)):
        if len(set(input_grains)) != 1:
            raise GraphValidationError(
                f"node '{node.name}' operation {type(node.operation).__name__} "
                f"requires same-grain inputs, got {input_grains}"
            )

    available_grains = dict(upstream_grains)
    for output in node.outputs:
        relation = output.relation
        _verify_operation_relation(node, relation)
        if isinstance(relation, (SameAs, SubsetOf)):
            source_grain = _require_source(node, relation.source, available_grains)
            if output.grain != source_grain:
                raise GraphValidationError(
                    f"{type(relation).__name__} output "
                    f"'{node.name}.{output.name}' must keep source grain "
                    f"'{source_grain}', got '{output.grain}'"
                )
        elif isinstance(relation, ChildrenOf):
            _require_source(node, relation.parent, available_grains)
            if not relation.label:
                raise GraphValidationError(
                    f"ChildrenOf output '{node.name}.{output.name}' needs label"
                )
            if output.grain != relation.label:
                raise GraphValidationError(
                    f"ChildrenOf output '{node.name}.{output.name}' grain "
                    f"'{output.grain}' must equal child label '{relation.label}'"
                )
        elif isinstance(relation, AggregateOf):
            anchor_grain = _require_node_input(
                node, relation.anchor, available_grains, "aggregate anchor"
            )
            for member in relation.members:
                _require_node_input(
                    node, member, available_grains, "aggregate member"
                )
            if output.grain != anchor_grain:
                raise GraphValidationError(
                    f"AggregateOf output '{node.name}.{output.name}' must use "
                    f"anchor grain '{anchor_grain}'"
                )
            expected_members = tuple(node.inputs[1:])
            if relation.anchor != node.inputs[0] or relation.members != expected_members:
                raise GraphValidationError(
                    f"Reduce node '{node.name}' must declare input 0 as anchor "
                    "and remaining inputs as members"
                )
        elif isinstance(relation, RelatedFrom):
            roles = [binding.role for binding in relation.roles]
            if not roles or len(set(roles)) != len(roles):
                raise GraphValidationError(
                    f"RelatedFrom output '{node.name}.{output.name}' needs "
                    "unique non-empty roles"
                )
            sources = tuple(binding.source for binding in relation.roles)
            if sources != node.inputs:
                raise GraphValidationError(
                    f"RelatedFrom output '{node.name}.{output.name}' roles "
                    "must cover node inputs in order"
                )
            for source in sources:
                _require_node_input(
                    node, source, available_grains, "relation role source"
                )
        available_grains[output.ref] = output.grain

    operation = node.operation
    if isinstance(operation, MapOp):
        if any(
            not isinstance(output.relation, SameAs)
            or output.relation.source != node.inputs[0]
            for output in node.outputs
        ):
            raise GraphValidationError(
                f"Map node '{node.name}' outputs must be SameAs input 0"
            )
    if isinstance(operation, FilterOp):
        expected = tuple(node.inputs)
        actual = tuple(
            output.relation.source
            for output in node.outputs
            if isinstance(output.relation, SubsetOf)
        )
        if len(node.outputs) != len(expected) or actual != expected:
            raise GraphValidationError(
                f"Filter node '{node.name}' must emit one SubsetOf per input"
            )
    if isinstance(operation, FilterByMaskOp):
        if operation.mask_input < 0 or operation.mask_input >= len(node.inputs):
            raise GraphValidationError(
                f"FilterByMask node '{node.name}' mask_input is out of range"
            )
        expected = tuple(
            ref
            for index, ref in enumerate(node.inputs)
            if index != operation.mask_input
        )
        actual = tuple(
            output.relation.source
            for output in node.outputs
            if isinstance(output.relation, SubsetOf)
        )
        if len(node.outputs) != len(expected) or actual != expected:
            raise GraphValidationError(
                f"FilterByMask node '{node.name}' must emit every non-mask input"
            )
    if isinstance(operation, RelateOp):
        if len(node.outputs) != 1:
            raise GraphValidationError(
                f"Relate node '{node.name}' currently requires one output"
            )
        relation = node.outputs[0].relation
        if isinstance(relation, RelatedFrom):
            relation_roles = tuple(binding.role for binding in relation.roles)
            matcher_fields = getattr(operation.matcher, "fields", ())
            if matcher_fields and tuple(role for role, _ in matcher_fields) != relation_roles:
                raise GraphValidationError(
                    f"Relate node '{node.name}' matcher roles differ from output roles"
                )
        if isinstance(operation.matcher, KeyJoinSpec):
            if not operation.matcher.fields or any(
                not role or not field for role, field in operation.matcher.fields
            ):
                raise GraphValidationError(
                    f"Relate node '{node.name}' key join fields must not be empty"
                )
        elif isinstance(operation.matcher, RelationAdapterSpec):
            if ":" not in operation.matcher.import_path:
                raise GraphValidationError(
                    f"Relate node '{node.name}' adapter must be a 'pkg.mod:fn' path"
                )


def _verify_operation_relation(node: NodeSpec, relation: object) -> None:
    allowed: tuple[type[object], ...]
    if isinstance(node.operation, MapOp):
        allowed = (SameAs,)
    elif isinstance(node.operation, (FilterOp, FilterByMaskOp)):
        allowed = (SubsetOf,)
    elif isinstance(node.operation, ExpandOp):
        allowed = (SameAs, ChildrenOf)
    elif isinstance(node.operation, ReduceOp):
        allowed = (AggregateOf,)
    elif isinstance(node.operation, RelateOp):
        allowed = (RelatedFrom,)
    else:
        raise GraphValidationError(
            f"node '{node.name}' uses unsupported operation {type(node.operation).__name__}"
        )
    if not isinstance(relation, allowed):
        expected = ", ".join(item.__name__ for item in allowed)
        raise GraphValidationError(
            f"node '{node.name}' operation {type(node.operation).__name__} "
            f"cannot produce {type(relation).__name__}; expected {expected}"
        )


def _require_source(
    node: NodeSpec,
    source: PortRef,
    grains: dict[PortRef, str],
) -> str:
    if source not in grains:
        if isinstance(source, NodeOutputRef) and source.node == node.name:
            raise GraphValidationError(
                f"node '{node.name}' relation has a self or forward source "
                f"'{ref_label(source)}'"
            )
        raise GraphValidationError(
            f"node '{node.name}' relation source '{ref_label(source)}' is unavailable"
        )
    if source not in node.inputs and not (
        isinstance(source, NodeOutputRef) and source.node == node.name
    ):
        raise GraphValidationError(
            f"node '{node.name}' relation source '{ref_label(source)}' must be "
            "an invocation input or earlier output"
        )
    return grains[source]


def _require_node_input(
    node: NodeSpec,
    source: PortRef,
    grains: dict[PortRef, str],
    purpose: str,
) -> str:
    if source not in node.inputs:
        raise GraphValidationError(
            f"node '{node.name}' {purpose} '{ref_label(source)}' is not a node input"
        )
    return grains[source]


def validate_shard_plan(
    partitions: Sequence[Sequence[int]],
    row_count: int,
) -> tuple[tuple[int, ...], ...]:
    """Validate exact-once partition coverage assumed by the reorder theorem."""

    if row_count < 0:
        raise ValueError("row_count must be >= 0")
    normalized = tuple(tuple(int(index) for index in part) for part in partitions)
    flat = [index for part in normalized for index in part]
    if any(index < 0 or index >= row_count for index in flat):
        raise ValueError("shard planner produced an out-of-range row index")
    if len(flat) != len(set(flat)):
        raise ValueError("shard planner produced duplicate row indices")
    if set(flat) != set(range(row_count)):
        raise ValueError("shard planner must cover every row exactly once")
    return normalized


__all__ = [
    "GraphValidationError",
    "validate_shard_plan",
    "verify_graph",
]
