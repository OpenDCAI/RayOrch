"""Runtime handlers for typed operations in the passive execution graph."""
from __future__ import annotations

from typing import Any, Mapping, Protocol

from ..data.batch import NodeExecution, PortBatch, group_by
from ..ir.graph import NodeSpec
from ..ir.operations import (
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
from ..ir.relations import AggregateOf, ChildrenOf, RelatedFrom
from ..primitives._binding import load_factory_object
from ..primitives.expand_reduce import Expand, Reduce
from ..primitives.map_filter import Filter, Map
from ..primitives.output import select_filter_outputs
from ..primitives.relate import Relate


class OperationHandler(Protocol):
    def prepare(self, node: NodeSpec) -> Any: ...

    def execute(
        self,
        node: NodeSpec,
        runtime: Any,
        inputs: tuple[PortBatch, ...],
        *,
        force_inline: bool,
    ) -> NodeExecution: ...


def _factory(node: NodeSpec) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
    factory = operator_factory(node.operation)
    if factory is None:
        raise TypeError(f"node '{node.name}' operation has no operator factory")
    return (
        load_factory_object(factory.import_path),
        tuple(factory.args),
        dict(factory.kwargs),
    )


def _outputs(value: Any) -> tuple[PortBatch, ...]:
    if isinstance(value, PortBatch):
        return (value,)
    if isinstance(value, tuple) and all(
        isinstance(item, PortBatch) for item in value
    ):
        return value
    raise TypeError("multigrain handler expected PortBatch outputs")


class MapHandler:
    def prepare(self, node: NodeSpec) -> Map:
        op_cls, args, kwargs = _factory(node)
        return Map(
            op_cls,
            *args,
            name=node.name,
            num_outputs=len(node.outputs),
            workers=node.workers,
            recovery=node.recovery,
            **kwargs,
        )

    def execute(
        self,
        node: NodeSpec,
        runtime: Map,
        inputs: tuple[PortBatch, ...],
        *,
        force_inline: bool,
    ) -> NodeExecution:
        result, deferred = runtime.run_with_recovery(
            *inputs,
            force_inline=force_inline,
        )
        return NodeExecution(_outputs(result), tuple(deferred))


class ExpandHandler:
    def prepare(self, node: NodeSpec) -> Expand:
        op_cls, args, kwargs = _factory(node)
        relations = tuple(output.relation for output in node.outputs)
        if not relations or not all(
            isinstance(relation, ChildrenOf) for relation in relations
        ):
            raise NotImplementedError(
                "mixed Expand output relations are reserved by the IR but "
                "runtime mg.out.same/children materialization is not implemented"
            )
        first = relations[0]
        assert isinstance(first, ChildrenOf)
        if any(relation != first for relation in relations[1:]):
            raise NotImplementedError(
                "current Expand runtime requires one shared parent relation"
            )
        return Expand(
            op_cls,
            *args,
            parent=node.inputs.index(first.parent),
            child_label=first.label,
            name=node.name,
            num_outputs=len(node.outputs),
            workers=node.workers,
            recovery=node.recovery,
            **kwargs,
        )

    def execute(
        self,
        node: NodeSpec,
        runtime: Expand,
        inputs: tuple[PortBatch, ...],
        *,
        force_inline: bool,
    ) -> NodeExecution:
        return NodeExecution(_outputs(runtime(*inputs)))


class FilterHandler:
    def prepare(self, node: NodeSpec) -> Filter:
        op_cls, args, kwargs = _factory(node)
        return Filter(
            op_cls,
            *args,
            name=node.name,
            workers=node.workers,
            recovery=node.recovery,
            **kwargs,
        )

    def execute(
        self,
        node: NodeSpec,
        runtime: Filter,
        inputs: tuple[PortBatch, ...],
        *,
        force_inline: bool,
    ) -> NodeExecution:
        return NodeExecution(_outputs(runtime(*inputs)))


class ReduceHandler:
    def prepare(self, node: NodeSpec) -> Reduce:
        op_cls, args, kwargs = _factory(node)
        relation = node.outputs[0].relation
        if not isinstance(relation, AggregateOf):
            raise TypeError("Reduce node requires AggregateOf outputs")
        return Reduce(
            op_cls,
            *args,
            name=node.name,
            num_outputs=len(node.outputs),
            missing_child=relation.incomplete,
            workers=node.workers,
            recovery=node.recovery,
            **kwargs,
        )

    def execute(
        self,
        node: NodeSpec,
        runtime: Reduce,
        inputs: tuple[PortBatch, ...],
        *,
        force_inline: bool,
    ) -> NodeExecution:
        return NodeExecution(_outputs(runtime(group_by(inputs[0], *inputs[1:]))))


class RelateHandler:
    def prepare(self, node: NodeSpec) -> Relate:
        operation = node.operation
        if not isinstance(operation, RelateOp):
            raise TypeError("RelateHandler requires RelateOp")
        op_cls, args, kwargs = _factory(node)
        relation = node.outputs[0].relation
        if not isinstance(relation, RelatedFrom):
            raise TypeError("Relate node requires RelatedFrom output")
        roles = tuple(binding.role for binding in relation.roles)
        on = None
        adapter = None
        if isinstance(operation.matcher, KeyJoinSpec):
            on = dict(operation.matcher.fields)
        elif isinstance(operation.matcher, RelationAdapterSpec):
            adapter = operation.matcher.import_path
        return Relate(
            op_cls,
            *args,
            name=node.name,
            output_grain=node.outputs[0].grain,
            roles=roles,
            on=on,
            relation_adapter=adapter,
            workers=node.workers,
            recovery=node.recovery,
            **kwargs,
        )

    def execute(
        self,
        node: NodeSpec,
        runtime: Relate,
        inputs: tuple[PortBatch, ...],
        *,
        force_inline: bool,
    ) -> NodeExecution:
        return NodeExecution(_outputs(runtime(*inputs)))


class FilterByMaskHandler:
    def prepare(self, node: NodeSpec) -> None:
        return None

    def execute(
        self,
        node: NodeSpec,
        runtime: None,
        inputs: tuple[PortBatch, ...],
        *,
        force_inline: bool,
    ) -> NodeExecution:
        operation = node.operation
        if not isinstance(operation, FilterByMaskOp):
            raise TypeError("FilterByMaskHandler requires FilterByMaskOp")
        return NodeExecution(
            select_filter_outputs(
                inputs,
                mask_index=operation.mask_input,
                output_count=len(node.outputs),
                op_name=node.name,
            )
        )


class OperationHandlerRegistry:
    def __init__(self, handlers: Mapping[type[Any], OperationHandler]) -> None:
        self._handlers = dict(handlers)

    def resolve(self, node: NodeSpec) -> OperationHandler:
        handler = self._handlers.get(type(node.operation))
        if handler is None:
            raise NotImplementedError(
                f"no handler for operation {type(node.operation).__name__}"
            )
        return handler


DEFAULT_HANDLER_REGISTRY = OperationHandlerRegistry(
    {
        MapOp: MapHandler(),
        ExpandOp: ExpandHandler(),
        FilterOp: FilterHandler(),
        ReduceOp: ReduceHandler(),
        RelateOp: RelateHandler(),
        FilterByMaskOp: FilterByMaskHandler(),
    }
)


__all__ = [
    "DEFAULT_HANDLER_REGISTRY",
    "ExpandHandler",
    "FilterByMaskHandler",
    "FilterHandler",
    "MapHandler",
    "OperationHandler",
    "OperationHandlerRegistry",
    "ReduceHandler",
    "RelateHandler",
]
