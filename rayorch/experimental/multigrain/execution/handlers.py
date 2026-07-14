"""Runtime handler registry for passive multigrain IR nodes.

Purpose: provide one authoritative dispatch point for compiled execution.
Handlers adapt IR recipes to the same eager wrappers users call directly; they
do not own actor pools, scheduling, or recovery lifecycle.
"""
from __future__ import annotations

from typing import Any, Mapping, Protocol

from ..data.batch import NodeExecution, PortBatch, group_by
from ..ir.model import (
    IRNode,
    MATERIALIZE_RECIPE,
    NodeKind,
    PROJECT_RECIPE,
    REBATCH_RECIPE,
    SELECT_FILTER_RECIPE,
)
from ..primitives._binding import load_recipe_object
from ..primitives.expand_reduce import Expand, Reduce
from ..primitives.map_filter import Filter, Map
from ..primitives.output import select_filter_outputs
from ..primitives.relate import Relate


class HandlerContext(Protocol):
    relation_fns: Mapping[str, Any]


class PrimitiveHandler(Protocol):
    def prepare(self, context: HandlerContext, node: IRNode) -> Any:
        ...

    def execute(
        self,
        context: HandlerContext,
        node: IRNode,
        runtime: Any,
        inputs: tuple[PortBatch, ...],
        *,
        force_inline: bool,
    ) -> NodeExecution:
        ...


def _recipe(node: IRNode) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
    return (
        load_recipe_object(node.op.cls_ref),
        tuple(node.op.args),
        dict(node.op.kwargs),
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
    def prepare(self, context: HandlerContext, node: IRNode) -> Map:
        op_cls, args, kwargs = _recipe(node)
        return Map(
            op_cls,
            *args,
            name=node.name,
            num_outputs=len(node.output_refs),
            properties=node.properties,
            physical=node.physical,
            recovery=node.recovery,
            **kwargs,
        )

    def execute(
        self,
        context: HandlerContext,
        node: IRNode,
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
    def prepare(self, context: HandlerContext, node: IRNode) -> Expand:
        op_cls, args, kwargs = _recipe(node)
        return Expand(
            op_cls,
            *args,
            parent=node.parent_input or 0,
            child_label=node.op.provenance.get("child_label"),
            name=node.name,
            num_outputs=len(node.output_refs),
            properties=node.properties,
            physical=node.physical,
            recovery=node.recovery,
            **kwargs,
        )

    def execute(
        self,
        context: HandlerContext,
        node: IRNode,
        runtime: Expand,
        inputs: tuple[PortBatch, ...],
        *,
        force_inline: bool,
    ) -> NodeExecution:
        return NodeExecution(_outputs(runtime(*inputs)))


class FilterHandler:
    def prepare(self, context: HandlerContext, node: IRNode) -> Filter:
        op_cls, args, kwargs = _recipe(node)
        return Filter(
            op_cls,
            *args,
            name=node.name,
            properties=node.properties,
            physical=node.physical,
            recovery=node.recovery,
            **kwargs,
        )

    def execute(
        self,
        context: HandlerContext,
        node: IRNode,
        runtime: Filter,
        inputs: tuple[PortBatch, ...],
        *,
        force_inline: bool,
    ) -> NodeExecution:
        return NodeExecution(_outputs(runtime(*inputs)))


class ReduceHandler:
    def prepare(self, context: HandlerContext, node: IRNode) -> Reduce:
        op_cls, args, kwargs = _recipe(node)
        return Reduce(
            op_cls,
            *args,
            name=node.name,
            num_outputs=len(node.output_refs),
            missing_child=node.op.provenance.get("missing_child", "fail_open"),
            properties=node.properties,
            physical=node.physical,
            recovery=node.recovery,
            **kwargs,
        )

    def execute(
        self,
        context: HandlerContext,
        node: IRNode,
        runtime: Reduce,
        inputs: tuple[PortBatch, ...],
        *,
        force_inline: bool,
    ) -> NodeExecution:
        return NodeExecution(_outputs(runtime(group_by(inputs[0], *inputs[1:]))))


class RelateHandler:
    def prepare(self, context: HandlerContext, node: IRNode) -> Relate:
        op_cls, args, kwargs = _recipe(node)
        roles = (
            node.contract.relations[0].roles
            if node.contract.relations
            else ()
        )
        provenance = node.op.provenance
        on = provenance.get("on")
        adapter = provenance.get("relation_adapter")
        relation_fn = context.relation_fns.get(node.name)
        if on is None and adapter is None and relation_fn is None:
            raise NotImplementedError(
                f"Relate node '{node.name}' needs on=, relation_adapter, "
                "or a registered relation_fn for local execution"
            )
        return Relate(
            op_cls,
            *args,
            name=node.name,
            output_grain=node.contract.output_grains[0],
            roles=roles,
            on=on,
            relation_adapter=adapter,
            relation_fn=relation_fn,
            num_outputs=len(node.output_refs),
            properties=node.properties,
            physical=node.physical,
            recovery=node.recovery,
            **kwargs,
        )

    def execute(
        self,
        context: HandlerContext,
        node: IRNode,
        runtime: Relate,
        inputs: tuple[PortBatch, ...],
        *,
        force_inline: bool,
    ) -> NodeExecution:
        return NodeExecution(_outputs(runtime(*inputs)))


class ProjectHandler:
    def prepare(self, context: HandlerContext, node: IRNode) -> None:
        return None

    def execute(
        self,
        context: HandlerContext,
        node: IRNode,
        runtime: None,
        inputs: tuple[PortBatch, ...],
        *,
        force_inline: bool,
    ) -> NodeExecution:
        return NodeExecution(inputs)


class UnaryIdentityHandler:
    def prepare(self, context: HandlerContext, node: IRNode) -> None:
        return None

    def execute(
        self,
        context: HandlerContext,
        node: IRNode,
        runtime: None,
        inputs: tuple[PortBatch, ...],
        *,
        force_inline: bool,
    ) -> NodeExecution:
        if len(inputs) != 1:
            raise ValueError(f"{node.kind.value} expects exactly one input")
        return NodeExecution(inputs)


class SelectFilterHandler:
    def prepare(self, context: HandlerContext, node: IRNode) -> None:
        return None

    def execute(
        self,
        context: HandlerContext,
        node: IRNode,
        runtime: None,
        inputs: tuple[PortBatch, ...],
        *,
        force_inline: bool,
    ) -> NodeExecution:
        return NodeExecution(
            select_filter_outputs(
                inputs,
                mask_index=int(node.op.provenance.get("mask_input", "0")),
                output_count=len(node.output_refs),
                op_name=node.name,
            )
        )


class PrimitiveHandlerRegistry:
    def __init__(
        self,
        *,
        kinds: Mapping[NodeKind, PrimitiveHandler],
        recipes: Mapping[str, PrimitiveHandler] | None = None,
    ) -> None:
        self._kinds = dict(kinds)
        self._recipes = dict(recipes or {})

    def resolve(self, node: IRNode) -> PrimitiveHandler:
        handler = self._recipes.get(node.op.cls_ref) or self._kinds.get(node.kind)
        if handler is None:
            raise NotImplementedError(
                f"no primitive handler for node kind {node.kind.value}"
            )
        return handler


DEFAULT_HANDLER_REGISTRY = PrimitiveHandlerRegistry(
    kinds={
        NodeKind.MAP: MapHandler(),
        NodeKind.EXPAND: ExpandHandler(),
        NodeKind.FILTER: FilterHandler(),
        NodeKind.REDUCE: ReduceHandler(),
        NodeKind.RELATE: RelateHandler(),
        NodeKind.PROJECT: ProjectHandler(),
        NodeKind.REBATCH: UnaryIdentityHandler(),
        NodeKind.MATERIALIZE: UnaryIdentityHandler(),
    },
    recipes={
        SELECT_FILTER_RECIPE: SelectFilterHandler(),
        PROJECT_RECIPE: ProjectHandler(),
        REBATCH_RECIPE: UnaryIdentityHandler(),
        MATERIALIZE_RECIPE: UnaryIdentityHandler(),
    },
)
