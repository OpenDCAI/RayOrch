"""Local executor for the experimental multigrain IR."""
from __future__ import annotations

import importlib
import time
from typing import Any, Callable, Mapping

from ._op_utils import take_with_lineage
from .core import PortBatch, group_by
from .graph import IRNode, IRPortRef, MultigrainIR, NodeKind
from .metrics import NodeMetric, RunMetrics
from .ops import Expand, Filter, Map, Reduce, Relate


class MultigrainExecutor:
    """Execute ``MultigrainIR`` locally using the eager operator wrappers.

    This executor is deliberately small. Its job is to prove that the compiled
    IR is not just display metadata: it can drive local execution without the
    original live ``Pipeline`` object.
    """

    def __init__(
        self,
        *,
        relation_fns: Mapping[str, Callable[[Any], Any]] | None = None,
        metrics: RunMetrics | None = None,
    ) -> None:
        self.relation_fns = dict(relation_fns or {})
        self.metrics = metrics
        # Factory-pattern cache: one operator wrapper per node, built once and
        # reused across calls. Because the wrapper instantiates its op lazily
        # (LazyOp), a model-holding op loads exactly once per executor instance
        # -- which, inside a persistent Ray actor, means once per replica.
        self._wrappers: dict[str, Any] = {}

    def execute(
        self,
        graph: MultigrainIR,
        inputs: Mapping[str, PortBatch],
    ) -> PortBatch | tuple[PortBatch, ...]:
        context: dict[IRPortRef, PortBatch] = {}
        for port in graph.inputs:
            if port.name not in inputs:
                raise KeyError(f"missing input port '{port.name}'")
            context[port.ref] = inputs[port.name]

        for node in graph.nodes:
            node_inputs = tuple(context[ref] for ref in node.input_refs)
            start = time.perf_counter()
            outputs = self._execute_node(node, node_inputs)
            elapsed = time.perf_counter() - start
            if self.metrics is not None:
                self.metrics.record(
                    NodeMetric(
                        name=node.name,
                        kind=node.kind.value,
                        replicas=1,
                        rows_in=len(node_inputs[0]) if node_inputs else 0,
                        rows_out=len(outputs[0]) if outputs else 0,
                        wall_s=elapsed,
                        shard_busy_s=[elapsed],
                    )
                )
            if len(outputs) != len(node.output_refs):
                raise ValueError(
                    f"node {node.name} produced {len(outputs)} outputs, "
                    f"expected {len(node.output_refs)}"
                )
            for ref, batch in zip(node.output_refs, outputs):
                context[ref] = batch

        result = tuple(context[ref] for ref in graph.graph_outputs)
        return result[0] if len(result) == 1 else result

    def _execute_node(
        self,
        node: IRNode,
        inputs: tuple[PortBatch, ...],
    ) -> tuple[PortBatch, ...]:
        if node.kind is NodeKind.PROJECT:
            return _as_output_tuple(inputs)
        if node.kind in (NodeKind.REBATCH, NodeKind.MATERIALIZE):
            if len(inputs) != 1:
                raise ValueError(f"{node.kind.value} expects exactly one input")
            return _as_output_tuple(inputs[0])
        if node.kind is NodeKind.FILTER and node.op.cls_ref.endswith("SelectFilter"):
            return self._execute_select_filter(node, inputs)

        wrapper = self._wrapper_for(node)
        if node.kind is NodeKind.REDUCE:
            result = wrapper(group_by(inputs[0], *inputs[1:]))
        else:
            result = wrapper(*inputs)
        return _as_output_tuple(result)

    def _wrapper_for(self, node: IRNode) -> Any:
        """Build (and cache) the operator wrapper for a node.

        Mirrors ``RayModule``'s factory: the wrapper stores the op class + init
        args and instantiates the op lazily on first call. Caching it per node
        means the op (and any heavy model in its ``__init__``) is built exactly
        once per executor instance -- i.e. once per replica when this executor
        lives inside a persistent Ray actor.
        """
        cached = self._wrappers.get(node.name)
        if cached is not None:
            return cached

        op_cls = _load_object(node.op.cls_ref)
        args = tuple(node.op.args)
        kwargs = dict(node.op.kwargs)
        output_count = len(node.output_refs)

        if node.kind is NodeKind.MAP:
            wrapper: Any = Map(
                op_cls,
                *args,
                name=node.name,
                num_outputs=output_count,
                properties=node.properties,
                physical=node.physical,
                **kwargs,
            )
        elif node.kind is NodeKind.EXPAND:
            wrapper = Expand(
                op_cls,
                *args,
                parent=node.parent_input or 0,
                child_label=node.op.provenance.get("child_label"),
                name=node.name,
                num_outputs=output_count,
                properties=node.properties,
                physical=node.physical,
                **kwargs,
            )
        elif node.kind is NodeKind.FILTER:
            wrapper = Filter(
                op_cls,
                *args,
                name=node.name,
                properties=node.properties,
                physical=node.physical,
                **kwargs,
            )
        elif node.kind is NodeKind.REDUCE:
            wrapper = Reduce(
                op_cls,
                *args,
                name=node.name,
                num_outputs=output_count,
                missing_child=node.op.provenance.get("missing_child", "fail_open"),
                properties=node.properties,
                physical=node.physical,
                **kwargs,
            )
        elif node.kind is NodeKind.RELATE:
            roles = (
                node.contract.relations[0].roles
                if node.contract.relations
                else ()
            )
            provenance = node.op.provenance
            on = provenance.get("on")
            relation_adapter = provenance.get("relation_adapter")
            relation_fn = self.relation_fns.get(node.name)
            if on is None and relation_adapter is None and relation_fn is None:
                raise NotImplementedError(
                    f"Relate node '{node.name}' needs on=, relation_adapter, "
                    "or a registered relation_fn for local execution"
                )
            wrapper = Relate(
                op_cls,
                *args,
                name=node.name,
                output_grain=node.contract.output_grains[0],
                roles=roles,
                on=on,
                relation_adapter=relation_adapter,
                relation_fn=relation_fn,
                num_outputs=output_count,
                properties=node.properties,
                physical=node.physical,
                **kwargs,
            )
        else:
            raise NotImplementedError(f"cannot execute node kind {node.kind.value}")

        self._wrappers[node.name] = wrapper
        return wrapper

    def warm(self, node: IRNode) -> None:
        """Force the node's op to instantiate now (e.g. load its model).

        Lets a persistent actor pay the model-load cost at creation instead of on
        the first shard, so per-shard timings are cold-start free.
        """
        wrapper = self._wrapper_for(node)
        _ = getattr(wrapper, "op", None)

    def _execute_select_filter(
        self,
        node: IRNode,
        inputs: tuple[PortBatch, ...],
    ) -> tuple[PortBatch, ...]:
        """Run the synthetic SelectFilter node: drop the mask column and keep rows."""
        mask_index = int(node.op.provenance.get("mask_input", "0"))
        if mask_index < 0 or mask_index >= len(inputs):
            raise ValueError(f"SelectFilter mask index {mask_index} out of range")
        mask_port = inputs[mask_index]
        kept = [i for i, keep in enumerate(mask_port.values) if keep]
        data_ports = [port for idx, port in enumerate(inputs) if idx != mask_index]
        if len(data_ports) != len(node.output_refs):
            raise ValueError(
                f"SelectFilter {node.name} has {len(data_ports)} data ports for "
                f"{len(node.output_refs)} outputs"
            )
        return tuple(
            take_with_lineage(port, kept, name=port.name, op_name=node.name)
            for port in data_ports
        )


def _load_object(ref: str) -> Any:
    module_name, _, attr = ref.rpartition(".")
    if not module_name or not attr:
        raise ValueError(f"invalid object reference: {ref}")
    module = importlib.import_module(module_name)
    target: Any = module
    for part in attr.split("."):
        target = getattr(target, part)
    return target


def _as_output_tuple(value: Any) -> tuple[PortBatch, ...]:
    if isinstance(value, PortBatch):
        return (value,)
    if isinstance(value, tuple) and all(isinstance(item, PortBatch) for item in value):
        return value
    raise TypeError("local multigrain executor expected PortBatch outputs")


__all__ = ["MultigrainExecutor"]
