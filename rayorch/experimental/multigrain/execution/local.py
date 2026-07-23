"""Local executor for the experimental multigrain IR."""
from __future__ import annotations

import time
from typing import Any, Mapping

from .handlers import (
    DEFAULT_HANDLER_REGISTRY,
    OperationHandlerRegistry,
    expected_output_domain,
)
from ..data.batch import NodeExecution, PortBatch
from ..ir.graph import ExecutionGraph, NodeSpec
from ..ir.operations import MapOp, operation_name
from ..ir.policy import RetryTiming
from ..ir.refs import PortRef
from ..ir.verify import verify_graph
from .metrics import NodeMetric, RunMetrics, lineage_footprint


class MultigrainExecutor:
    """Execute an ``ExecutionGraph`` locally using eager operator semantics.

    This executor is deliberately small. Its job is to prove that the compiled
    IR is not just display metadata: it can drive local execution without the
    original live ``Pipeline`` object.
    """

    def __init__(
        self,
        *,
        metrics: RunMetrics | None = None,
        handlers: OperationHandlerRegistry | None = None,
    ) -> None:
        self.metrics = metrics
        self.handlers = handlers or DEFAULT_HANDLER_REGISTRY
        # Factory-pattern cache: one prepared runtime per node and executor.
        self._runtimes: dict[str, Any] = {}
        self._runtime_nodes: dict[str, NodeSpec] = {}

    def execute(
        self,
        graph: ExecutionGraph,
        inputs: Mapping[str, PortBatch],
    ) -> PortBatch | tuple[PortBatch, ...]:
        verify_graph(graph)
        unsupported = [
            node.name
            for node in graph.nodes
            if node.recovery.max_record_retries > 0
            and (
                not isinstance(node.operation, MapOp)
                or node.recovery.retry_timing is RetryTiming.DEFERRED
            )
        ]
        if unsupported:
            raise NotImplementedError(
                "local recovery currently supports inline Map record retry only; "
                f"nodes={unsupported}"
            )
        context: dict[PortRef, PortBatch] = {}
        for port in graph.inputs:
            if port.name not in inputs:
                raise KeyError(f"missing input port '{port.name}'")
            batch = inputs[port.name]
            if batch.grain != port.grain:
                raise ValueError(
                    f"input '{port.name}' has runtime grain '{batch.grain}', "
                    f"expected '{port.grain}'"
                )
            context[port.ref] = batch

        for node in graph.nodes:
            node_inputs = tuple(context[ref] for ref in node.inputs)
            start = time.perf_counter()
            outputs = self._execute_node(node, node_inputs)
            elapsed = time.perf_counter() - start
            if self.metrics is not None:
                footprint = lineage_footprint(outputs)
                self.metrics.record(
                    NodeMetric(
                        name=node.name,
                        kind=operation_name(node.operation),
                        replicas=1,
                        rows_in=len(node_inputs[0]) if node_inputs else 0,
                        rows_out=len(outputs[0]) if outputs else 0,
                        wall_s=elapsed,
                        shard_busy_s=[elapsed],
                        shard_rows_in=[len(node_inputs[0]) if node_inputs else 0],
                        shard_rows_out=[len(outputs[0]) if outputs else 0],
                        relation_entries_out=footprint["relation_entries"],
                        lineage_bytes_out=footprint["approx_bytes"],
                    )
                )
            if len(outputs) != len(node.outputs):
                raise ValueError(
                    f"node {node.name} produced {len(outputs)} outputs, "
                    f"expected {len(node.outputs)}"
                )
            for ref, batch in zip(node.output_refs, outputs):
                context[ref] = batch

        result = tuple(context[ref] for ref in graph.outputs)
        return result[0] if len(result) == 1 else result

    def _execute_node(
        self,
        node: NodeSpec,
        inputs: tuple[PortBatch, ...],
    ) -> tuple[PortBatch, ...]:
        result = self._execute_node_result(node, inputs)
        if result.deferred:
            raise NotImplementedError(
                "deferred node results require the streaming coordinator"
            )
        return result.outputs

    def _execute_node_result(
        self,
        node: NodeSpec,
        inputs: tuple[PortBatch, ...],
        *,
        force_inline: bool = False,
    ) -> NodeExecution:
        handler = self.handlers.resolve(node)
        runtime = self._runtime_for(node, handler)
        result = handler.execute(
            node,
            runtime,
            inputs,
            force_inline=force_inline,
        )
        if len(result.outputs) != len(node.outputs):
            raise ValueError(
                f"node {node.name} produced {len(result.outputs)} outputs, "
                f"expected {len(node.outputs)}"
            )
        for output, spec in zip(result.outputs, node.outputs):
            if output.name != spec.grain:
                raise ValueError(
                    f"node {node.name} output '{spec.name}' produced runtime "
                    f"grain '{output.name}', expected '{spec.grain}'"
                )
            expected_domain = expected_output_domain(node, inputs, spec)
            if output.identity_domain != expected_domain:
                raise ValueError(
                    f"node {node.name} output '{spec.name}' produced identity "
                    f"domain {output.identity_domain}, expected {expected_domain}"
                )
        return result

    def _runtime_for(self, node: NodeSpec, handler: Any | None = None) -> Any:
        """Prepare and cache one handler runtime per node/executor replica."""
        if node.name in self._runtimes:
            if self._runtime_nodes[node.name] != node:
                raise ValueError(
                    f"operator cache name collision for node '{node.name}'; "
                    "use a separate executor for a different graph/node factory"
                )
            return self._runtimes[node.name]
        resolved = handler or self.handlers.resolve(node)
        runtime = resolved.prepare(node)
        self._runtimes[node.name] = runtime
        self._runtime_nodes[node.name] = node
        return runtime

    def warm(self, node: NodeSpec) -> None:
        """Force the node's op to instantiate now (e.g. load its model).

        Lets a persistent actor pay the model-load cost at creation instead of on
        the first shard, so per-shard timings are cold-start free.
        """
        runtime = self._runtime_for(node)
        _ = getattr(runtime, "op", None)

__all__ = ["MultigrainExecutor"]
