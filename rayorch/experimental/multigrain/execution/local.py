"""Local executor for the experimental multigrain IR."""
from __future__ import annotations

import time
from typing import Any, Callable, Mapping

from .handlers import DEFAULT_HANDLER_REGISTRY, PrimitiveHandlerRegistry
from ..data.batch import NodeExecution, PortBatch
from ..ir.model import IRNode, IRPortRef, MultigrainIR, NodeKind, RetryTiming
from .metrics import NodeMetric, RunMetrics


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
        handlers: PrimitiveHandlerRegistry | None = None,
    ) -> None:
        self.relation_fns = dict(relation_fns or {})
        self.metrics = metrics
        self.handlers = handlers or DEFAULT_HANDLER_REGISTRY
        # Factory-pattern cache: one prepared runtime per node and executor.
        self._runtimes: dict[str, Any] = {}
        self._runtime_nodes: dict[str, IRNode] = {}

    def execute(
        self,
        graph: MultigrainIR,
        inputs: Mapping[str, PortBatch],
    ) -> PortBatch | tuple[PortBatch, ...]:
        unsupported = [
            node.name
            for node in graph.nodes
            if node.recovery.max_record_retries > 0
            and (
                node.kind is not NodeKind.MAP
                or node.recovery.retry_timing is RetryTiming.DEFERRED
            )
        ]
        if unsupported:
            raise NotImplementedError(
                "local recovery currently supports inline Map record retry only; "
                f"nodes={unsupported}"
            )
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
        result = self._execute_node_result(node, inputs)
        if result.deferred:
            raise NotImplementedError(
                "deferred node results require the streaming coordinator"
            )
        return result.outputs

    def _execute_node_result(
        self,
        node: IRNode,
        inputs: tuple[PortBatch, ...],
        *,
        force_inline: bool = False,
    ) -> NodeExecution:
        handler = self.handlers.resolve(node)
        runtime = self._runtime_for(node, handler)
        return handler.execute(
            self,
            node,
            runtime,
            inputs,
            force_inline=force_inline,
        )

    def _runtime_for(self, node: IRNode, handler: Any | None = None) -> Any:
        """Prepare and cache one handler runtime per node/executor replica."""
        if node.name in self._runtimes:
            if self._runtime_nodes[node.name] != node:
                raise ValueError(
                    f"operator cache name collision for node '{node.name}'; "
                    "use a separate executor for a different graph/node recipe"
                )
            return self._runtimes[node.name]
        resolved = handler or self.handlers.resolve(node)
        runtime = resolved.prepare(self, node)
        self._runtimes[node.name] = runtime
        self._runtime_nodes[node.name] = node
        return runtime

    def warm(self, node: IRNode) -> None:
        """Force the node's op to instantiate now (e.g. load its model).

        Lets a persistent actor pay the model-load cost at creation instead of on
        the first shard, so per-shard timings are cold-start free.
        """
        runtime = self._runtime_for(node)
        _ = getattr(runtime, "op", None)

__all__ = ["MultigrainExecutor"]
