"""Torch-like tracing that lowers authoring primitives to an ExecutionGraph."""
from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any, Sequence

from .ir.graph import ExecutionGraph, GraphInputSpec, NodeSpec, OutputSpec
from .ir.operations import OperationSpec
from .ir.policy import RecoveryPolicy, WorkerPoolSpec
from .ir.refs import GraphInputRef, NodeOutputRef, PortRef
from .ir.relations import OutputRelation


@dataclass(frozen=True)
class TracePort:
    """Trace-time token; never stored in the passive execution graph."""

    ref: PortRef
    grain: str
    tracer: "GraphTracer"

    @property
    def name(self) -> str:
        if isinstance(self.ref, GraphInputRef):
            return self.ref.name
        return self.ref.output


class GraphTracer:
    def __init__(self, *, name: str = "pipeline") -> None:
        self.name = name
        self.nodes: list[NodeSpec] = []
        self._name_counts: dict[str, int] = {}

    def source(self, name: str) -> TracePort:
        return TracePort(GraphInputRef(name), name, self)

    def unique_name(self, base: str) -> str:
        count = self._name_counts.get(base, 0)
        self._name_counts[base] = count + 1
        return base if count == 0 else f"{base}_{count}"

    def add_node(
        self,
        *,
        name: str,
        inputs: Sequence[TracePort],
        operation: OperationSpec,
        output_grains: Sequence[str],
        relations: Sequence[OutputRelation],
        output_names: Sequence[str] | None = None,
        workers: WorkerPoolSpec | None = None,
        recovery: RecoveryPolicy | None = None,
    ) -> TracePort | tuple[TracePort, ...]:
        if not inputs:
            raise ValueError("a traced node requires at least one input")
        if any(port.tracer is not self for port in inputs):
            raise ValueError("cannot mix trace ports from different traces")
        if len(output_grains) != len(relations):
            raise ValueError("each traced output requires one grain and relation")
        if not output_grains:
            raise ValueError("a traced node requires at least one output")

        node_name = self.unique_name(name)
        names = tuple(output_names or ())
        if names and len(names) != len(output_grains):
            raise ValueError("output_names must match output_grains")
        if not names:
            names = tuple(
                "out" if index == 0 else f"out_{index}"
                for index in range(len(output_grains))
            )
        if len(set(names)) != len(names):
            raise ValueError(f"node '{node_name}' has duplicate output names")

        outputs = tuple(
            TracePort(NodeOutputRef(node_name, output_name), grain, self)
            for output_name, grain in zip(names, output_grains)
        )
        self.nodes.append(
            NodeSpec(
                name=node_name,
                inputs=tuple(port.ref for port in inputs),
                outputs=tuple(
                    OutputSpec(port.ref, port.grain, relation)
                    for port, relation in zip(outputs, relations)
                    if isinstance(port.ref, NodeOutputRef)
                ),
                operation=operation,
                workers=workers or WorkerPoolSpec(),
                recovery=recovery or RecoveryPolicy(),
            )
        )
        return outputs[0] if len(outputs) == 1 else outputs

    def build(
        self,
        inputs: Sequence[TracePort],
        outputs: Any,
    ) -> ExecutionGraph:
        graph = ExecutionGraph(
            name=self.name,
            inputs=tuple(
                GraphInputSpec(port.ref, port.grain)
                for port in inputs
                if isinstance(port.ref, GraphInputRef)
            ),
            nodes=tuple(self.nodes),
            outputs=tuple(port.ref for port in _flatten_trace_outputs(outputs)),
        )
        from .ir.verify import verify_graph

        verify_graph(graph)
        return graph


class Pipeline:
    """Authoring facade whose ``forward`` is traced once into an execution graph."""

    def __init__(self) -> None:
        self._compiled: ExecutionGraph | None = None

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def compile(self) -> ExecutionGraph:
        tracer = GraphTracer(name=type(self).__name__)
        signature = inspect.signature(self.forward)
        inputs: list[TracePort] = []
        args: list[TracePort] = []
        kwargs: dict[str, TracePort] = {}
        for parameter in signature.parameters.values():
            if parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                raise TypeError(
                    "forward() cannot use *args/**kwargs in multigrain tracing"
                )
            port = tracer.source(parameter.name)
            inputs.append(port)
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                args.append(port)
            elif parameter.kind == inspect.Parameter.KEYWORD_ONLY:
                kwargs[parameter.name] = port
        self._compiled = tracer.build(inputs, self.forward(*args, **kwargs))
        return self._compiled


def _flatten_trace_outputs(value: Any) -> list[TracePort]:
    if isinstance(value, TracePort):
        return [value]
    if isinstance(value, tuple):
        ports: list[TracePort] = []
        for item in value:
            ports.extend(_flatten_trace_outputs(item))
        return ports
    raise TypeError("Pipeline.forward() must return TracePort or tuple[TracePort, ...]")


def ensure_trace_ports(values: Sequence[Any]) -> tuple[TracePort, ...] | None:
    if not values:
        return None
    if isinstance(values[0], TracePort):
        if not all(isinstance(value, TracePort) for value in values):
            raise TypeError("cannot mix trace and eager ports")
        return tuple(values)
    return None


__all__ = ["GraphTracer", "Pipeline", "TracePort", "ensure_trace_ports"]
