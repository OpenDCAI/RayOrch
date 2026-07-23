"""Minimal passive execution graph for relation-aware multigrain pipelines."""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any, Mapping

from .operations import OperationSpec, operation_name
from .policy import RecoveryPolicy, WorkerPoolSpec
from .refs import GraphInputRef, NodeOutputRef, PortRef, ref_label
from .relations import OutputRelation


@dataclass(frozen=True)
class GraphInputSpec:
    ref: GraphInputRef
    grain: str

    @property
    def name(self) -> str:
        return self.ref.name


@dataclass(frozen=True)
class OutputSpec:
    ref: NodeOutputRef
    grain: str
    relation: OutputRelation

    @property
    def name(self) -> str:
        return self.ref.output


@dataclass(frozen=True)
class NodeSpec:
    name: str
    inputs: tuple[PortRef, ...]
    outputs: tuple[OutputSpec, ...]
    operation: OperationSpec
    workers: WorkerPoolSpec = field(default_factory=WorkerPoolSpec)
    recovery: RecoveryPolicy = field(default_factory=RecoveryPolicy)

    @property
    def output_refs(self) -> tuple[NodeOutputRef, ...]:
        return tuple(output.ref for output in self.outputs)


@dataclass(frozen=True)
class ExecutionGraph:
    """Immutable DAG consumed by validation, coordination, and execution."""

    name: str
    inputs: tuple[GraphInputSpec, ...]
    nodes: tuple[NodeSpec, ...]
    outputs: tuple[PortRef, ...]

    @property
    def dependencies(self) -> dict[str, tuple[str, ...]]:
        dependencies: dict[str, tuple[str, ...]] = {}
        for node in self.nodes:
            seen: list[str] = []
            for ref in node.inputs:
                if isinstance(ref, NodeOutputRef) and ref.node not in seen:
                    seen.append(ref.node)
            dependencies[node.name] = tuple(seen)
        return dependencies

    @property
    def consumers(self) -> dict[str, tuple[str, ...]]:
        consumers: dict[str, list[str]] = {node.name: [] for node in self.nodes}
        for node, dependencies in self.dependencies.items():
            for dependency in dependencies:
                consumers.setdefault(dependency, []).append(node)
        return {name: tuple(items) for name, items in consumers.items()}

    def node(self, name: str) -> NodeSpec:
        for node in self.nodes:
            if node.name == name:
                return node
        raise KeyError(name)

    def describe(self) -> str:
        lines = [f"ExecutionGraph({self.name})"]
        lines.append(f"  inputs: {[spec.name for spec in self.inputs]}")
        for node in self.nodes:
            inputs = ", ".join(ref_label(ref) for ref in node.inputs)
            outputs = ", ".join(
                f"{output.name}:{output.grain}/{type(output.relation).__name__}"
                for output in node.outputs
            )
            lines.append(
                f"  [{operation_name(node.operation)}] {node.name}\n"
                f"        in:  {inputs or '-'}\n"
                f"        out: {outputs or '-'}"
            )
        lines.append(
            "  outputs: " + ", ".join(ref_label(ref) for ref in self.outputs)
        )
        return "\n".join(lines)

    def to_mermaid(self) -> str:
        lines = ["flowchart TD"]
        for spec in self.inputs:
            key = _mermaid_id(spec.ref)
            lines.append(f'    {key}["in: {spec.name}:{spec.grain}"]')
        for node in self.nodes:
            node_id = _mermaid_node(node.name)
            lines.append(
                f'    {node_id}["{node.name}<br/>{operation_name(node.operation)}"]'
            )
            for output in node.outputs:
                output_id = _mermaid_id(output.ref)
                lines.append(
                    f'    {output_id}["{output.name}:{output.grain}"]'
                )
                lines.append(f"    {node_id} --> {output_id}")
            for ref in node.inputs:
                lines.append(f"    {_mermaid_id(ref)} --> {node_id}")
        lines.append("    graph_sink([out])")
        for ref in self.outputs:
            lines.append(f"    {_mermaid_id(ref)} --> graph_sink")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(self)


def _mermaid_node(name: str) -> str:
    return "node_" + "".join(ch if ch.isalnum() else "_" for ch in name)


def _mermaid_id(ref: PortRef) -> str:
    if isinstance(ref, GraphInputRef):
        return "input_" + "".join(
            ch if ch.isalnum() else "_" for ch in ref.name
        )
    return (
        _mermaid_node(ref.node)
        + "_out_"
        + "".join(ch if ch.isalnum() else "_" for ch in ref.output)
    )


def _jsonify(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "type": type(value).__name__,
            **{
                item.name: _jsonify(getattr(value, item.name))
                for item in fields(value)
            },
        }
    if isinstance(value, Mapping):
        return {key: _jsonify(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonify(item) for item in value]
    if isinstance(value, list):
        return [_jsonify(item) for item in value]
    return value


__all__ = ["ExecutionGraph", "GraphInputSpec", "NodeSpec", "OutputSpec"]
