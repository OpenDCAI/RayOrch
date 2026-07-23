"""The sole static graph IR for Multigrain V2.2."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, TypeAlias

from ray import cloudpickle

from .identity import DomainId, NodeId, PortId


@dataclass(frozen=True, slots=True)
class ModuleConfig:
    replicas: int = 1
    batch_size: int = 64
    num_cpus: float = 1.0
    num_gpus: float = 0.0
    resources: tuple[tuple[str, float], ...] = ()
    runtime_env: Mapping[str, object] | None = None
    max_retries: int = 1
    timeout_s: float | None = None


@dataclass(frozen=True, slots=True)
class OperatorFactory:
    payload: bytes

    def build(self) -> object:
        udf_cls, args, kwargs = cloudpickle.loads(self.payload)
        return udf_cls(*args, **kwargs)


@dataclass(frozen=True, slots=True)
class CompiledPort:
    id: PortId
    node: NodeId | None
    slot: int
    domain: DomainId

    def __post_init__(self) -> None:
        if type(self.slot) is not int or self.slot < 0:
            raise ValueError("CompiledPort slot must be a non-negative int")


@dataclass(frozen=True, slots=True)
class SourceRelation:
    pass


@dataclass(frozen=True, slots=True)
class SameAs:
    role_names: tuple[str, ...]
    parents: tuple[PortId, ...]


@dataclass(frozen=True, slots=True)
class SubsetOf:
    target: PortId
    controls: tuple[PortId, ...]


@dataclass(frozen=True, slots=True)
class ChildrenOf:
    parent: PortId
    context_ports: tuple[PortId, ...]


@dataclass(frozen=True, slots=True)
class AggregateOf:
    anchor: PortId
    role_names: tuple[str, ...]
    member_ports: tuple[PortId, ...]


@dataclass(frozen=True, slots=True)
class RelatedFrom:
    role_names: tuple[str, ...]
    role_ports: tuple[PortId, ...]


PortRelation: TypeAlias = (
    SourceRelation | SameAs | SubsetOf | ChildrenOf | AggregateOf | RelatedFrom
)


def _validate_role_names(role_names: tuple[str, ...], owner: str) -> None:
    if any(type(name) is not str or not name for name in role_names):
        raise ValueError(f"{owner} role names must be non-empty strings")
    if len(set(role_names)) != len(role_names):
        raise ValueError(f"{owner} role names must be unique")


@dataclass(frozen=True, slots=True)
class MapSpec:
    primary: PortId
    role_names: tuple[str, ...]
    inputs: tuple[PortId, ...]

    def __post_init__(self) -> None:
        if not self.inputs or self.primary != self.inputs[0]:
            raise ValueError("MapSpec primary must be the first input")
        if len(self.role_names) != len(self.inputs):
            raise ValueError("MapSpec role_names must align with inputs")
        _validate_role_names(self.role_names, "MapSpec")


@dataclass(frozen=True, slots=True)
class FilterSpec:
    mode: Literal["predicate", "mask", "select"]
    role_names: tuple[str, ...]
    control_inputs: tuple[PortId, ...]
    targets: tuple[PortId, ...]
    annotation_arity: int = 0

    def __post_init__(self) -> None:
        if self.mode not in ("predicate", "mask", "select"):
            raise ValueError(f"unsupported FilterSpec mode {self.mode!r}")
        if not self.targets:
            raise ValueError("FilterSpec requires at least one target")
        if (
            type(self.annotation_arity) is not int
            or self.annotation_arity < 0
        ):
            raise ValueError(
                "FilterSpec annotation_arity must be a non-negative int"
            )
        if self.mode in ("predicate", "mask"):
            if len(self.control_inputs) != 1 or self.annotation_arity != 0:
                raise ValueError(
                    f"{self.mode} FilterSpec requires one control and no annotations"
                )
            expected_roles = 1 if self.mode == "predicate" else 0
            if len(self.role_names) != expected_roles:
                raise ValueError(
                    f"{self.mode} FilterSpec requires {expected_roles} UDF roles"
                )
            if self.role_names:
                _validate_role_names(self.role_names, "FilterSpec")
        elif (
            not self.control_inputs
            or self.control_inputs != self.targets
            or len(self.role_names) != len(self.control_inputs)
        ):
            raise ValueError(
                "select FilterSpec requires controls == targets with aligned roles"
            )
        else:
            _validate_role_names(self.role_names, "FilterSpec")


@dataclass(frozen=True, slots=True)
class ExpandSpec:
    parent: PortId
    role_names: tuple[str, ...]
    inputs: tuple[PortId, ...]

    def __post_init__(self) -> None:
        if not self.inputs or self.parent != self.inputs[0]:
            raise ValueError("ExpandSpec parent must be the first input")
        if len(self.role_names) != len(self.inputs):
            raise ValueError("ExpandSpec role_names must align with inputs")
        _validate_role_names(self.role_names, "ExpandSpec")


@dataclass(frozen=True, slots=True)
class ReduceMemberSpec:
    role: str
    port: PortId

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError("Reduce member role must be non-empty")


@dataclass(frozen=True, slots=True)
class ReduceSpec:
    anchor: PortId
    members: tuple[ReduceMemberSpec, ...]

    def __post_init__(self) -> None:
        _validate_role_names(
            tuple(member.role for member in self.members),
            "ReduceSpec",
        )


@dataclass(frozen=True, slots=True)
class RelateRoleSpec:
    name: str
    value_port: PortId
    key_port: PortId | None

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise ValueError("Relate role name must be a non-empty string")


@dataclass(frozen=True, slots=True)
class RelateSpec:
    mode: Literal["key", "custom"]
    roles: tuple[RelateRoleSpec, ...]

    def __post_init__(self) -> None:
        if self.mode not in ("key", "custom"):
            raise ValueError(f"unsupported RelateSpec mode {self.mode!r}")
        if not self.roles:
            raise ValueError("RelateSpec requires at least one role")
        names = tuple(role.name for role in self.roles)
        _validate_role_names(names, "RelateSpec")
        if self.mode == "key" and any(role.key_port is None for role in self.roles):
            raise ValueError("key RelateSpec requires a key port for every role")
        if self.mode == "custom" and any(
            role.key_port is not None for role in self.roles
        ):
            raise ValueError("custom RelateSpec cannot have planning key ports")


OperationSpec: TypeAlias = MapSpec | FilterSpec | ExpandSpec | ReduceSpec | RelateSpec


@dataclass(frozen=True, slots=True)
class CompiledNode:
    id: NodeId
    spec: OperationSpec
    factory: OperatorFactory | None
    config: ModuleConfig
    accepts_context: bool
    output_ports: tuple[PortId, ...]


@dataclass(frozen=True, slots=True)
class CompiledInputGroup:
    name: str
    position: int
    domain: DomainId
    port: PortId

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise ValueError("input group name must be a non-empty string")
        if type(self.position) is not int or self.position < 0:
            raise ValueError("input group position must be a non-negative int")


def _spec_ports(spec: OperationSpec) -> tuple[PortId, ...]:
    if isinstance(spec, (MapSpec, ExpandSpec)):
        return spec.inputs
    if isinstance(spec, FilterSpec):
        return spec.control_inputs + spec.targets
    if isinstance(spec, ReduceSpec):
        return (spec.anchor,) + tuple(member.port for member in spec.members)
    if isinstance(spec, RelateSpec):
        return tuple(
            port
            for role in spec.roles
            for port in (
                (role.value_port,)
                if role.key_port is None
                else (role.value_port, role.key_port)
            )
        )
    raise TypeError(f"unsupported operation spec {type(spec).__name__}")


@dataclass(frozen=True, slots=True)
class CompiledGraph:
    nodes: tuple[CompiledNode, ...]
    ports: tuple[CompiledPort, ...]
    inputs: tuple[CompiledInputGroup, ...]
    outputs: tuple[PortId, ...]
    topological_nodes: tuple[NodeId, ...]

    def __post_init__(self) -> None:
        node_ids = tuple(node.id for node in self.nodes)
        port_ids = tuple(port.id for port in self.ports)
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("CompiledGraph node IDs must be unique")
        if len(set(port_ids)) != len(port_ids):
            raise ValueError("CompiledGraph port IDs must be unique")
        if set(self.topological_nodes) != set(node_ids) or len(
            self.topological_nodes
        ) != len(node_ids):
            raise ValueError("topological_nodes must contain every node exactly once")
        known_ports = set(port_ids)
        if any(port not in known_ports for port in self.outputs):
            raise ValueError("graph output references an unknown port")
        if any(group.port not in known_ports for group in self.inputs):
            raise ValueError("input group references an unknown port")
        by_port = {port.id: port for port in self.ports}
        by_node = {node.id: node for node in self.nodes}
        if len({group.name for group in self.inputs}) != len(self.inputs):
            raise ValueError("input group names must be unique")
        if len({group.port for group in self.inputs}) != len(self.inputs):
            raise ValueError("input group source ports must be unique")
        if tuple(sorted(group.position for group in self.inputs)) != tuple(
            range(len(self.inputs))
        ):
            raise ValueError("input group positions must be contiguous")
        source_ports = {port.id for port in self.ports if port.node is None}
        if source_ports != {group.port for group in self.inputs}:
            raise ValueError(
                "every source port must belong to exactly one input group"
            )
        if any(
            port.node is not None and port.node not in by_node
            for port in self.ports
        ):
            raise ValueError("compiled port references an unknown producer node")
        for group in self.inputs:
            source = by_port[group.port]
            if source.node is not None or source.domain != group.domain:
                raise ValueError("input groups must reference matching source ports")
        topological_position = {
            node: position
            for position, node in enumerate(self.topological_nodes)
        }
        for node in self.nodes:
            dependencies = _spec_ports(node.spec)
            if any(port not in known_ports for port in dependencies):
                raise ValueError("operation spec references an unknown port")
            if isinstance(node.spec, FilterSpec):
                expected = len(node.spec.targets) + node.spec.annotation_arity
                if len(node.output_ports) != expected:
                    raise ValueError(
                        "Filter node output count must match targets and annotations"
                    )
            for expected_slot, output in enumerate(node.output_ports):
                port = by_port.get(output)
                if port is None or port.node != node.id:
                    raise ValueError("node output_ports must reference its own ports")
                if port.slot != expected_slot:
                    raise ValueError(
                        "node output port slots must be contiguous and ordered"
                    )
            for dependency in dependencies:
                producer = by_port[dependency].node
                if (
                    producer is not None
                    and topological_position[producer]
                    >= topological_position[node.id]
                ):
                    raise ValueError(
                        "topological_nodes must order producers before consumers"
                    )
        for port in self.ports:
            if port.node is None:
                continue
            producer = by_node.get(port.node)
            if producer is None:
                raise ValueError("compiled port references an unknown producer node")
            if port.id not in producer.output_ports:
                raise ValueError(
                    "compiled port must be listed by its producer node"
                )


def _ordered_unique(ports: tuple[PortId, ...]) -> tuple[PortId, ...]:
    return tuple(dict.fromkeys(ports))


def ordered_input_ports(spec: OperationSpec) -> tuple[PortId, ...]:
    if isinstance(spec, (MapSpec, ExpandSpec)):
        return spec.inputs
    if isinstance(spec, FilterSpec):
        if spec.mode == "select":
            return spec.control_inputs
        additional_targets = tuple(
            target
            for target in spec.targets
            if target not in spec.control_inputs
        )
        return spec.control_inputs + additional_targets
    if isinstance(spec, ReduceSpec):
        return (spec.anchor,) + tuple(member.port for member in spec.members)
    if isinstance(spec, RelateSpec):
        # Planning keys are driver-materialized through RelateRoleSpec.key_port.
        # Only role value columns enter the actor-side transport ABI.
        return tuple(role.value_port for role in spec.roles)
    raise TypeError(f"unsupported operation spec {type(spec).__name__}")


def _controls_for(
    inputs: tuple[PortId, ...],
    target: PortId,
) -> tuple[PortId, ...]:
    return _ordered_unique(tuple(port for port in inputs if port != target))


def relation_of(graph: CompiledGraph, port: PortId) -> PortRelation:
    ports = {compiled.id: compiled for compiled in graph.ports}
    compiled_port = ports.get(port)
    if compiled_port is None:
        raise KeyError(f"unknown PortId {port!r}")
    if compiled_port.node is None:
        return SourceRelation()

    nodes = {node.id: node for node in graph.nodes}
    node = nodes.get(compiled_port.node)
    if node is None or port not in node.output_ports:
        raise ValueError("port producer is inconsistent with CompiledGraph")
    output_slot = compiled_port.slot
    spec = node.spec

    if isinstance(spec, MapSpec):
        return SameAs(spec.role_names, spec.inputs)
    if isinstance(spec, ExpandSpec):
        return ChildrenOf(
            spec.parent,
            _controls_for(spec.inputs, spec.parent),
        )
    if isinstance(spec, ReduceSpec):
        return AggregateOf(
            spec.anchor,
            tuple(member.role for member in spec.members),
            tuple(member.port for member in spec.members),
        )
    if isinstance(spec, RelateSpec):
        return RelatedFrom(
            tuple(role.name for role in spec.roles),
            tuple(role.value_port for role in spec.roles),
        )
    if isinstance(spec, FilterSpec):
        target_count = len(spec.targets)
        target = (
            spec.targets[output_slot]
            if output_slot < target_count
            else spec.targets[0]
        )
        if output_slot >= target_count + spec.annotation_arity:
            raise ValueError("Filter output slot exceeds declared output arity")
        return SubsetOf(
            target,
            _controls_for(spec.control_inputs, target),
        )
    raise TypeError(f"unsupported operation spec {type(spec).__name__}")


__all__ = [
    "AggregateOf",
    "ChildrenOf",
    "CompiledGraph",
    "CompiledInputGroup",
    "CompiledNode",
    "CompiledPort",
    "ExpandSpec",
    "FilterSpec",
    "MapSpec",
    "ModuleConfig",
    "OperationSpec",
    "OperatorFactory",
    "PortRelation",
    "ReduceMemberSpec",
    "ReduceSpec",
    "RelateRoleSpec",
    "RelateSpec",
    "RelatedFrom",
    "SameAs",
    "SourceRelation",
    "SubsetOf",
    "ordered_input_ports",
    "relation_of",
]
