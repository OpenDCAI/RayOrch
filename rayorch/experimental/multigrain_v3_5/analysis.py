"""LogicalProgram 上的纯派生事实；不包含 runtime state 或 actor 配置。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, TypeAlias

from .logical import LogicalProgram
from .model import CallRef, CompileError, DomainRef, PortRef
from .semantics import InputRole, PrimitiveKind, PrimitiveSemantics, describe_origin


@dataclass(frozen=True, slots=True)
class CallUse:
    call: CallRef
    input_index: int


@dataclass(frozen=True, slots=True)
class PrimitiveUse:
    port: PortRef
    role: InputRole


LogicalUse: TypeAlias = CallUse | PrimitiveUse


@dataclass(frozen=True, slots=True)
class DerivedFacts:
    """可丢弃、可重算的 compiler analysis 结果。"""

    semantics_by_port: Mapping[PortRef, PrimitiveSemantics] = field(repr=False)
    consumers_by_port: Mapping[PortRef, tuple[LogicalUse, ...]] = field(repr=False)
    outputs_by_call: Mapping[CallRef, tuple[PortRef, ...]] = field(repr=False)
    shape_reporters_by_domain: Mapping[DomainRef, tuple[PortRef, ...]] = field(
        repr=False
    )
    control_ports: frozenset[PortRef] = frozenset()
    group_depth_by_port: Mapping[PortRef, int] = field(repr=False, default_factory=dict)


def analyze(logical: LogicalProgram) -> DerivedFacts:
    """一次性计算所有 derived facts，control demand 走统一语义表。"""

    semantics = {
        port: describe_origin(spec.origin) for port, spec in logical.ports.items()
    }
    consumers: dict[PortRef, list[LogicalUse]] = {}
    outputs_by_call: dict[CallRef, list[tuple[int, PortRef]]] = {}
    shape_reporters: dict[DomainRef, list[PortRef]] = {}

    for call, spec in logical.calls.items():
        for index, input_ in enumerate(spec.ordered_inputs):
            consumers.setdefault(input_.port, []).append(CallUse(call, index))

    for port, semantic in semantics.items():
        for input_ in semantic.inputs:
            consumers.setdefault(input_.port, []).append(
                PrimitiveUse(port, input_.role)
            )
        if semantic.kind is PrimitiveKind.CALL_OUTPUT:
            assert semantic.producing_call is not None
            assert semantic.output_index is not None
            outputs_by_call.setdefault(semantic.producing_call, []).append(
                (semantic.output_index, port)
            )
        if semantic.kind is PrimitiveKind.EXPAND:
            group = semantic.inputs[0].port
            shape_reporters.setdefault(logical.port(port).domain, []).append(group)

    control_ports = {
        demanded
        for semantic in semantics.values()
        for demanded in semantic.control_demands
    }
    pending = list(control_ports)
    while pending:
        port = pending.pop()
        semantic = semantics[port]
        if semantic.rejects_control:
            raise CompileError(
                "group-valued Port cannot be used as a scalar filter mask"
            )
        for predecessor in semantic.control_predecessors:
            if predecessor not in control_ports:
                control_ports.add(predecessor)
                pending.append(predecessor)

    depths: dict[PortRef, int] = {}

    def group_depth(port: PortRef, visiting: set[PortRef]) -> int:
        known = depths.get(port)
        if known is not None:
            return known
        if port in visiting:
            raise CompileError("logical Port dependency cycle")
        semantic = semantics[port]
        if semantic.kind is not PrimitiveKind.GROUP:
            depths[port] = 0
            return 0
        visiting.add(port)
        depth = 1 + group_depth(semantic.inputs[0].port, visiting)
        visiting.remove(port)
        depths[port] = depth
        return depth

    for port in logical.ports:
        group_depth(port, set())

    from .logical import freeze_mapping

    return DerivedFacts(
        semantics_by_port=freeze_mapping(semantics),
        consumers_by_port=freeze_mapping(
            {port: tuple(uses) for port, uses in consumers.items()}
        ),
        outputs_by_call=freeze_mapping(
            {
                call: tuple(port for _, port in sorted(outputs))
                for call, outputs in outputs_by_call.items()
            }
        ),
        shape_reporters_by_domain=freeze_mapping(
            {
                domain: tuple(dict.fromkeys(reporters))
                for domain, reporters in shape_reporters.items()
            }
        ),
        control_ports=frozenset(control_ports),
        group_depth_by_port=freeze_mapping(depths),
    )


__all__ = [
    "CallUse",
    "DerivedFacts",
    "LogicalUse",
    "PrimitiveUse",
    "analyze",
]
