"""v3.5 的纯逻辑 IR。

本模块只描述用户声明的 Call、Port、Domain 与 provenance。反向索引、control
demand、物理路由和 actor pool 都不属于 LogicalProgram。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, TypeAlias

from .model import CallRef, DomainRef, InputMode, PortRef


@dataclass(frozen=True, slots=True)
class SourceOrigin:
    source_index: int
    name: str


@dataclass(frozen=True, slots=True)
class CallOutputOrigin:
    call: CallRef
    output_index: int


@dataclass(frozen=True, slots=True)
class ExpandOrigin:
    group_port: PortRef


@dataclass(frozen=True, slots=True)
class GroupOrigin:
    value_port: PortRef
    members_port: PortRef


@dataclass(frozen=True, slots=True)
class BroadcastOrigin:
    source_port: PortRef


@dataclass(frozen=True, slots=True)
class FilterOrigin:
    source_port: PortRef
    mask_port: PortRef


PortOrigin: TypeAlias = (
    SourceOrigin
    | CallOutputOrigin
    | ExpandOrigin
    | GroupOrigin
    | BroadcastOrigin
    | FilterOrigin
)


@dataclass(frozen=True, slots=True)
class PortSpec:
    ref: PortRef
    domain: DomainRef
    origin: PortOrigin


@dataclass(frozen=True, slots=True)
class DomainSpec:
    ref: DomainRef
    parent: DomainRef | None = None
    debug_name: str | None = None


@dataclass(frozen=True, slots=True)
class KernelSpec:
    target: Any
    init_args: tuple[Any, ...] = ()
    init_kwargs: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class InputSpec:
    name: str
    port: PortRef
    mode: InputMode = InputMode.REQUIRED
    keyword: bool = False


@dataclass(frozen=True, slots=True)
class CallSpec:
    ref: CallRef
    kernel: KernelSpec
    execution_domain: DomainRef
    inputs: tuple[InputSpec, ...]
    driving_input: int

    def __post_init__(self) -> None:
        if not self.inputs:
            raise ValueError("CallSpec requires at least one input")
        if not 0 <= self.driving_input < len(self.inputs):
            raise ValueError("driving_input is outside CallSpec.inputs")
        if self.inputs[self.driving_input].mode is not InputMode.REQUIRED:
            raise ValueError("driving input must be REQUIRED")


@dataclass(frozen=True, slots=True)
class LogicalProgram:
    """不可变逻辑图；字段中不允许混入任何 derived fact。"""

    calls: Mapping[CallRef, CallSpec] = field(repr=False)
    ports: Mapping[PortRef, PortSpec] = field(repr=False)
    domains: Mapping[DomainRef, DomainSpec] = field(repr=False)
    source_ports: tuple[PortRef, ...] = ()
    output_tree: object = ()

    def port(self, ref: PortRef) -> PortSpec:
        return self.ports[ref]

    def domain(self, ref: DomainRef) -> DomainSpec:
        return self.domains[ref]

    def call(self, ref: CallRef) -> CallSpec:
        return self.calls[ref]


def freeze_mapping(values: Mapping[Any, Any]) -> Mapping[Any, Any]:
    """复制并冻结映射，隔离 builder 和 compiler 的可变工作区。"""

    return MappingProxyType(dict(values))


__all__ = [
    "BroadcastOrigin",
    "CallOutputOrigin",
    "CallSpec",
    "DomainSpec",
    "ExpandOrigin",
    "FilterOrigin",
    "GroupOrigin",
    "InputSpec",
    "KernelSpec",
    "LogicalProgram",
    "PortOrigin",
    "PortSpec",
    "SourceOrigin",
    "freeze_mapping",
]
