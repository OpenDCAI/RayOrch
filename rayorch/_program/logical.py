"""Immutable, lightweight logical program for RayOrch.

This module describes only user-declared Calls, Ports, Domains, and provenance.
Reverse indexes, control demand, physical routing, and actor pools do not belong
to LogicalProgram.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, TypeAlias

from .._model import CallRef, DomainRef, InputMode, PortRef


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
class ReduceOrigin:
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
    | ReduceOrigin
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
class UdfSpec:
    target: Any
    init_args: tuple[Any, ...] = ()
    init_kwargs: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class CallInputSpec:
    """Value portion of one positional or keyword logical argument binding."""

    port: PortRef
    mode: InputMode = InputMode.REQUIRED


@dataclass(frozen=True, slots=True)
class CallSpec:
    ref: CallRef
    udf: UdfSpec
    execution_domain: DomainRef
    args: tuple[CallInputSpec, ...] = ()
    kwargs: tuple[tuple[str, CallInputSpec], ...] = ()

    def __post_init__(self) -> None:
        if not self.ordered_inputs:
            raise ValueError("CallSpec requires at least one input")
        names = tuple(name for name, _ in self.kwargs)
        if any(not name for name in names):
            raise ValueError("CallSpec keyword input names must be non-empty")
        if len(set(names)) != len(names):
            raise ValueError("CallSpec keyword input names must be unique")

    @property
    def ordered_inputs(self) -> tuple[CallInputSpec, ...]:
        """Return dense inputs in Python invocation order."""

        return self.args + tuple(input_ for _, input_ in self.kwargs)


@dataclass(frozen=True, slots=True)
class LogicalProgram:
    """Immutable logical graph containing no derived facts."""

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
    """Copy and freeze a mapping to isolate builder and compiler workspaces."""

    return MappingProxyType(dict(values))


__all__ = [
    "BroadcastOrigin",
    "CallOutputOrigin",
    "CallSpec",
    "DomainSpec",
    "ExpandOrigin",
    "FilterOrigin",
    "CallInputSpec",
    "LogicalProgram",
    "PortOrigin",
    "PortSpec",
    "ReduceOrigin",
    "SourceOrigin",
    "UdfSpec",
    "freeze_mapping",
]
