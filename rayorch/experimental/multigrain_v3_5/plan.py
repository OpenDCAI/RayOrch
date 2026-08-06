"""Compiler 产出的完整 RuntimePlan 与 logical→physical explain。"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, TypeAlias

from .model import CallRef, DomainRef, PoolRef, PortRef
from .protocol import InputLayout, OutputLayout

if TYPE_CHECKING:
    from .analysis import DerivedFacts
    from .logical import CallSpec, DomainSpec, LogicalProgram


@dataclass(frozen=True, slots=True)
class CallInputRoute:
    call: CallRef
    input_index: int


@dataclass(frozen=True, slots=True)
class FilterRoute:
    port: PortRef


@dataclass(frozen=True, slots=True)
class GroupRoute:
    port: PortRef


@dataclass(frozen=True, slots=True)
class BroadcastRoute:
    port: PortRef


RuntimeRoute: TypeAlias = CallInputRoute | FilterRoute | GroupRoute | BroadcastRoute


@dataclass(frozen=True, slots=True)
class ExpansionRule:
    port: PortRef
    child_domain: DomainRef
    control_required: bool


@dataclass(frozen=True, slots=True)
class FilterRule:
    source_port: PortRef
    mask_port: PortRef
    copy_source_control: bool


@dataclass(frozen=True, slots=True)
class GroupRule:
    value_port: PortRef
    members_port: PortRef
    child_domain: DomainRef
    value_depth: int


@dataclass(frozen=True, slots=True)
class BroadcastRule:
    source_port: PortRef
    source_domain: DomainRef
    target_domain: DomainRef
    copy_source_control: bool


@dataclass(frozen=True, slots=True)
class PoolSpec:
    ref: PoolRef
    call: CallRef
    options: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class CanonicalRewrite:
    kind: str
    port: PortRef
    before: tuple[PortRef, ...]
    after: tuple[PortRef, ...]


@dataclass(frozen=True, slots=True)
class PortExplanation:
    port: PortRef
    logical_kind: str
    domain: DomainRef
    logical_inputs: tuple[PortRef, ...]
    physical_rule: str
    control_required: bool
    rewrites: tuple[CanonicalRewrite, ...] = ()


@dataclass(frozen=True, slots=True)
class ExplainPlan:
    optimized: bool
    ports: Mapping[PortRef, PortExplanation] = field(repr=False)
    rewrites: tuple[CanonicalRewrite, ...] = ()

    def format(self) -> str:
        """返回稳定、适合测试和诊断的逐 Port 映射。"""

        mode = "optimized" if self.optimized else "unoptimized"
        lines = [f"RuntimePlan[{mode}]"]
        for port in sorted(self.ports, key=lambda ref: ref.value):
            item = self.ports[port]
            inputs = ",".join(str(ref.value) for ref in item.logical_inputs) or "-"
            control = " control" if item.control_required else ""
            lines.append(
                f"  p{port.value} {item.logical_kind} d{item.domain.value} "
                f"<- [{inputs}] => {item.physical_rule}{control}"
            )
            for rewrite in item.rewrites:
                before = ",".join(f"p{ref.value}" for ref in rewrite.before)
                after = ",".join(f"p{ref.value}" for ref in rewrite.after)
                lines.append(f"    rewrite {rewrite.kind}: [{before}] -> [{after}]")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class RuntimePlan:
    """Arena、Executor、Worker 调度所需的全部静态事实。

    RuntimePlan 不含 PortOrigin，运行时无需导入或解释逻辑 provenance。
    """

    calls: Mapping[CallRef, CallSpec] = field(repr=False)
    domains: Mapping[DomainRef, DomainSpec] = field(repr=False)
    port_domains: Mapping[PortRef, DomainRef] = field(repr=False)
    source_ports: tuple[PortRef, ...] = ()
    output_tree: object = ()
    routes_by_port: Mapping[PortRef, tuple[RuntimeRoute, ...]] = field(
        repr=False, default_factory=dict
    )
    outputs_by_call: Mapping[CallRef, tuple[PortRef, ...]] = field(
        repr=False, default_factory=dict
    )
    expansions_by_source: Mapping[PortRef, tuple[ExpansionRule, ...]] = field(
        repr=False, default_factory=dict
    )
    shape_reporters_by_domain: Mapping[DomainRef, tuple[PortRef, ...]] = field(
        repr=False, default_factory=dict
    )
    control_ports: frozenset[PortRef] = frozenset()
    filter_rules: Mapping[PortRef, FilterRule] = field(repr=False, default_factory=dict)
    group_rules: Mapping[PortRef, GroupRule] = field(repr=False, default_factory=dict)
    broadcast_rules: Mapping[PortRef, BroadcastRule] = field(
        repr=False, default_factory=dict
    )
    groups_by_child_domain: Mapping[DomainRef, tuple[PortRef, ...]] = field(
        repr=False, default_factory=dict
    )
    broadcasts_by_target_domain: Mapping[DomainRef, tuple[PortRef, ...]] = field(
        repr=False, default_factory=dict
    )
    pools: Mapping[PoolRef, PoolSpec] = field(repr=False, default_factory=dict)
    call_to_pool: Mapping[CallRef, PoolRef] = field(repr=False, default_factory=dict)
    output_layouts_by_call: Mapping[CallRef, tuple[OutputLayout, ...]] = field(
        repr=False, default_factory=dict
    )
    input_layouts_by_call: Mapping[CallRef, InputLayout] = field(
        repr=False, default_factory=dict
    )

    def call(self, ref: CallRef) -> CallSpec:
        return self.calls[ref]

    def domain(self, ref: DomainRef) -> DomainSpec:
        return self.domains[ref]

    def port_domain(self, ref: PortRef) -> DomainRef:
        return self.port_domains[ref]


@dataclass(frozen=True, slots=True)
class CompiledProgram:
    logical: LogicalProgram
    facts: DerivedFacts
    runtime: RuntimePlan
    explain: ExplainPlan

    def explain_text(self) -> str:
        return self.explain.format()


def freeze_mapping(values: Mapping[Any, Any]) -> Mapping[Any, Any]:
    return MappingProxyType(dict(values))


__all__ = [
    "BroadcastRoute",
    "BroadcastRule",
    "CallInputRoute",
    "CanonicalRewrite",
    "CompiledProgram",
    "ExpansionRule",
    "ExplainPlan",
    "FilterRoute",
    "FilterRule",
    "GroupRoute",
    "GroupRule",
    "PoolSpec",
    "PortExplanation",
    "RuntimePlan",
    "RuntimeRoute",
    "freeze_mapping",
]
