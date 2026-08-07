"""Compiler 产出的完整 RuntimePlan 与 logical→physical explain。"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, TypeAlias

from .model import CallRef, DomainRef, PortRef
from .protocol import InputLayout, OutputLayout
from .recovery import DEFAULT_RECOVERY_POLICY, RecoveryPolicy

if TYPE_CHECKING:
    from .analysis import DerivedFacts
    from .logical import CallSpec, DomainSpec, LogicalProgram


@dataclass(frozen=True, slots=True)
class CallInputEffect:
    call: CallRef
    input_index: int


@dataclass(frozen=True, slots=True)
class FilterEffect:
    target_port: PortRef
    source_port: PortRef
    mask_port: PortRef
    copy_source_control: bool


@dataclass(frozen=True, slots=True)
class GroupEffect:
    target_port: PortRef
    value_port: PortRef
    members_port: PortRef
    child_domain: DomainRef
    value_depth: int


@dataclass(frozen=True, slots=True)
class BroadcastEffect:
    target_port: PortRef
    source_port: PortRef
    source_domain: DomainRef
    target_domain: DomainRef
    copy_source_control: bool


StructuralEffect: TypeAlias = FilterEffect | GroupEffect | BroadcastEffect
ItemEffect: TypeAlias = CallInputEffect | StructuralEffect


@dataclass(frozen=True, slots=True)
class ExpansionRule:
    port: PortRef
    child_domain: DomainRef
    control_required: bool


@dataclass(frozen=True, slots=True)
class PoolSpec:
    """一个 Call 唯一的强类型 actor-pool 执行合同。"""

    replicas: int = 1
    batch_size: int = 1
    batch_scope: str = "elastic"
    recovery: RecoveryPolicy = DEFAULT_RECOVERY_POLICY
    ray_options: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if type(self.replicas) is not int or self.replicas <= 0:
            raise ValueError("replicas must be a positive integer")
        if type(self.batch_size) is not int or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(self.batch_scope, str) or self.batch_scope not in {
            "elastic",
            "parent_bound",
        }:
            raise ValueError("batch_scope must be 'elastic' or 'parent_bound'")
        if not isinstance(self.recovery, RecoveryPolicy):
            raise TypeError("recovery must be a RecoveryPolicy")


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
    effects_by_item_port: Mapping[PortRef, tuple[ItemEffect, ...]] = field(
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
    structural_effects: Mapping[PortRef, StructuralEffect] = field(
        repr=False, default_factory=dict
    )
    effects_by_shape_domain: Mapping[DomainRef, tuple[GroupEffect, ...]] = field(
        repr=False, default_factory=dict
    )
    effects_by_entity_domain: Mapping[
        DomainRef, tuple[BroadcastEffect, ...]
    ] = field(
        repr=False, default_factory=dict
    )
    pools_by_call: Mapping[CallRef, PoolSpec] = field(
        repr=False, default_factory=dict
    )
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

    def pool(self, call: CallRef) -> PoolSpec:
        """返回一个 Call 唯一的物理 actor-pool 合同。"""

        return self.pools_by_call[call]


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
    "BroadcastEffect",
    "CallInputEffect",
    "CanonicalRewrite",
    "CompiledProgram",
    "ExpansionRule",
    "ExplainPlan",
    "FilterEffect",
    "GroupEffect",
    "ItemEffect",
    "PoolSpec",
    "PortExplanation",
    "RuntimePlan",
    "StructuralEffect",
    "freeze_mapping",
]
