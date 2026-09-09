"""Compiler 产出的完整 RuntimePlan 与 logical→physical explanation。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Mapping, TypeAlias

from ..model import CallRef, DomainRef, PortRef
from ..protocol import CallInputLayout, CallOutputLayout
from ..recovery import DEFAULT_RECOVERY_POLICY, RecoveryPolicy

if TYPE_CHECKING:
    from .analysis import ProgramAnalysis
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
class ReduceEffect:
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


StructuralEffect: TypeAlias = FilterEffect | ReduceEffect | BroadcastEffect
ItemEffect: TypeAlias = CallInputEffect | StructuralEffect


@dataclass(frozen=True, slots=True)
class ExpandEffect:
    port: PortRef
    child_domain: DomainRef
    control_required: bool


@dataclass(frozen=True, slots=True)
class ActorPoolSpec:
    """One Call's actor-pool and physical dispatch-packing contract.

    ``any_parent`` packs READY Grains from any parent anchor in the Call.
    ``single_parent`` restricts each DispatchBatch to one parent anchor.
    This option changes only RPC packing, never lineage or failure scope.
    """

    replicas: int = 1
    batch_size: int = 1
    batching_policy: Literal["any_parent", "single_parent"] = "any_parent"
    recovery: RecoveryPolicy = DEFAULT_RECOVERY_POLICY
    ray_options: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if type(self.replicas) is not int or self.replicas <= 0:
            raise ValueError("replicas must be a positive integer")
        if type(self.batch_size) is not int or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(self.batching_policy, str) or self.batching_policy not in {
            "any_parent",
            "single_parent",
        }:
            raise ValueError("batching_policy must be 'any_parent' or 'single_parent'")
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
class ProgramExplanation:
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
    """MicrobatchEngine、Executor、Worker 所需的全部静态执行接线。

    RuntimePlan 不含 PortOrigin，运行时无需导入或解释逻辑 provenance。
    """

    calls: Mapping[CallRef, CallSpec] = field(repr=False)
    domains: Mapping[DomainRef, DomainSpec] = field(repr=False)
    port_domains: Mapping[PortRef, DomainRef] = field(repr=False)
    source_ports: tuple[PortRef, ...] = ()
    output_tree: object = ()
    item_effects_by_source: Mapping[PortRef, tuple[ItemEffect, ...]] = field(
        repr=False, default_factory=dict
    )
    outputs_by_call: Mapping[CallRef, tuple[PortRef, ...]] = field(
        repr=False, default_factory=dict
    )
    expand_effects_by_source: Mapping[PortRef, tuple[ExpandEffect, ...]] = field(
        repr=False, default_factory=dict
    )
    expansion_sources_by_domain: Mapping[DomainRef, tuple[PortRef, ...]] = field(
        repr=False, default_factory=dict
    )
    control_ports: frozenset[PortRef] = frozenset()
    structural_effects_by_target: Mapping[PortRef, StructuralEffect] = field(
        repr=False, default_factory=dict
    )
    reduce_effects_by_child_domain: Mapping[DomainRef, tuple[ReduceEffect, ...]] = field(
        repr=False, default_factory=dict
    )
    broadcast_effects_by_target_domain: Mapping[
        DomainRef, tuple[BroadcastEffect, ...]
    ] = field(
        repr=False, default_factory=dict
    )
    actor_pools_by_call: Mapping[CallRef, ActorPoolSpec] = field(
        repr=False, default_factory=dict
    )
    output_layouts_by_call: Mapping[CallRef, tuple[CallOutputLayout, ...]] = field(
        repr=False, default_factory=dict
    )
    input_layouts_by_call: Mapping[CallRef, CallInputLayout] = field(
        repr=False, default_factory=dict
    )

    def call(self, ref: CallRef) -> CallSpec:
        return self.calls[ref]

    def domain(self, ref: DomainRef) -> DomainSpec:
        return self.domains[ref]

    def port_domain(self, ref: PortRef) -> DomainRef:
        return self.port_domains[ref]

    def pool(self, call: CallRef) -> ActorPoolSpec:
        """返回一个 Call 唯一的物理 actor-pool 合同。"""

        return self.actor_pools_by_call[call]


@dataclass(frozen=True, slots=True)
class CompiledProgram:
    logical: LogicalProgram
    analysis: ProgramAnalysis
    plan: RuntimePlan
    explanation: ProgramExplanation

    def explain_text(self) -> str:
        return self.explanation.format()


__all__ = [
    "BroadcastEffect",
    "CallInputEffect",
    "CanonicalRewrite",
    "CompiledProgram",
    "ExpandEffect",
    "ProgramExplanation",
    "FilterEffect",
    "ReduceEffect",
    "ItemEffect",
    "ActorPoolSpec",
    "PortExplanation",
    "RuntimePlan",
    "StructuralEffect",
]
