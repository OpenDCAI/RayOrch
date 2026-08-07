"""Multigrain v3.4 的不可变静态 Program、Port provenance 与执行计划。"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, TypeAlias

from .model import CallRef, DomainRef, InputMode, PoolRef, PortRef
from .protocol import OutputLayout


@dataclass(frozen=True, slots=True)
class SourceOrigin:
    """Port 直接来自 Pipeline 的第 ``source_index`` 个输入列。"""

    source_index: int
    name: str


@dataclass(frozen=True, slots=True)
class CallOutputOrigin:
    """Port 来自一个 Call 的指定逻辑输出位。"""

    call: CallRef
    output_index: int


@dataclass(frozen=True, slots=True)
class ExpandOrigin:
    """Port 是对 group Port 展开一级后得到的结构视图。"""

    group_port: PortRef


@dataclass(frozen=True, slots=True)
class GroupOrigin:
    """Port 按 members 的 lineage 把 value 聚合回父 Domain。"""

    value_port: PortRef
    members_port: PortRef


@dataclass(frozen=True, slots=True)
class BroadcastOrigin:
    """Port 把祖先 Domain 的值投影到目标后代 Domain。"""

    source_port: PortRef


@dataclass(frozen=True, slots=True)
class FilterOrigin:
    """Port 用同 Domain 的布尔 mask 保留或丢弃 source Item。"""

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
    """逻辑 Port 的稳定标识、所属 Domain 与唯一来源。"""

    ref: PortRef
    domain: DomainRef
    origin: PortOrigin


@dataclass(frozen=True, slots=True)
class DomainSpec:
    """Entity 集合的逻辑层级；parent 构成一棵有根 Domain 树。"""

    ref: DomainRef
    parent: DomainRef | None = None
    max_fanout: int | None = None
    debug_name: str | None = None

    def __post_init__(self) -> None:
        if self.max_fanout is not None and self.max_fanout < 0:
            raise ValueError("max_fanout must be non-negative")


@dataclass(frozen=True, slots=True)
class KernelSpec:
    """可序列化的 UDF 构造配方，不持有任何运行中实例。"""

    target: Any
    init_args: tuple[Any, ...] = ()
    init_kwargs: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class InputSpec:
    """Call 一个有序输入槽的 Port 绑定与缺失值策略。"""

    name: str
    port: PortRef
    mode: InputMode = InputMode.REQUIRED


@dataclass(frozen=True, slots=True)
class CallSpec:
    """唯一计算边界；输入策略与 driving input 均在编译期冻结。"""

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
class CallConsumer:
    """某个 Port 被指定 Call 输入槽消费的反向索引项。"""

    call: CallRef
    input_index: int


@dataclass(frozen=True, slots=True)
class ViewConsumer:
    """某个 Port 参与结构视图计算的反向索引项。"""

    port: PortRef
    role: str


Consumer: TypeAlias = CallConsumer | ViewConsumer


@dataclass(frozen=True, slots=True)
class PoolSpec:
    """一个 Call 的物理 Worker 池配置。"""

    ref: PoolRef
    call: CallRef
    options: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """纯物理计划；承载 actor pool 配置与 Worker 输出布局。"""

    pools: Mapping[PoolRef, PoolSpec] = field(repr=False)
    call_to_pool: Mapping[CallRef, PoolRef] = field(repr=False)
    output_layouts_by_call: Mapping[CallRef, tuple[OutputLayout, ...]] = field(
        repr=False
    )


@dataclass(frozen=True, slots=True)
class Program:
    """不可变逻辑图；不包含 actor handle、ObjectRef 或运行时状态。"""

    calls: Mapping[CallRef, CallSpec] = field(repr=False)
    ports: Mapping[PortRef, PortSpec] = field(repr=False)
    domains: Mapping[DomainRef, DomainSpec] = field(repr=False)
    source_ports: tuple[PortRef, ...] = ()
    output_tree: object = ()
    consumers_by_port: Mapping[PortRef, tuple[Consumer, ...]] = field(
        default_factory=dict,
        repr=False,
    )
    outputs_by_call: Mapping[CallRef, tuple[PortRef, ...]] = field(
        default_factory=dict,
        repr=False,
    )
    shape_reporters_by_domain: Mapping[DomainRef, tuple[PortRef, ...]] = field(
        default_factory=dict,
        repr=False,
    )
    control_ports: frozenset[PortRef] = frozenset()

    def port(self, ref: PortRef) -> PortSpec:
        """按稳定引用读取 Port 定义。"""

        return self.ports[ref]

    def domain(self, ref: DomainRef) -> DomainSpec:
        """按稳定引用读取 Domain 定义。"""

        return self.domains[ref]

    def call(self, ref: CallRef) -> CallSpec:
        """按稳定引用读取 Call 定义。"""

        return self.calls[ref]


@dataclass(frozen=True, slots=True)
class CompiledProgram:
    """逻辑 Program 与物理 ExecutionPlan 的成对编译结果。"""

    program: Program
    execution: ExecutionPlan


def freeze_mapping(values: Mapping[Any, Any]) -> Mapping[Any, Any]:
    """复制并冻结映射，隔离 builder 的后续可变状态。"""

    return MappingProxyType(dict(values))


__all__ = [
    "BroadcastOrigin",
    "CallConsumer",
    "CallOutputOrigin",
    "CallSpec",
    "CompiledProgram",
    "Consumer",
    "DomainSpec",
    "ExecutionPlan",
    "ExpandOrigin",
    "FilterOrigin",
    "GroupOrigin",
    "InputSpec",
    "KernelSpec",
    "PoolSpec",
    "PortOrigin",
    "PortSpec",
    "Program",
    "SourceOrigin",
    "ViewConsumer",
    "freeze_mapping",
]
