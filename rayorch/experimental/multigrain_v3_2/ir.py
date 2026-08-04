"""Multigrain V3.2 的 Port-first 逻辑 IR。

逻辑图刻意分离 value Port、entity Domain 与
compute Node。Grain transform 创建新 Port，而不修改
producer node。本模块不导入 Arena、Ray 或 V3 物理
StageSpec 模型。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, TypeAlias


class LogicalCompileError(ValueError):
    """authoring graph 违反 Port/domain 不变量。"""


@dataclass(frozen=True, slots=True, order=True)
class DomainId:
    """由对齐到同一逻辑 entity 的值共享的身份域。"""

    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError("DomainId must be non-negative")


@dataclass(frozen=True, slots=True, order=True)
class ExpansionId:
    """一条显式的 parent-domain 到 child-domain cardinality 关系。"""

    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError("ExpansionId must be non-negative")


@dataclass(frozen=True, slots=True, order=True)
class LogicalPortId:
    """逻辑 node 的输出坐标。"""

    node: int
    output: int

    def __post_init__(self) -> None:
        if self.node < 0 or self.output < 0:
            raise ValueError("LogicalPortId fields must be non-negative")


@dataclass(frozen=True, slots=True)
class DirectValue:
    """在 Port 自身 entity domain 中直接产生的值。"""


@dataclass(frozen=True, slots=True)
class GroupValue:
    """由一个 leaf Port 派生的 parent-domain 有序 group。"""

    leaf: LogicalPortId
    expansion_path: tuple[ExpansionId, ...]

    def __post_init__(self) -> None:
        if not self.expansion_path:
            raise ValueError("GroupValue requires at least one expansion")


ValueLayout: TypeAlias = DirectValue | GroupValue


@dataclass(frozen=True, slots=True)
class PortSpec:
    """带有显式 entity domain 与 value layout 的逻辑 Port。"""

    id: LogicalPortId
    domain: DomainId
    layout: ValueLayout = DirectValue()


@dataclass(frozen=True, slots=True)
class DomainSpec:
    """根或子 domain；每个子 domain 恰有一个 expansion parent。"""

    id: DomainId
    parent: DomainId | None = None
    via_expansion: ExpansionId | None = None

    def __post_init__(self) -> None:
        if (self.parent is None) != (self.via_expansion is None):
            raise ValueError("parent and via_expansion must appear together")


@dataclass(frozen=True, slots=True)
class ExpansionSpec:
    """由 Port 而非 RayModule 拥有的显式 aligned fan-out 关系。"""

    id: ExpansionId
    parent_domain: DomainId
    child_domain: DomainId
    group_inputs: tuple[LogicalPortId, ...]

    def __post_init__(self) -> None:
        if not self.group_inputs:
            raise ValueError("ExpansionSpec requires at least one group input")


@dataclass(frozen=True, slots=True)
class ModuleSpec:
    """声明式持久 actor/UDF 配方。"""

    target: Any
    init_args: tuple[Any, ...] = ()
    init_kwargs: tuple[tuple[str, Any], ...] = ()
    options: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class CallInput:
    """一条具名 RayModule 输入边。"""

    name: str
    port: LogicalPortId
    optional: bool = False


@dataclass(frozen=True, slots=True)
class SourceNode:
    id: int
    output: LogicalPortId
    parameter: str


@dataclass(frozen=True, slots=True)
class CallNode:
    id: int
    module: ModuleSpec
    inputs: tuple[CallInput, ...]
    outputs: tuple[LogicalPortId, ...]


@dataclass(frozen=True, slots=True)
class ExpandNode:
    """Port 级 aligned expansion；输入仍是有效的 parent-domain Port。"""

    id: int
    expansion: ExpansionId
    inputs: tuple[LogicalPortId, ...]
    outputs: tuple[LogicalPortId, ...]


@dataclass(frozen=True, slots=True)
class ReduceNode:
    """Port 级 aligned grouping，返回最近的 parent domain。"""

    id: int
    expansion: ExpansionId
    inputs: tuple[LogicalPortId, ...]
    outputs: tuple[LogicalPortId, ...]


LogicalNode: TypeAlias = SourceNode | CallNode | ExpandNode | ReduceNode


@dataclass(frozen=True, slots=True)
class ConsumerEdge:
    node: int
    input_index: int


@dataclass(frozen=True, slots=True)
class LogicalDAG:
    """不可变的 Port-first authoring 结果。"""

    nodes: tuple[LogicalNode, ...]
    ports: Mapping[LogicalPortId, PortSpec] = field(repr=False)
    domains: Mapping[DomainId, DomainSpec] = field(repr=False)
    expansions: Mapping[ExpansionId, ExpansionSpec] = field(repr=False)
    consumers_by_port: Mapping[LogicalPortId, tuple[ConsumerEdge, ...]] = field(
        repr=False
    )
    source_ports: tuple[LogicalPortId, ...] = ()
    output_ports: tuple[LogicalPortId, ...] = ()

    def port(self, port: LogicalPortId) -> PortSpec:
        try:
            return self.ports[port]
        except KeyError as error:
            raise LogicalCompileError(f"unknown logical Port: {port}") from error

    def domain(self, domain: DomainId) -> DomainSpec:
        try:
            return self.domains[domain]
        except KeyError as error:
            raise LogicalCompileError(f"unknown Domain: {domain}") from error


def freeze_mapping(values: dict[Any, Any]) -> Mapping[Any, Any]:
    """把 compiler 拥有的字典暴露为不可变 mapping。"""

    return MappingProxyType(dict(values))
