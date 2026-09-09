"""运行时、执行器与 Worker 共用的稳定 DTO；本模块不依赖 Ray。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from .model import GrainRef, PortRef


@dataclass(frozen=True, slots=True)
class BlockRef:
    """粗粒度不可变块的透明句柄；行选择由 RowBinding 单独表达。"""
    handle: Any


@dataclass(frozen=True, slots=True)
class RowBinding:
    """数据块内一行的物理定位，不携带业务 payload。"""

    block: BlockRef
    row: int

    def __post_init__(self) -> None:
        if self.row < 0:
            raise ValueError("row must be non-negative")


@dataclass(frozen=True, slots=True)
class ExpandedRows:
    """一个 Expand Port 的有序行及可选逐行控制面副本。"""

    port: PortRef
    rows: tuple[RowBinding, ...]
    controls: tuple[bool, ...] | None = None


@dataclass(frozen=True, slots=True)
class PortOutputReport:
    """One output Port's physical binding report inside a Grain report."""

    port: PortRef
    scalar: RowBinding | None = None
    expansions: tuple[ExpandedRows, ...] = ()
    control: bool | None = None


@dataclass(frozen=True, slots=True)
class GrainReport:
    """Worker 对单个 Grain 的原子完成报告。"""

    grain: GrainRef
    generation: int
    outputs: tuple[PortOutputReport, ...]


@dataclass(frozen=True, slots=True)
class NestedGroupInput:
    """Read rows and reconstruct one ordered nested value group."""

    bindings: tuple[RowBinding, ...]
    offsets_by_level: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class MissingInput:
    """Optional+DROPPED 输入；Worker 中重建为唯一 MISSING 哨兵。"""


GrainInput = RowBinding | NestedGroupInput | MissingInput


@dataclass(frozen=True, slots=True)
class GrainInvocation:
    """One Grain generation and its resolved physical UDF inputs."""

    grain: GrainRef
    generation: int
    inputs: tuple[GrainInput, ...]


@dataclass(frozen=True, slots=True)
class RecordFailure:
    """UDF 在 batch 内标记单个 Grain 业务失败的值哨兵。"""

    cause: Any


@dataclass(frozen=True, slots=True)
class GroupFailure:
    """UDF 标记当前 Grain 失败，并隔离同 Call、同直接父级兄弟。"""

    cause: Any


@dataclass(frozen=True, slots=True)
class GrainFailureReport:
    """Worker 对单个 Grain 的 generation-fenced 失败报告。"""

    grain: GrainRef
    generation: int
    cause: Any
    suppress_siblings: bool = False


WorkerReport = GrainReport | GrainFailureReport


class DispatchFailureKind(Enum):
    """A whole Worker dispatch failed before per-Grain reports existed."""

    UDF_ERROR = auto()
    CONTRACT_ERROR = auto()


@dataclass(frozen=True, slots=True)
class DispatchFailure:
    """Serializable exception snapshot for one whole-dispatch failure."""

    kind: DispatchFailureKind
    error_type: str
    message: str
    traceback: str


WorkerDispatchResult = tuple[WorkerReport, ...] | DispatchFailure


@dataclass(frozen=True, slots=True)
class CallInputLayout:
    """一个 Call 的稳定 positional/keyword Worker 调用布局。"""

    positional_count: int
    keyword_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.positional_count < 0:
            raise ValueError("positional_count must be non-negative")
        if any(not name for name in self.keyword_names):
            raise ValueError("keyword input names must be non-empty")
        if len(set(self.keyword_names)) != len(self.keyword_names):
            raise ValueError("keyword input names must be unique")

    @property
    def input_count(self) -> int:
        """返回布局覆盖的全部逻辑输入槽数。"""

        return self.positional_count + len(self.keyword_names)


@dataclass(frozen=True, slots=True)
class CallOutputLayout:
    """编译器生成的输出布局及 control manifest 需求。"""

    port: PortRef
    expanded_ports: tuple[PortRef, ...] = ()
    control_ports: frozenset[PortRef] = frozenset()


def restore_nested_group(
    leaves: list[Any],
    offsets_by_level: tuple[tuple[int, ...], ...],
) -> list[Any]:
    """由最内层叶子向上应用 CSR offsets，恢复唯一根 nested group。"""

    nodes: list[Any] = leaves
    for offsets in reversed(offsets_by_level):
        nodes = [
            nodes[offsets[index] : offsets[index + 1]]
            for index in range(len(offsets) - 1)
        ]
    if len(nodes) != 1:
        raise ValueError("NestedGroupLayout does not have one root")
    return nodes[0]


__all__ = [
    "BlockRef",
    "GrainFailureReport",
    "GrainReport",
    "DispatchFailure",
    "DispatchFailureKind",
    "ExpandedRows",
    "GroupFailure",
    "NestedGroupInput",
    "CallInputLayout",
    "GrainInput",
    "GrainInvocation",
    "MissingInput",
    "CallOutputLayout",
    "PortOutputReport",
    "RecordFailure",
    "RowBinding",
    "WorkerReport",
    "WorkerDispatchResult",
    "restore_nested_group",
]
