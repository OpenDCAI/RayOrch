"""运行时、执行器与 Worker 共用的稳定 DTO；本模块不依赖 Ray。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .model import GrainRef, PortRef


@dataclass(frozen=True, slots=True, order=True)
class BlockRef:
    """粗粒度数据块的逻辑引用；行选择由 RowBinding 单独表达。"""

    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError("BlockRef must be non-negative")


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
    """一个父 Grain 为某个 Expand Port 产生的有序扁平行。"""

    port: PortRef
    rows: tuple[RowBinding, ...]


@dataclass(frozen=True, slots=True)
class OutputReport:
    """一个 Call 输出 Port 的物理实现报告。"""

    port: PortRef
    scalar: RowBinding | None = None
    expansions: tuple[ExpandedRows, ...] = ()
    control: bool | None = None


@dataclass(frozen=True, slots=True)
class CallReport:
    """Worker 对单个 Grain 的原子完成报告。"""

    grain: GrainRef
    generation: int
    outputs: tuple[OutputReport, ...]


@dataclass(frozen=True, slots=True)
class ScalarTake:
    """从一个粗粒度块读取一行标量。"""

    binding: RowBinding


@dataclass(frozen=True, slots=True)
class GroupTake:
    """读取若干行，并按 CSR offsets 重建有序嵌套 group。"""

    bindings: tuple[RowBinding, ...]
    offsets_by_level: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class MissingTake:
    """Optional+DROPPED 输入；Worker 中重建为唯一 MISSING 哨兵。"""


InputTake = ScalarTake | GroupTake | MissingTake


@dataclass(frozen=True, slots=True)
class InvocationPlan:
    """一个 Grain 的 generation 与纯物理输入选择计划。"""

    grain: GrainRef
    generation: int
    inputs: tuple[InputTake, ...]


@dataclass(frozen=True, slots=True)
class RecordFailure:
    """UDF 在 batch 内标记单个 Grain 业务失败的值哨兵。"""

    cause: Any


@dataclass(frozen=True, slots=True)
class CallFailureReport:
    """Worker 对单个 Grain 的 generation-fenced 失败报告。"""

    grain: GrainRef
    generation: int
    cause: Any


WorkerReport = CallReport | CallFailureReport


@dataclass(frozen=True, slots=True)
class OutputLayout:
    """编译器生成的输出物理布局，Worker 无需读取 Program。"""

    port: PortRef
    expanded_ports: tuple[PortRef, ...] = ()
    is_mask: bool = False


__all__ = [
    "BlockRef",
    "CallFailureReport",
    "CallReport",
    "ExpandedRows",
    "GroupTake",
    "InputTake",
    "InvocationPlan",
    "MissingTake",
    "OutputLayout",
    "OutputReport",
    "RecordFailure",
    "RowBinding",
    "ScalarTake",
    "WorkerReport",
]
