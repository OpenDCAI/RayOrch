"""ArenaEngine、StageExecutor 与 Worker 之间的稳定协议 DTO。

本模块只描述控制 manifest、row selectors、completion/failure 和 detached snapshots，
不拥有状态、不依赖 Ray，也不解释 DAG lineage。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .model import AttemptToken, GrainFailure, GrainId, GroupShape, ItemRef


@dataclass(frozen=True, slots=True)
class RowTake:
    """从当前 RPC 去重后的某个 coarse block 参数中选择一行。"""
    ref_slot: int
    row: int


@dataclass(frozen=True, slots=True)
class ValueTake:
    """一个 scalar 或 GROUP UDF 输入的有序 row selectors。

    GROUP 输入额外携带 GroupShape，Worker 据此在不读取 Driver payload 的情况下恢复
    nested Python list。
    """
    rows: tuple[RowTake, ...]
    group_shape: GroupShape | None = None

    def __post_init__(self) -> None:
        """校验 GroupShape leaf 数量与 RowTake 数量一致。"""

        if self.group_shape is not None and (
            self.group_shape.leaf_count != len(self.rows)
        ):
            raise ValueError("ValueTake shape does not match selected rows")


@dataclass(frozen=True, slots=True)
class MissingTake:
    """显式 OPTIONAL_ONE 缺失；Worker 将其还原为 MISSING sentinel。"""
    pass


InputTake = ValueTake | MissingTake


@dataclass(frozen=True, slots=True)
class Invocation:
    """一个 Logical Grain 到物理 BatchCall 的轻量投影。"""
    token: AttemptToken
    inputs: tuple[InputTake, ...]


@dataclass(frozen=True, slots=True)
class BatchCall:
    """发送给一个 persistent Stage actor 的小型控制 manifest。"""
    dispatch: int
    stage: int
    invocations: tuple[Invocation, ...]


@dataclass(frozen=True, slots=True)
class ValueAck:
    """Map/Expand/Reduce 的单 Grain 输出行数确认。

    `output_counts[i]` 表示该 Grain 在 output Port i 上占用多少行。它不携带业务值，
    业务值位于对应 coarse output block。
    """

    token: AttemptToken
    output_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        """校验所有输出行数均为非负整数。"""

        if any(count < 0 for count in self.output_counts):
            raise ValueError("output counts must be non-negative")


@dataclass(frozen=True, slots=True)
class FilterAck:
    """Filter 的单 Grain bool mask 确认。

    Filter 不返回业务 output block；Arena 根据 `keep` 决定 alias 输入位置或同步发布
    DROPPED。
    """

    token: AttemptToken
    keep: bool


@dataclass(frozen=True, slots=True)
class BatchReport:
    """有界 Worker report；业务值始终位于独立 coarse blocks。

    同一个 report 中 ack 类型必须由 Stage kind 唯一决定：Filter 使用 FilterAck，
    其他可执行原语使用 ValueAck。
    """

    dispatch: int
    acks: tuple[ValueAck | FilterAck, ...]
    column_lengths: tuple[int, ...]
    worker_started_at: float | None = None
    worker_finished_at: float | None = None
    worker_rss_bytes: int | None = None


class FailureKind(Enum):
    """框架刻意保持精简的失败分类。"""
    BAD_RECORD = "bad_record"
    UDF_ERROR = "udf_error"
    CONTRACT_ABORT = "contract_abort"
    INFRA_FAILURE = "infra_failure"


@dataclass(frozen=True, slots=True)
class DispatchFailure:
    """StageExecutor 经 RunDriver 返回 Arena 的结构化失败。"""
    dispatch: int
    kind: FailureKind
    message: str
    bad_token: AttemptToken | None = None
    worker_slot: int | None = None


@dataclass(frozen=True, slots=True)
class DispatchIntent:
    """Arena 生成、交给 StageExecutor 的物理执行请求。

    `input_blocks` 是 opaque coarse block handles；Intent 不暴露 Arena 内部 tables。
    """
    arena_id: int
    call: BatchCall
    input_blocks: tuple[Any, ...]
    actor_policy: str = "any"
    avoid_worker_slot: int | None = None
    flush_reason: str = "full"


@dataclass(frozen=True, slots=True)
class DispatchCompletion:
    """StageExecutor 返回所属 Arena 的成功 completion。"""
    arena_id: int
    call: BatchCall
    report: BatchReport
    output_blocks: tuple[Any, ...]
    worker_slot: int
    submitted_at: float | None = None
    report_received_at: float | None = None


@dataclass(frozen=True, slots=True)
class DispatchTimeline:
    """一个 accepted dispatch 的只读观测时间线，不参与调度决策。"""
    arena: int
    stage: int
    dispatch: int
    worker_slot: int
    grains: int
    flush_reason: str
    submitted_at: float
    report_received_at: float
    committed_at: float
    worker_started_at: float | None
    worker_finished_at: float | None
    worker_rss_bytes: int | None
    status: str


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    """Arena reclaim 后仍保留的 detached source identity。"""
    grain: GrainId
    item: ItemRef


@dataclass(frozen=True, slots=True)
class FailureSnapshot:
    """Arena reclaim 后仍保留的 Failed Grain 快照。"""
    grain: GrainId
    failure: GrainFailure


@dataclass(frozen=True, slots=True)
class SuppressionSnapshot:
    """Arena reclaim 后仍保留的 Suppressed Grain 快照。"""
    grain: GrainId
    direct_causes: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class BlockSlice:
    """RunResult 持有的 final coarse block 行切片，delivery 前无需 driver-side get。"""
    block: Any
    row: int


@dataclass(frozen=True, slots=True)
class ArenaResult:
    """可在 Arena state reclaim 后安全使用的 per-Arena detached delivery。"""
    outputs: tuple[Any, ...]
    failures: tuple[FailureSnapshot, ...]
    suppressions: tuple[SuppressionSnapshot, ...]
    sources: tuple[SourceSnapshot, ...]
    metrics: dict[str, float]
    timeline: tuple[DispatchTimeline, ...] = ()
