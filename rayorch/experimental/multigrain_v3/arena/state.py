"""Arena 内部的轻量被动状态记录。

这些 dataclass 不包含编排算法或 Ray handle。把 queue、recovery 和 lease records 独立
出来，是为了让 ArenaEngine 聚焦状态转换，同时避免 mixin 或第二 runtime authority。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from ..contracts import ExecutionError
from ..dag import RecoveryPreset
from ..model import EntityId, GrainId, ItemRef
from ..protocol import BatchCall


class ArenaAbort(ExecutionError):
    """表示 run-control 或 contract 错误导致一个 bounded Arena 中止。"""


@dataclass(frozen=True, slots=True)
class ArenaLimits:
    """Arena 在发布 metadata 前必须检查的硬边界。

    Overflow 属于控制面 abort，不能伪装成某个用户 Grain 的 Failed outcome。
    """

    max_grains: int = 100_000
    max_fanout_per_grain: int = 100_000
    max_reduce_slots: int = 1_000_000
    max_pending_dispatches: int = 256
    max_blocks: int = 100_000

    def __post_init__(self) -> None:
        """校验所有 Arena hard limit 为正数。"""

        if min(
            self.max_grains,
            self.max_fanout_per_grain,
            self.max_reduce_slots,
            self.max_pending_dispatches,
            self.max_blocks,
        ) <= 0:
            raise ValueError("Arena limits must be positive")


@dataclass(slots=True)
class PendingInvocation:
    """等待 aligned inputs 全部 terminal 的短命 fan-in 状态。

    它只保存 ItemRef slots，不复制 ItemRecord；分类完成后立即删除。
    """

    stage: int
    entity: EntityId
    inputs: list[ItemRef | None]


@dataclass(slots=True)
class StageBatchQueue:
    """一个 Arena/Stage 组合拥有的 normal 与 recovery queues。"""

    normal: deque[GrainId] = field(default_factory=deque)
    normal_set: set[GrainId] = field(default_factory=set)
    immediate: deque["RecoveryTask"] = field(default_factory=deque)
    tail: deque["RecoveryTask"] = field(default_factory=deque)
    first_wait_at: float | None = None


@dataclass(slots=True)
class RecoveryBudgetState:
    """一个 recovery split tree 的所有子任务共享的预算计数。"""

    extra_rpcs: int = 0
    reexecuted_grains: int = 0


@dataclass(slots=True)
class RecoveryTask:
    """一组 deferred/immediate 物理重执行 Grain。

    RecoveryTask 不进入 GrainRecord，也不形成持久 recovery graph。
    """

    stage: int
    grains: tuple[GrainId, ...]
    preset: RecoveryPreset
    attempts: int = 0
    depth: int = 0
    budget: RecoveryBudgetState = field(default_factory=RecoveryBudgetState)
    actor_policy: str = "any"
    avoid_worker_slot: int | None = None
    phase: str = "retry"


@dataclass(slots=True)
class DispatchLease:
    """Arena 对一个 pending physical dispatch 的 commit authority。

    Lease 保存 GrainIds、输入 block ids 和 recovery context；Ray ObjectRefs 由 StageExecutor
    的 PendingRPC 单独拥有。
    """

    call: BatchCall
    block_ids: tuple[int, ...]
    grain_ids: tuple[GrainId, ...]
    recovery: RecoveryTask | None = None
    flush_reason: str = "full"
