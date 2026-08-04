"""被动运行时记录与表；不包含调度策略，也不依赖 Ray。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

from ..model import (
    DomainRef,
    EntityRef,
    GrainOutcome,
    GrainPhase,
    GrainRef,
    ItemOutcome,
    ItemRef,
    PortRef,
    ShapeState,
)
from ..protocol import RowBinding


class CommitError(RuntimeError):
    """提交报告与已经发布的不可变运行时事实冲突。"""


@dataclass(frozen=True, slots=True)
class ShapeKey:
    """一个父 Entity 在指定 child Domain 上的唯一 fan-out 身份。"""

    child_domain: DomainRef
    parent_entity: EntityRef


@dataclass(frozen=True, slots=True)
class EntityOrigin:
    """子 Entity 的显式父引用、稳定 ordinal 与来源 Shape。"""

    domain: DomainRef
    parent_entity: EntityRef
    ordinal: int
    shape_key: ShapeKey


@dataclass(frozen=True, slots=True)
class ItemRecord:
    """Item 的终态与可选因果；业务 payload 单独存放。"""

    outcome: ItemOutcome
    cause: object | None = None


@dataclass(slots=True)
class GrainRecord:
    """Grain 的输入快照、阶段、终态与 retry generation。"""

    ref: GrainRef
    inputs: tuple[ItemRef, ...]
    phase: GrainPhase
    outcome: GrainOutcome | None = None
    generation: int = 0
    active_attempt: int | None = None
    infra_failures: int = 0


@dataclass(slots=True)
class ShapeRecord:
    """多输出共同报告的 cardinality 屏障及其终态。"""

    state: ShapeState
    cardinality: int | None
    expected_reporters: tuple[PortRef, ...]
    received_reports: dict[PortRef, int] = field(default_factory=dict)
    cause: object | None = None


@dataclass(frozen=True, slots=True)
class GroupShape:
    """从一个隐式根到有序叶子的 CSR 风格层级 offsets。"""

    offsets_by_level: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not self.offsets_by_level:
            raise ValueError("GroupShape requires at least one level")
        for offsets in self.offsets_by_level:
            if not offsets or offsets[0] != 0:
                raise ValueError("every GroupShape level must start at zero")
            if any(left > right for left, right in zip(offsets, offsets[1:])):
                raise ValueError("GroupShape offsets must be monotonic")

    @property
    def depth(self) -> int:
        """返回从隐式根到叶子的结构层数。"""

        return len(self.offsets_by_level)

    @classmethod
    def one_level(cls, count: int) -> GroupShape:
        """构造包含 ``count`` 个有序叶子的一层 group。"""

        if count < 0:
            raise ValueError("group count must be non-negative")
        return cls(((0, count),))

    @classmethod
    def nest(cls, children: tuple[GroupShape, ...], *, child_depth: int) -> GroupShape:
        """把同深度 child shapes 拼成更高一级；空 group 也保留完整层数。"""

        if child_depth <= 0:
            raise ValueError("child_depth must be positive")
        if any(child.depth != child_depth for child in children):
            raise ValueError("nested children must have one canonical depth")

        top = (0, len(children))
        if not children:
            return cls((top,) + tuple((0,) for _ in range(child_depth)))

        merged: list[tuple[int, ...]] = [top]
        for level in range(child_depth):
            offsets = [0]
            total = 0
            for child in children:
                current = child.offsets_by_level[level]
                for left, right in zip(current, current[1:]):
                    total += right - left
                    offsets.append(total)
            merged.append(tuple(offsets))
        return cls(tuple(merged))


@dataclass(frozen=True, slots=True)
class GroupBinding:
    """规范化层级 Shape 与扁平叶子 Item 的纯引用绑定。"""

    shape: GroupShape
    flat_items: tuple[ItemRef, ...]

    def __post_init__(self) -> None:
        if self.shape.offsets_by_level[-1][-1] != len(self.flat_items):
            raise ValueError("GroupShape leaf count does not match flat_items")


ValueBinding: TypeAlias = RowBinding | GroupBinding


@dataclass(frozen=True, slots=True)
class ArenaLimits:
    """单个 Arena 的硬资源上限，防止异常 fan-out 无界膨胀。"""

    max_entities: int = 1_000_000
    max_items: int = 10_000_000
    max_grains: int = 10_000_000
    max_group_slots: int = 10_000_000

    def __post_init__(self) -> None:
        if min(
            self.max_entities,
            self.max_items,
            self.max_grains,
            self.max_group_slots,
        ) <= 0:
            raise ValueError("Arena limits must be positive")


@dataclass(slots=True)
class PendingInvocation:
    """尚未凑齐的 Call 输入槽；槽位顺序与 CallSpec 一致。"""

    slots: list[ItemRef | None]


@dataclass(slots=True)
class RuntimeState:
    """ArenaEngine 独占写入的全部被动语义表。"""

    grains: dict[GrainRef, GrainRecord] = field(default_factory=dict)
    items: dict[ItemRef, ItemRecord] = field(default_factory=dict)
    shapes: dict[ShapeKey, ShapeRecord] = field(default_factory=dict)
    entity_lineage: dict[EntityRef, EntityOrigin] = field(default_factory=dict)
    values: dict[ItemRef, ValueBinding] = field(default_factory=dict)
    pending: dict[GrainRef, PendingInvocation] = field(default_factory=dict)


__all__ = [
    "ArenaLimits",
    "CommitError",
    "EntityOrigin",
    "GrainRecord",
    "GroupBinding",
    "GroupShape",
    "ItemRecord",
    "PendingInvocation",
    "RuntimeState",
    "ShapeKey",
    "ShapeRecord",
    "ValueBinding",
]
