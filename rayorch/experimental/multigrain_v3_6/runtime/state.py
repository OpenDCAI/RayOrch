"""被动运行时记录与表；不包含调度策略，也不依赖 Ray。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

from ..model import (
    DomainRef,
    EntityRef,
    GrainRef,
    ItemOutcome,
    ItemRef,
    ExpansionOutcome,
)
from ..protocol import RowBinding


class CommitError(RuntimeError):
    """提交报告与已经发布的不可变运行时事实冲突。"""


@dataclass(frozen=True, slots=True)
class ExpansionRef:
    """一个父 Entity 在指定 child Domain 上的唯一 fan-out 身份。"""

    child_domain: DomainRef
    parent_entity: EntityRef


@dataclass(frozen=True, slots=True)
class EntityParent:
    """子 Entity 的显式父引用与稳定 ordinal。"""

    parent_entity: EntityRef
    ordinal: int


@dataclass(frozen=True, slots=True)
class ItemRecord:
    """一个 Item 的不可变语义事实。

    ``outcome`` 描述成员/计算终态，``cause`` 记录非正常终态的来源；
    ``control`` 是调度所需的小型控制面副本（当前为 Filter 的布尔值）。
    业务 payload 仍只存在 ``RuntimeState.values``，MicrobatchEngine 无需读取 payload
    就能恢复并继续结构传播。
    """

    outcome: ItemOutcome
    cause: object | None = None
    control: bool | None = None


@dataclass(frozen=True, slots=True)
class ExpansionRecord:
    """一次 expansion 的终态及其唯一有序 children 事实。"""

    outcome: ExpansionOutcome
    children: tuple[EntityRef, ...] | None
    cause: object | None = None

    def __post_init__(self) -> None:
        if (self.outcome is ExpansionOutcome.SUCCEEDED) != (self.children is not None):
            raise ValueError("only SUCCEEDED Expansion may contain children")

    @property
    def cardinality(self) -> int | None:
        return None if self.children is None else len(self.children)


@dataclass(frozen=True, slots=True)
class NestedGroupLayout:
    """从一个隐式根到有序叶子的 CSR 风格层级 offsets。"""

    offsets_by_level: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not self.offsets_by_level:
            raise ValueError("NestedGroupLayout requires at least one level")
        for offsets in self.offsets_by_level:
            if not offsets or offsets[0] != 0:
                raise ValueError("every NestedGroupLayout level must start at zero")
            if any(left > right for left, right in zip(offsets, offsets[1:])):
                raise ValueError("NestedGroupLayout offsets must be monotonic")

    @property
    def depth(self) -> int:
        """返回从隐式根到叶子的结构层数。"""

        return len(self.offsets_by_level)

    @classmethod
    def one_level(cls, count: int) -> NestedGroupLayout:
        """构造包含 ``count`` 个有序叶子的一层 group。"""

        if count < 0:
            raise ValueError("group count must be non-negative")
        return cls(((0, count),))

    @classmethod
    def nest(
        cls,
        children: tuple[NestedGroupLayout, ...],
        *,
        child_depth: int,
    ) -> NestedGroupLayout:
        """把同深度 child layouts 拼成更高一级；空 group 也保留完整层数。"""

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
class NestedGroupBinding:
    """规范化层级 layout 与扁平叶子 Item 的纯引用绑定。"""

    layout: NestedGroupLayout
    flat_items: tuple[ItemRef, ...]

    def __post_init__(self) -> None:
        if self.layout.offsets_by_level[-1][-1] != len(self.flat_items):
            raise ValueError("NestedGroupLayout leaf count does not match flat_items")


ValueBinding: TypeAlias = RowBinding | NestedGroupBinding


@dataclass(slots=True)
class PendingGrain:
    """尚未凑齐的 Call 输入槽；槽位顺序与 CallSpec 一致。"""

    slots: list[ItemRef | None]


@dataclass(slots=True)
class RuntimeState:
    """MicrobatchEngine 独占写入的全部被动语义表。"""

    items: dict[ItemRef, ItemRecord] = field(default_factory=dict)
    expansions: dict[ExpansionRef, ExpansionRecord] = field(default_factory=dict)
    entity_lineage: dict[EntityRef, EntityParent] = field(default_factory=dict)
    values: dict[ItemRef, ValueBinding] = field(default_factory=dict)
    pending_grains: dict[GrainRef, PendingGrain] = field(default_factory=dict)


__all__ = [
    "CommitError",
    "EntityParent",
    "NestedGroupBinding",
    "NestedGroupLayout",
    "ItemRecord",
    "PendingGrain",
    "RuntimeState",
    "ExpansionRef",
    "ExpansionRecord",
    "ValueBinding",
]
