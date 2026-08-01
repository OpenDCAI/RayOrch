"""Hierarchical Reduce 状态与 canonical nested-list shape 构建。

本模块不访问 Arena tables 或 Ray。ArenaEngine 只把 terminal fanout/leaf facts 路由给
ReduceAccumulator，并消费最终 GroupShape。这样一层和任意深度 Reduce 共享同一抽象。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..model import GrainId, GroupShape, InvariantError, ItemRecord, ItemRef


class FanoutTerminal(Enum):
    """一个 parent/Expand occurrence 的终态分类。"""

    SUCCESS = "success"
    DROPPED = "dropped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ExpandInstance:
    """一个 terminal dynamic fanout fact。

    SUCCESS 保存已知 cardinality；DROPPED 表示该 Expand 未执行且中间 node 应省略；
    FAILED 表示 failed-before-output，并携带直接 Grain cause。
    """

    grain: GrainId | None
    stage: int
    anchor: ItemRef
    terminal: FanoutTerminal
    cardinality: int | None = None
    cause: GrainId | None = None

    @classmethod
    def success(
        cls,
        grain: GrainId,
        stage: int,
        anchor: ItemRef,
        cardinality: int,
    ) -> "ExpandInstance":
        """构造已知 cardinality 的成功 fanout fact。"""

        if cardinality < 0:
            raise ValueError("fanout cardinality must be non-negative")
        return cls(grain, stage, anchor, FanoutTerminal.SUCCESS, cardinality)

    @classmethod
    def dropped(
        cls,
        stage: int,
        anchor: ItemRef,
        cause: GrainId | None,
    ) -> "ExpandInstance":
        """构造正常 absence 导致的未执行 fanout fact。"""

        return cls(None, stage, anchor, FanoutTerminal.DROPPED, cause=cause)

    @classmethod
    def failed(
        cls,
        grain: GrainId,
        stage: int,
        anchor: ItemRef,
    ) -> "ExpandInstance":
        """构造 failed-before-output 的 fanout fact。"""

        return cls(grain, stage, anchor, FanoutTerminal.FAILED, cause=grain)


@dataclass(slots=True)
class ReduceAccumulator:
    """一个 anchor 的任意深度 GROUP 累加器。

    `fanouts` 以 `(depth, parent_ordinal_path)` 为 key，`leaf_groups` 以完整 ordinal
    path 为 key。一层 Reduce 只是长度为 1 的 scope_path。

    设计理念：

    - 一个 Expand 固定对应一层 list；
    - intermediate `N=0` 与 normal drop 保持不同语义；
    - 多个 GROUP 共用同一 canonical tree shape；
    - shape 算法保持纯函数式，不读取 Arena tables。
    """

    stage: int
    anchor: ItemRef
    scope_path: tuple[int, ...]
    fanouts: dict[tuple[int, tuple[int, ...]], ExpandInstance] = field(
        default_factory=dict
    )
    leaf_groups: dict[int, dict[tuple[int, ...], ItemRecord]] = field(
        default_factory=dict
    )
    scalar_inputs: dict[int, ItemRef] = field(default_factory=dict)
    slot_cost: int = 0

    def depth_for(self, expand_stage: int) -> int | None:
        """返回 Expand 在 scope_path 中的深度；不属于该 Reduce 时返回 None。"""

        try:
            return self.scope_path.index(expand_stage)
        except ValueError:
            return None

    def settle_fanout(
        self,
        depth: int,
        parent_path: tuple[int, ...],
        instance: ExpandInstance,
    ) -> int:
        """写入一个 fanout fact，并返回新增 metadata slot 成本。"""

        key = (depth, parent_path)
        existing = self.fanouts.get(key)
        if existing is not None:
            if existing != instance:
                raise InvariantError("Reduce fanout terminal changed")
            return 0
        self.fanouts[key] = instance
        added = 1 + (
            instance.cardinality
            if instance.terminal is FanoutTerminal.SUCCESS
            else 0
        )
        self.slot_cost += added
        return added

    def settle_leaf(
        self,
        input_index: int,
        ordinal_path: tuple[int, ...],
        record: ItemRecord,
    ) -> int:
        """写入一个 GROUP leaf receipt，并返回新增 metadata slot 成本。"""

        group = self.leaf_groups.setdefault(input_index, {})
        existing = group.get(ordinal_path)
        if existing is not None:
            if existing != record:
                raise InvariantError("Reduce leaf terminal changed")
            return 0
        group[ordinal_path] = record
        self.slot_cost += 1
        return 1

    def expected_leaf_paths(self) -> set[tuple[int, ...]] | None:
        """返回完整 tree 期望的 active leaf paths；尚未 settled 时返回 None。"""

        structure = self._active_structure()
        return None if structure is None else set(structure[1][-1])

    def ready(
        self,
        *,
        group_indexes: tuple[int, ...],
        scalar_indexes: tuple[int, ...],
    ) -> bool:
        """判断所有 scalar inputs 和 required GROUP leaves 是否已 terminal。"""

        if not all(index in self.scalar_inputs for index in scalar_indexes):
            return False
        expected = self.expected_leaf_paths()
        return expected is not None and all(
            expected.issubset(self.leaf_groups[index])
            for index in group_indexes
        )

    def build_shape(
        self,
        selected: set[tuple[int, ...]] | None = None,
    ) -> tuple[GroupShape, tuple[tuple[int, ...], ...]]:
        """构建 canonical CSR offsets 与 flat leaf path 顺序。

        `selected` 用于 Filter 投影：保留 intermediate tree node，只在最内层省略未选 leaf。
        """

        structure = self._active_structure()
        if structure is None:
            raise InvariantError("Reduce shape requested before fanouts settle")
        offsets_by_level, paths_by_level = structure
        if selected is None:
            return GroupShape(offsets_by_level), tuple(paths_by_level[-1])

        last_depth = len(offsets_by_level) - 1
        parents = [()] if last_depth == 0 else paths_by_level[last_depth - 1]
        offsets = [0]
        selected_leaves: list[tuple[int, ...]] = []
        for parent_path in parents:
            children = [
                path
                for path in paths_by_level[-1]
                if path[:-1] == parent_path and path in selected
            ]
            selected_leaves.extend(children)
            offsets.append(len(selected_leaves))
        return (
            GroupShape((*offsets_by_level[:-1], tuple(offsets))),
            tuple(selected_leaves),
        )

    def _active_structure(
        self,
    ) -> tuple[
        tuple[tuple[int, ...], ...],
        tuple[tuple[tuple[int, ...], ...], ...],
    ] | None:
        """生成 active paths：保留 successful N=0，省略 dropped intermediate node。"""

        parent_paths: list[tuple[int, ...]] = [()]
        offsets_by_level: list[tuple[int, ...]] = []
        paths_by_level: list[tuple[tuple[int, ...], ...]] = []
        last_depth = len(self.scope_path) - 1
        for depth in range(len(self.scope_path)):
            offsets = [0]
            next_paths: list[tuple[int, ...]] = []
            for parent_path in parent_paths:
                instance = self.fanouts.get((depth, parent_path))
                if instance is None:
                    return None
                if instance.terminal is FanoutTerminal.FAILED:
                    return None
                if instance.terminal is FanoutTerminal.DROPPED:
                    offsets.append(len(next_paths))
                    continue
                assert instance.cardinality is not None
                for ordinal in range(instance.cardinality):
                    child_path = (*parent_path, ordinal)
                    if depth < last_depth:
                        child = self.fanouts.get((depth + 1, child_path))
                        if child is None:
                            return None
                        if child.terminal is FanoutTerminal.FAILED:
                            return None
                        if child.terminal is FanoutTerminal.DROPPED:
                            continue
                    next_paths.append(child_path)
                offsets.append(len(next_paths))
            offsets_by_level.append(tuple(offsets))
            paths_by_level.append(tuple(next_paths))
            parent_paths = next_paths
        return tuple(offsets_by_level), tuple(paths_by_level)
