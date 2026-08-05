"""Local/Ray 执行器共用的输出物化逻辑。

该模块只通过 :class:`ArenaEngine` 的只读公开接口访问语义状态；执行器不读取
Arena 内部 table，也不解释 PortOrigin。
"""

from __future__ import annotations

from typing import Any, Protocol

from .model import ItemOutcome, ItemRef, PortRef
from .program import Program
from .protocol import RowBinding, restore_group
from .runtime import ArenaEngine, GroupBinding


class ReadableStore(Protocol):
    """结果物化所需的最小块读取接口。"""

    def get(self, binding: RowBinding) -> Any:
        """读取一个物理行绑定对应的业务值。"""

        ...


def materialize_tree(
    program: Program,
    arena: ArenaEngine,
    store: ReadableStore,
    tree: object | None = None,
) -> object:
    """按稳定 lineage 顺序物化 Program 输出树。"""

    current = program.output_tree if tree is None else tree
    if isinstance(current, PortRef):
        return [
            _materialize_item(arena, store, item)
            for item in arena.ordered_items(current)
        ]
    if isinstance(current, tuple):
        return tuple(
            materialize_tree(program, arena, store, child)
            for child in current
        )
    raise RuntimeError("invalid Program.output_tree")


def _materialize_item(
    arena: ArenaEngine,
    store: ReadableStore,
    item: ItemRef,
) -> Any:
    """物化单个终态 Item；非 PRESENT 结果保留其语义 outcome。"""

    outcome = arena.item_outcome(item)
    if outcome is not ItemOutcome.PRESENT:
        return outcome
    binding = arena.value_binding(item)
    if isinstance(binding, RowBinding):
        return store.get(binding)
    if not isinstance(binding, GroupBinding):  # pragma: no cover - 防御分支
        raise RuntimeError(f"unsupported ValueBinding: {binding!r}")
    leaves = [store.get(row) for row in arena.group_rows(binding)]
    return restore_group(leaves, binding.shape.offsets_by_level)


__all__ = ["ReadableStore", "materialize_tree"]
