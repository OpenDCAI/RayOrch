"""Executor 的输出物化逻辑。

该模块只通过 :class:`MicrobatchEngine` 的只读公开接口访问语义状态；执行器不读取
内部 table，也不解释逻辑 provenance。
"""

from __future__ import annotations

from typing import Any, Protocol

from ..model import ItemOutcome, ItemRef, PortRef
from ..program.plan import RuntimePlan
from ..protocol import RowBinding, restore_group
from .engine import MicrobatchEngine
from .state import GroupBinding


class ReadableStore(Protocol):
    """结果物化所需的最小块读取接口。"""

    def get(self, binding: RowBinding) -> Any:
        """读取一个物理行绑定对应的业务值。"""

        ...


def materialize_tree(
    plan: RuntimePlan,
    engine: MicrobatchEngine,
    store: ReadableStore,
    tree: object | None = None,
) -> object:
    """按稳定 lineage 顺序物化 Program 输出树。"""

    current = plan.output_tree if tree is None else tree
    if isinstance(current, PortRef):
        return [
            _materialize_item(engine, store, item)
            for item in engine.ordered_items(current)
        ]
    if isinstance(current, tuple):
        return tuple(
            materialize_tree(plan, engine, store, child)
            for child in current
        )
    raise RuntimeError("invalid Program.output_tree")


def _materialize_item(
    engine: MicrobatchEngine,
    store: ReadableStore,
    item: ItemRef,
) -> Any:
    """物化单个终态 Item；非 PRESENT 结果保留其语义 outcome。"""

    outcome = engine.item_outcome(item)
    if outcome is not ItemOutcome.PRESENT:
        return outcome
    binding = engine.value_binding(item)
    if isinstance(binding, RowBinding):
        return store.get(binding)
    if not isinstance(binding, GroupBinding):  # pragma: no cover - 防御分支
        raise RuntimeError(f"unsupported ValueBinding: {binding!r}")
    leaves = [store.get(row) for row in engine.group_rows(binding)]
    return restore_group(leaves, binding.layout.offsets_by_level)


__all__ = ["ReadableStore", "materialize_tree"]
