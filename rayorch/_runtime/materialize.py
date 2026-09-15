"""Output materialization for Executor.

This module accesses semantic state only through MicrobatchEngine's read-only
interface; the executor neither reads internal tables nor interprets logical
provenance.
"""

from __future__ import annotations

from typing import Any, Protocol

from .._model import ItemOutcome, ItemRef, PortRef
from .._program.plan import RuntimePlan
from .._protocol import RowBinding, restore_nested_group
from .engine import MicrobatchEngine
from .state import NestedGroupBinding


class ReadableStore(Protocol):
    """Minimal block-reading interface required for result materialization."""

    def get(self, binding: RowBinding) -> Any:
        """Read the business value referenced by one physical row binding."""

        ...


def materialize_tree(
    plan: RuntimePlan,
    engine: MicrobatchEngine,
    store: ReadableStore,
    tree: object | None = None,
) -> object:
    """Materialize the Program output tree in stable lineage order."""

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
    """Materialize one terminal Item, preserving non-PRESENT outcomes."""

    outcome = engine.item_outcome(item)
    if outcome is not ItemOutcome.PRESENT:
        return outcome
    binding = engine.value_binding(item)
    if isinstance(binding, RowBinding):
        return store.get(binding)
    if not isinstance(binding, NestedGroupBinding):  # pragma: no cover - defensive
        raise RuntimeError(f"unsupported ValueBinding: {binding!r}")
    leaves = [store.get(row) for row in engine.nested_group_rows(binding)]
    return restore_nested_group(leaves, binding.layout.offsets_by_level)


__all__ = ["ReadableStore", "materialize_tree"]
