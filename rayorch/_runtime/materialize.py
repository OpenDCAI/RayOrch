"""Output materialization for Executor.

This module accesses semantic state only through InputBatchEngine's read-only
interface; the executor neither reads internal tables nor interprets logical
provenance.
"""

from __future__ import annotations

from typing import Any, Protocol

from .._model import ItemOutcome, ItemRef, PortRef
from .._program.plan import RuntimePlan
from .._protocol import DispatchFailure, RowBinding, restore_nested_group
from ..errors import ExecutionError
from ..result import OutputIssue
from .engine import InputBatchEngine
from .state import NestedGroupBinding


class ReadableStore(Protocol):
    """Minimal block-reading interface required for result materialization."""

    def get(self, binding: RowBinding) -> Any:
        """Read the business value referenced by one physical row binding."""

        ...


def materialize_tree(
    plan: RuntimePlan,
    engine: InputBatchEngine,
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
    engine: InputBatchEngine,
    store: ReadableStore,
    item: ItemRef,
) -> Any:
    """Materialize one terminal Item, preserving non-PRESENT outcomes."""

    outcome = engine.item_outcome(item)
    if outcome is not ItemOutcome.PRESENT:
        cause = (
            None if outcome is ItemOutcome.DROPPED
            else _cause_text(engine.item_cause(item))
        )
        return OutputIssue(outcome, cause)
    binding = engine.value_binding(item)
    if isinstance(binding, RowBinding):
        return _read_value(store, binding)
    if not isinstance(binding, NestedGroupBinding):  # pragma: no cover - defensive
        raise RuntimeError(f"unsupported ValueBinding: {binding!r}")
    leaves = [_read_value(store, row) for row in engine.nested_group_rows(binding)]
    return restore_nested_group(leaves, binding.layout.offsets_by_level)


def _read_value(store: ReadableStore, binding: RowBinding) -> Any:
    value = store.get(binding)
    if isinstance(value, OutputIssue):
        raise ExecutionError("OutputIssue is reserved for framework output results")
    return value


def _cause_text(cause: object | None) -> str | None:
    """Detach readable diagnostics from business objects and Worker reports."""

    if cause is None:
        return None
    if isinstance(cause, DispatchFailure):
        return f"{cause.error_type}: {cause.message}"
    try:
        return str(cause)
    except Exception:
        # A faulty business __str__ must not hide an already recorded failure.
        return type(cause).__name__


__all__ = ["ReadableStore", "materialize_tree"]
