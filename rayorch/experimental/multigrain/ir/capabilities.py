"""Scheduling facts derived from a verified operation and its relations."""
from __future__ import annotations

from .graph import NodeSpec
from .operations import ExpandOp, FilterByMaskOp, FilterOp, MapOp


def is_row_partitionable(node: NodeSpec) -> bool:
    """Whether arbitrary exact row partitions preserve node semantics."""

    return isinstance(node.operation, (MapOp, FilterOp, FilterByMaskOp, ExpandOp))

__all__ = ["is_row_partitionable"]
