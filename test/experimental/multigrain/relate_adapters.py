"""Importable relation adapters for the by-ref (tier ③) escape hatch.

These live in a real module (not a closure) so ``relation_adapter="pkg:fn"``
can resolve them and the IR stays serializable. Adapters only speak
invocation-local indices; they never touch internal record ids.
"""
from __future__ import annotations

from typing import Any, List, Tuple


def link_by_index(raw_values: List[Any]) -> List[Tuple[Any, dict[str, int]]]:
    """Pair the i-th image with the i-th caption (positional relation)."""
    return [
        (value, {"image": index, "caption": index})
        for index, value in enumerate(raw_values)
    ]
