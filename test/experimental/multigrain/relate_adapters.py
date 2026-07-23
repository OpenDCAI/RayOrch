"""Importable relation adapters for the by-ref (tier ③) escape hatch.

These live in a real module (not a closure) so ``relation_adapter="pkg:fn"``
can resolve them and the IR stays serializable. Adapters only speak
invocation-local indices; they never touch internal record ids.
"""
from __future__ import annotations

from typing import Any, List, Tuple

NON_CALLABLE_ADAPTER = 42


def link_by_index(raw_values: List[Any]) -> List[Tuple[Any, dict[str, int]]]:
    """Pair the i-th image with the i-th caption (positional relation)."""
    return [
        (value, {"image": index, "caption": index})
        for index, value in enumerate(raw_values)
    ]


def pair_from_fields(raw_values: List[Any]) -> List[Tuple[Any, dict[str, int]]]:
    """Use explicit invocation-local indexes emitted in each output value."""

    return [
        (
            value,
            {
                "image": int(value["image_idx"]),
                "caption": int(value["caption_idx"]),
            },
        )
        for value in raw_values
    ]


def pair_left_right_by_index(
    raw_values: List[Any],
) -> List[Tuple[Any, dict[str, int]]]:
    return [
        (value, {"left": index, "right": index})
        for index, value in enumerate(raw_values)
    ]


def pair_candidates_by_key(
    raw_values: List[Any],
) -> List[Tuple[Any, dict[str, int]]]:
    """Strip local indexes from values while retaining them as relation evidence."""
    return [
        (
            {"left": value["left"], "right": value["right"]},
            {
                "left": int(value["left_index"]),
                "right": int(value["right_index"]),
            },
        )
        for value in raw_values
    ]
