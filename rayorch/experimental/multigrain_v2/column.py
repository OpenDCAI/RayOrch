"""The sole V2.2 value-column ABI."""
from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar


T = TypeVar("T")


def validate_column(value: object) -> list[object]:
    if type(value) is not list:
        raise TypeError("Multigrain value columns must be built-in lists")
    return value


def take(column: list[T], selector: slice | tuple[int, ...]) -> list[T]:
    validate_column(column)
    if isinstance(selector, slice):
        return column[selector]
    if not isinstance(selector, tuple) or any(
        isinstance(index, bool) or not isinstance(index, int)
        for index in selector
    ):
        raise TypeError("column selector must be a slice or tuple of ints")
    return [column[index] for index in selector]


def concat(columns: Sequence[list[T]]) -> list[T]:
    result: list[T] = []
    for column in columns:
        validate_column(column)
        result.extend(column)
    return result


__all__ = ["concat", "take", "validate_column"]
