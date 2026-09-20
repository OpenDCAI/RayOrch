"""Explicit business-failure values that a UDF may return."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RecordFailure:
    """Mark one logical record as a business failure."""

    cause: Any


@dataclass(frozen=True, slots=True)
class GroupFailure:
    """Fail one record and suppress siblings with the same direct parent."""

    cause: Any

__all__ = ["GroupFailure", "RecordFailure"]
