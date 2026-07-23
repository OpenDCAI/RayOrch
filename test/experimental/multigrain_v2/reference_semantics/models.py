"""Oracle-only records for semantic fixtures."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExpectedEntity:
    key: str
    parents: tuple[tuple[str, str], ...]
    port: str = ""


@dataclass(frozen=True, slots=True)
class ExpectedWorkUnit:
    kind: str
    coordinate: str
    members: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExpectedFailure:
    coordinate: str
    outcome: str
    causes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SemanticCase:
    name: str
    primitive: str
    entities: tuple[ExpectedEntity, ...]
    work_units: tuple[ExpectedWorkUnit, ...]
    failures: tuple[ExpectedFailure, ...] = ()
