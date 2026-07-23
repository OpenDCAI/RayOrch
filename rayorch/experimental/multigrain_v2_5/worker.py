"""Stable transport DTOs; Ray execution begins in Phase 3."""

from __future__ import annotations

from dataclasses import dataclass

from .grain import AttemptToken


@dataclass(frozen=True, slots=True)
class RowTake:
    ref_slot: int
    row: int


@dataclass(frozen=True, slots=True)
class DispatchEntry:
    token: AttemptToken
    role_takes: tuple[tuple[RowTake, ...], ...]


@dataclass(frozen=True, slots=True)
class DispatchPlan:
    id: int
    node: int
    entries: tuple[DispatchEntry, ...]


@dataclass(frozen=True, slots=True)
class Span:
    start: int
    stop: int


@dataclass(frozen=True, slots=True)
class GrainAck:
    token: AttemptToken
    spans_by_port: tuple[Span, ...]


@dataclass(frozen=True, slots=True)
class BatchManifest:
    dispatch: int
    acks: tuple[GrainAck, ...]
    column_lengths: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DispatchErrorReport:
    dispatch: int
    kind: str
    bad_token: AttemptToken | None
    message: str


WorkerManifest = BatchManifest | DispatchErrorReport
