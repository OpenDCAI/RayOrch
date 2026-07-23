"""Stable transport DTOs; Ray execution begins in Phase 3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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


class WorkerContractError(RuntimeError):
    """A normalized worker result does not match its DispatchPlan shape."""


NormalizedBatchOutput = tuple[
    tuple[tuple[Any, ...], ...],
    ...,
]


def build_batch_manifest(
    plan: DispatchPlan,
    outputs_by_port: NormalizedBatchOutput,
) -> tuple[BatchManifest, tuple[tuple[Any, ...], ...]]:
    """Flatten normalized ``[port][grain][row]`` outputs into coarse blocks."""

    if not isinstance(outputs_by_port, tuple):
        raise WorkerContractError("normalized outputs must be a port tuple")
    grain_count = len(plan.entries)
    columns: list[tuple[Any, ...]] = []
    spans: list[list[Span]] = [[] for _ in plan.entries]
    for port_rows in outputs_by_port:
        if not isinstance(port_rows, tuple) or len(port_rows) != grain_count:
            raise WorkerContractError(
                "each output port must contain one row sequence per grain"
            )
        flat: list[Any] = []
        for index, grain_rows in enumerate(port_rows):
            if not isinstance(grain_rows, tuple):
                raise WorkerContractError(
                    "each normalized grain output must be a tuple"
                )
            start = len(flat)
            flat.extend(grain_rows)
            spans[index].append(Span(start, len(flat)))
        columns.append(tuple(flat))

    manifest = BatchManifest(
        dispatch=plan.id,
        acks=tuple(
            GrainAck(entry.token, tuple(spans[index]))
            for index, entry in enumerate(plan.entries)
        ),
        column_lengths=tuple(len(column) for column in columns),
    )
    return manifest, tuple(columns)
