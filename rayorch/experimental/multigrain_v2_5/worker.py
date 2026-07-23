"""Stable transport DTOs; Ray execution begins in Phase 3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .api import BadRecordError
from .graph import Primitive
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


def _call_udf(udf: Any, *role_columns: list[Any]) -> Any:
    function = getattr(udf, "run", udf)
    return function(*role_columns)


def _as_sequence(value: Any, *, label: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise WorkerContractError(f"{label} must be a list or tuple")
    return tuple(value)


def _normalize_one_per_grain(
    raw: Any,
    *,
    output_arity: int,
    grain_count: int,
) -> NormalizedBatchOutput:
    raw_ports = (raw,) if output_arity == 1 else _as_sequence(
        raw,
        label="multi-output result",
    )
    if len(raw_ports) != output_arity:
        raise WorkerContractError("UDF output port count mismatch")
    normalized = []
    for port_values in raw_ports:
        values = _as_sequence(port_values, label="output column")
        if len(values) != grain_count:
            raise WorkerContractError("output column grain count mismatch")
        normalized.append(tuple((value,) for value in values))
    return tuple(normalized)


def _normalize_expand(
    raw: Any,
    *,
    output_arity: int,
    grain_count: int,
) -> NormalizedBatchOutput:
    raw_ports = (raw,) if output_arity == 1 else _as_sequence(
        raw,
        label="multi-output Expand result",
    )
    if len(raw_ports) != output_arity:
        raise WorkerContractError("Expand output port count mismatch")
    normalized = []
    for port_values in raw_ports:
        values = _as_sequence(port_values, label="Expand output column")
        if len(values) != grain_count:
            raise WorkerContractError("Expand output grain count mismatch")
        normalized.append(
            tuple(
                _as_sequence(children, label="Expand grain children")
                for children in values
            )
        )
    return tuple(normalized)


def _role_columns(
    plan: DispatchPlan,
    input_blocks: tuple[tuple[Any, ...], ...],
    role_names: tuple[str, ...],
) -> tuple[list[Any], ...]:
    if not plan.entries:
        return ()
    role_count = len(plan.entries[0].role_takes)
    if len(role_names) != role_count:
        raise WorkerContractError("compiled role names do not match DispatchPlan")
    columns: list[list[Any]] = [[] for _ in range(role_count)]
    for entry in plan.entries:
        if len(entry.role_takes) != role_count:
            raise WorkerContractError("DispatchEntry role arity mismatch")
        for role_index, takes in enumerate(entry.role_takes):
            values = []
            for take in takes:
                try:
                    values.append(input_blocks[take.ref_slot][take.row])
                except (IndexError, TypeError) as error:
                    raise WorkerContractError("invalid input RowTake") from error
            columns[role_index].append(
                values
                if role_names[role_index] == "members"
                else (values[0] if len(values) == 1 else values)
            )
    return tuple(columns)


_RAY_WORKER_CLASS = None


def get_ray_worker_class():
    """Create the Ray actor lazily so Phase 0/1 imports remain Ray-free."""

    global _RAY_WORKER_CLASS
    if _RAY_WORKER_CLASS is not None:
        return _RAY_WORKER_CLASS

    import ray

    @ray.remote(max_concurrency=1, max_restarts=0)
    class RayWorker:
        def __init__(
            self,
            target: Any,
            init_args: tuple[Any, ...],
            init_kwargs: dict[str, Any],
        ) -> None:
            self.udf = (
                target(*init_args, **init_kwargs)
                if isinstance(target, type)
                else target
            )
            self.calls = 0

        def run(
            self,
            kind_value: str,
            output_arity: int,
            role_names: tuple[str, ...],
            plan: DispatchPlan,
            *input_blocks: tuple[Any, ...],
        ):
            self.calls += 1
            try:
                kind = Primitive(kind_value)
                role_columns = _role_columns(
                    plan,
                    tuple(input_blocks),
                    role_names,
                )
                if kind is Primitive.FILTER:
                    if output_arity != 1 or not role_columns:
                        raise WorkerContractError(
                            "Phase 3 Filter supports one target output"
                        )
                    mask = _as_sequence(
                        _call_udf(self.udf, *role_columns),
                        label="Filter mask",
                    )
                    if len(mask) != len(plan.entries) or any(
                        type(value) is not bool for value in mask
                    ):
                        raise WorkerContractError(
                            "Filter UDF must return one bool per grain"
                        )
                    outputs: NormalizedBatchOutput = (
                        tuple(
                            ((role_columns[0][index],) if keep else ())
                            for index, keep in enumerate(mask)
                        ),
                    )
                else:
                    raw = _call_udf(self.udf, *role_columns)
                    if kind is Primitive.EXPAND:
                        outputs = _normalize_expand(
                            raw,
                            output_arity=output_arity,
                            grain_count=len(plan.entries),
                        )
                    elif kind in {
                        Primitive.MAP,
                        Primitive.REDUCE,
                        Primitive.RELATE,
                    }:
                        outputs = _normalize_one_per_grain(
                            raw,
                            output_arity=output_arity,
                            grain_count=len(plan.entries),
                        )
                    else:
                        raise WorkerContractError(
                            f"{kind.value} cannot execute on RayWorker"
                        )
                manifest, columns = build_batch_manifest(plan, outputs)
                return (manifest, *columns)
            except BadRecordError as error:
                bad_token = (
                    plan.entries[error.index].token
                    if error.index < len(plan.entries)
                    else None
                )
                report = DispatchErrorReport(
                    plan.id,
                    "bad_record",
                    bad_token,
                    str(error),
                )
                return (report, *(tuple() for _ in range(output_arity)))
            except WorkerContractError as error:
                report = DispatchErrorReport(
                    plan.id,
                    "contract",
                    None,
                    str(error),
                )
                return (report, *(tuple() for _ in range(output_arity)))
            except Exception as error:
                report = DispatchErrorReport(
                    plan.id,
                    "generic_udf",
                    None,
                    f"{type(error).__name__}: {error}",
                )
                return (report, *(tuple() for _ in range(output_arity)))

        def stats(self) -> dict[str, int]:
            return {"calls": self.calls}

    _RAY_WORKER_CLASS = RayWorker
    return RayWorker
