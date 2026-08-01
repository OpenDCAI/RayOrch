"""Stage worker wrapper and value-only UDF ABI."""

from __future__ import annotations

import os
import time
from typing import Any

from .api import BadRecordError, MISSING
from .protocol import (
    BatchCall,
    BatchReport,
    DispatchFailure,
    FailureKind,
    InvocationAck,
    MissingTake,
    ValueTake,
)
from .dag import InputMode, Primitive, StageSpec


class WorkerContractError(RuntimeError):
    """A UDF result does not satisfy its primitive contract."""


def _sequence(value: Any, label: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise WorkerContractError(f"{label} must be list or tuple")
    return tuple(value)


def _udf_call(udf: Any, *columns: list[Any]) -> Any:
    return getattr(udf, "run", udf)(*columns)


def _input_columns(
    call: BatchCall,
    input_blocks: tuple[tuple[Any, ...], ...],
    variadic_inputs: frozenset[int],
) -> tuple[list[Any], ...]:
    if not call.invocations:
        return ()
    width = len(call.invocations[0].inputs)
    columns: list[list[Any]] = [[] for _ in range(width)]
    for invocation in call.invocations:
        if len(invocation.inputs) != width:
            raise WorkerContractError("Invocation input arity changed")
        for index, take in enumerate(invocation.inputs):
            if isinstance(take, MissingTake):
                columns[index].append(MISSING)
                continue
            assert isinstance(take, ValueTake)
            try:
                values = [
                    input_blocks[row_take.ref_slot][row_take.row]
                    for row_take in take.rows
                ]
            except (IndexError, TypeError) as error:
                raise WorkerContractError("invalid ValueTake") from error
            if take.group_shape is not None:
                columns[index].append(
                    _restore_group(values, take.group_shape.offsets_by_level)
                )
            else:
                columns[index].append(
                    values
                    if index in variadic_inputs
                    else (values[0] if len(values) == 1 else values)
                )
    return tuple(columns)


def _restore_group(
    leaves: list[Any],
    offsets_by_level: tuple[tuple[int, ...], ...],
) -> list[Any]:
    """Reconstruct nested lists bottom-up from canonical CSR offsets."""

    nodes: list[Any] = leaves
    for offsets in reversed(offsets_by_level):
        nodes = [
            nodes[offsets[index] : offsets[index + 1]]
            for index in range(len(offsets) - 1)
        ]
    if len(nodes) != 1:
        raise WorkerContractError("group shape does not have one root")
    return nodes[0]


def _normalize_one(
    raw: Any,
    output_count: int,
    grain_count: int,
) -> tuple[tuple[tuple[Any, ...], ...], ...]:
    ports = (raw,) if output_count == 1 else _sequence(raw, "multi-output result")
    if len(ports) != output_count:
        raise WorkerContractError("UDF output port count mismatch")
    normalized = []
    for values in ports:
        rows = _sequence(values, "output column")
        if len(rows) != grain_count:
            raise WorkerContractError("output grain count mismatch")
        if any(value is MISSING for value in rows):
            raise WorkerContractError("MISSING cannot be a UDF output")
        normalized.append(tuple((value,) for value in rows))
    return tuple(normalized)


def _normalize_expand(
    raw: Any,
    output_count: int,
    grain_count: int,
) -> tuple[tuple[tuple[Any, ...], ...], ...]:
    ports = (raw,) if output_count == 1 else _sequence(
        raw, "multi-output Expand result"
    )
    if len(ports) != output_count:
        raise WorkerContractError("Expand output port count mismatch")
    normalized = []
    for values in ports:
        grains = _sequence(values, "Expand output column")
        if len(grains) != grain_count:
            raise WorkerContractError("Expand output grain count mismatch")
        port_rows = tuple(
            _sequence(children, "Expand children") for children in grains
        )
        if any(value is MISSING for rows in port_rows for value in rows):
            raise WorkerContractError("MISSING cannot be an Expand output")
        normalized.append(port_rows)
    return tuple(normalized)


def _flatten(
    call: BatchCall,
    normalized: tuple[tuple[tuple[Any, ...], ...], ...],
    *,
    started_at: float,
) -> tuple[BatchReport, tuple[tuple[Any, ...], ...]]:
    blocks: list[tuple[Any, ...]] = []
    counts_by_grain = [[] for _ in call.invocations]
    for port_rows in normalized:
        flat = []
        for index, rows in enumerate(port_rows):
            counts_by_grain[index].append(len(rows))
            flat.extend(rows)
        blocks.append(tuple(flat))
    report = BatchReport(
        dispatch=call.dispatch,
        acks=tuple(
            InvocationAck(invocation.token, tuple(counts_by_grain[index]))
            for index, invocation in enumerate(call.invocations)
        ),
        column_lengths=tuple(len(block) for block in blocks),
        worker_started_at=started_at,
        worker_finished_at=time.monotonic(),
        worker_rss_bytes=_rss_bytes(),
    )
    return report, tuple(blocks)


def execute_call(
    stage: StageSpec,
    udf: Any,
    call: BatchCall,
    input_blocks: tuple[tuple[Any, ...], ...],
) -> tuple[BatchReport | DispatchFailure, tuple[tuple[Any, ...], ...]]:
    """Execute one physical batch without importing Ray."""

    started_at = time.monotonic()
    try:
        rpc_specs = tuple(
            spec
            for spec in stage.inputs
            if spec.mode is not InputMode.ANCHOR
        )
        variadic_inputs = frozenset(
            index
            for index, spec in enumerate(rpc_specs)
            if spec.mode is InputMode.GROUP
        )
        columns = _input_columns(call, input_blocks, variadic_inputs)
        if stage.kind is Primitive.FILTER:
            mask = _sequence(_udf_call(udf, *columns), "Filter mask")
            if len(mask) != len(call.invocations) or any(
                type(value) is not bool for value in mask
            ):
                raise WorkerContractError(
                    "Filter must return one bool per invocation"
                )
            report = BatchReport(
                dispatch=call.dispatch,
                acks=tuple(
                    InvocationAck(invocation.token, keep=keep)
                    for invocation, keep in zip(call.invocations, mask)
                ),
                column_lengths=(),
                worker_started_at=started_at,
                worker_finished_at=time.monotonic(),
                worker_rss_bytes=_rss_bytes(),
            )
            return report, ()

        raw = _udf_call(udf, *columns)
        normalized = (
            _normalize_expand(raw, stage.output_count, len(call.invocations))
            if stage.kind is Primitive.EXPAND
            else _normalize_one(raw, stage.output_count, len(call.invocations))
        )
        return _flatten(call, normalized, started_at=started_at)
    except BadRecordError as error:
        bad_token = (
            call.invocations[error.index].token
            if error.index < len(call.invocations)
            else None
        )
        return (
            DispatchFailure(
                call.dispatch,
                FailureKind.BAD_RECORD,
                str(error),
                bad_token=bad_token,
            ),
            (),
        )
    except WorkerContractError as error:
        return (
            DispatchFailure(
                call.dispatch,
                FailureKind.CONTRACT_ABORT,
                str(error),
            ),
            (),
        )
    except Exception as error:
        return (
            DispatchFailure(
                call.dispatch,
                FailureKind.UDF_ERROR,
                f"{type(error).__name__}: {error}",
            ),
            (),
        )


_RAY_WORKER = None


def get_ray_worker_class():
    """Create the Ray actor lazily so DAG/semantic tests remain Ray-free."""

    global _RAY_WORKER
    if _RAY_WORKER is not None:
        return _RAY_WORKER
    import ray

    @ray.remote(max_concurrency=1, max_restarts=0)
    class RayStageWorker:
        def __init__(
            self,
            stage: StageSpec,
            target: Any,
            init_args: tuple[Any, ...],
            init_kwargs: dict[str, Any],
        ) -> None:
            self.stage = stage
            self.udf = (
                target(*init_args, **init_kwargs)
                if isinstance(target, type)
                else target
            )
            self.calls = 0

        def run(self, call: BatchCall, *input_blocks: tuple[Any, ...]):
            self.calls += 1
            result, blocks = execute_call(
                self.stage,
                self.udf,
                call,
                tuple(input_blocks),
            )
            if self.stage.kind is Primitive.FILTER:
                return result
            if isinstance(result, DispatchFailure):
                return (
                    result,
                    *(tuple() for _ in range(self.stage.output_count)),
                )
            return (result, *blocks)

        def stats(self) -> dict[str, int]:
            return {
                "calls": self.calls,
                "pid": os.getpid(),
                "rss_bytes": _rss_bytes(),
            }

    _RAY_WORKER = RayStageWorker
    return RayStageWorker


def _rss_bytes() -> int:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return 0
