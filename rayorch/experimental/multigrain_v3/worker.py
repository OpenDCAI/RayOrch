"""Stage Worker wrapper 与 value-only UDF ABI。

Worker 只根据 BatchCall/RowTake 读取 coarse input blocks、调用用户 UDF、校验原语输出合同
并生成 BatchReport。它不访问 CompiledDAG 全图、Arena tables 或 lineage accumulator。
"""

from __future__ import annotations

import os
import time
from typing import Any

from .contracts import BadRecordError, MISSING
from .protocol import (
    BatchCall,
    BatchReport,
    DispatchFailure,
    FailureKind,
    FilterAck,
    MissingTake,
    ValueTake,
    ValueAck,
)
from .dag import InputMode, Primitive, StageSpec


class WorkerContractError(RuntimeError):
    """表示 UDF 返回值不满足当前 Primitive 的 shape/cardinality 合同。"""


def _sequence(value: Any, label: str) -> tuple[Any, ...]:
    """把 UDF list/tuple 输出规范化为 tuple，否则抛 contract error。"""

    if not isinstance(value, (list, tuple)):
        raise WorkerContractError(f"{label} must be list or tuple")
    return tuple(value)


def _udf_call(udf: Any, *columns: list[Any]) -> Any:
    """优先调用对象的 `run` 方法，否则调用 callable 本身。"""

    return getattr(udf, "run", udf)(*columns)


def _input_columns(
    call: BatchCall,
    input_blocks: tuple[tuple[Any, ...], ...],
    variadic_inputs: frozenset[int],
) -> tuple[list[Any], ...]:
    """按 Invocation/InputTake 从多个 coarse blocks 重建 column-major UDF 输入。"""

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
    """根据 canonical CSR offsets 自底向上恢复 nested Python list。"""

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
    """校验 Map/Reduce 每个 Grain、每个 output Port 恰好一个业务值。"""

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
    """校验 Expand port-major 输出，并保留每个 Grain 的动态 child 序列。"""

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
    """把 normalized `[port][grain][row]` 展平成 per-Port coarse blocks 与 ValueAck。"""

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
            ValueAck(invocation.token, tuple(counts_by_grain[index]))
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
    """在不导入 Ray 的情况下执行一个物理 batch。

    该函数是单进程测试与 Ray actor 共用的执行内核，并把异常规范化为四类框架 failure。
    """

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
                    FilterAck(invocation.token, keep)
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
    """延迟创建 Ray actor class，使 DAG/semantic 单测保持 Ray-free。"""

    global _RAY_WORKER
    if _RAY_WORKER is not None:
        return _RAY_WORKER
    import ray

    @ray.remote(max_concurrency=1, max_restarts=0)
    class RayStageWorker:
        """持有一个 Stage UDF instance 的 persistent 单并发 Ray actor。"""

        def __init__(
            self,
            stage: StageSpec,
            target: Any,
            init_args: tuple[Any, ...],
            init_kwargs: dict[str, Any],
        ) -> None:
            """按 UdfSpec 构造一次 UDF，并保存只读 Stage execution kernel。"""

            self.stage = stage
            self.udf = (
                target(*init_args, **init_kwargs)
                if isinstance(target, type)
                else target
            )
            self.calls = 0

        def run(self, call: BatchCall, *input_blocks: tuple[Any, ...]):
            """执行 BatchCall；Filter 只返回 report，其他原语返回 report 与 output blocks。"""

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
            """返回 observation-only 的调用次数、PID、RSS 和 UDF audit 快照。

            UDF 若暴露 ``batch_audit()`` 或 ``last_*_audit`` mapping，worker 只读取
            其小型标量摘要；它不让 audit 参与 planner、lineage 或 retry 决策。
            """

            audit: dict[str, int | float | str] = {}
            batch_audit = getattr(self.udf, "batch_audit", None)
            if callable(batch_audit):
                value = batch_audit()
            else:
                value = getattr(self.udf, "last_batch_audit", None)
                if value is None:
                    value = getattr(self.udf, "last_table_batch_audit", None)
                if value is None:
                    value = getattr(self.udf, "last_ocr_batch_audit", None)
            if isinstance(value, dict):
                for key, item in value.items():
                    if isinstance(item, (int, float, str)):
                        audit[str(key)] = item

            result: dict[str, Any] = {
                "calls": self.calls,
                "pid": os.getpid(),
                "rss_bytes": _rss_bytes(),
            }
            if audit:
                result["audit"] = audit
            return result

    _RAY_WORKER = RayStageWorker
    return RayStageWorker


def _rss_bytes() -> int:
    """best-effort 读取当前 Worker RSS；不可用时返回 0。"""

    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return 0
