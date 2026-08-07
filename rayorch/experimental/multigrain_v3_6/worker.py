"""与执行环境解耦的 Worker ABI；本模块本身不依赖 Ray。"""

from __future__ import annotations

import os
import traceback
from dataclasses import dataclass
from typing import Any, Protocol

from .model import MISSING
from .protocol import (
    BlockRef,
    GrainFailureReport,
    GrainReport,
    DispatchFailure,
    DispatchFailureKind,
    ExpandedRows,
    GroupInput,
    CallInputLayout,
    GrainInput,
    GrainPlan,
    MissingInput,
    CallOutputLayout,
    OutputReport,
    RecordFailure,
    RowBinding,
    WorkerReport,
    WorkerResult,
    restore_group,
)


class WorkerContractError(RuntimeError):
    """UDF 返回值不满足已编译 Worker ABI。"""


@dataclass(frozen=True, slots=True)
class WorkerSnapshot:
    """一次只读 Worker 快照；不参与执行语义或调度决策。"""

    lifetime_calls: int
    pid: int
    rss_bytes: int
    audit: tuple[tuple[str, int | float | str], ...] = ()
    error: str | None = None


class BlockStore(Protocol):
    """Worker 所需的最小块存储接口。"""

    def get(self, binding: RowBinding) -> Any:
        """读取一个物理行绑定对应的业务值。"""

        ...

    def put(self, values: tuple[Any, ...]) -> BlockRef:
        """写入一个粗粒度值块并返回逻辑块引用。"""

        ...


class Worker:
    """持久 UDF 实例；对外只暴露 value-only、列式批处理 ABI。"""

    def __init__(
        self,
        target: Any,
        init_args: tuple[Any, ...] = (),
        init_kwargs: tuple[tuple[str, Any], ...] = (),
        *,
        input_layout: CallInputLayout,
    ) -> None:
        kwargs = dict(init_kwargs)
        self.udf = target(*init_args, **kwargs) if isinstance(target, type) else target
        self.input_layout = input_layout
        self.calls = 0

    def execute(
        self,
        grain_plans: tuple[GrainPlan, ...],
        layouts: tuple[CallOutputLayout, ...],
        store: BlockStore,
    ) -> WorkerResult:
        """执行一批 Grain，并让业务失败严格停留在对应记录。"""

        try:
            return self._execute(grain_plans, layouts, store)
        except WorkerContractError as error:
            return self._dispatch_failure(
                DispatchFailureKind.CONTRACT_ERROR,
                error,
            )

    def _execute(
        self,
        grain_plans: tuple[GrainPlan, ...],
        layouts: tuple[CallOutputLayout, ...],
        store: BlockStore,
    ) -> WorkerResult:
        """Execute after the public boundary has installed contract capture."""

        if not grain_plans:
            return ()
        self.calls += 1
        columns = self._input_columns(grain_plans, store)
        layout = self.input_layout
        if layout.input_count != len(columns):
            raise WorkerContractError(
                "input layout expected "
                f"{layout.input_count} columns, got {len(columns)}"
            )
        positional = columns[: layout.positional_count]
        keyword_columns = columns[layout.positional_count :]
        keywords = dict(zip(layout.keyword_names, keyword_columns))
        try:
            raw = getattr(self.udf, "run", self.udf)(*positional, **keywords)
        except Exception as error:
            return self._dispatch_failure(DispatchFailureKind.UDF_ERROR, error)
        normalized = self._normalize_outputs(raw, layouts, len(grain_plans))

        # 任一输出列把某个位置标为 RecordFailure 时，该 Grain 的所有输出都
        # 不可见；这保持 multi-output Call 的逐 Grain 原子性。
        failures: list[RecordFailure | None] = [None] * len(grain_plans)
        for values in normalized:
            for index, value in enumerate(values):
                if isinstance(value, RecordFailure) and failures[index] is None:
                    failures[index] = value

        reports = [dict() for _ in grain_plans]
        for layout, values in zip(layouts, normalized):
            live_values = tuple(
                value for index, value in enumerate(values)
                if failures[index] is None
            )
            if layout.expanded_ports:
                groups = tuple(
                    () if failures[index] is not None
                    else self._sequence(value, "expanded value")
                    for index, value in enumerate(values)
                )
                # 同一逻辑输出只扁平化一次；失败位置不产生 provisional rows。
                flat = tuple(value for group in groups for value in group)
                control_expansions = layout.control_ports.intersection(
                    layout.expanded_ports
                )
                if control_expansions and any(type(value) is not bool for value in flat):
                    raise WorkerContractError(
                        f"output {layout.port!r} control rows must contain bool values"
                    )
                block = store.put(flat)
                offset = 0
                for index, group in enumerate(groups):
                    if failures[index] is not None:
                        continue
                    rows = tuple(
                        RowBinding(block, row)
                        for row in range(offset, offset + len(group))
                    )
                    offset += len(group)
                    reports[index][layout.port] = OutputReport(
                        layout.port,
                        expansions=tuple(
                            ExpandedRows(
                                port,
                                rows,
                                tuple(group) if port in control_expansions else None,
                            )
                            for port in layout.expanded_ports
                        ),
                    )
                continue

            requires_control = layout.port in layout.control_ports
            if requires_control and any(
                type(value) is not bool for value in live_values
            ):
                raise WorkerContractError(
                    f"output {layout.port!r} control rows must contain bool values"
                )

            # 保留原 batch 行号，使每个成功 Grain 的 RowBinding 稳定对应 UDF
            # 返回位置；失败行虽然位于粗块中，但不会被任何 ItemRef 引用。
            block = store.put(values)
            for index, value in enumerate(values):
                if failures[index] is not None:
                    continue
                if value is MISSING:
                    raise WorkerContractError(
                        f"output {layout.port!r} cannot contain MISSING"
                    )
                reports[index][layout.port] = OutputReport(
                    layout.port,
                    scalar=RowBinding(block, index),
                    control=value if requires_control else None,
                )

        results: list[WorkerReport] = []
        for grain_plan, report, failure in zip(grain_plans, reports, failures):
            if failure is not None:
                results.append(
                    GrainFailureReport(
                        grain_plan.grain,
                        grain_plan.generation,
                        failure.cause,
                    )
                )
            else:
                results.append(
                    GrainReport(
                        grain_plan.grain,
                        grain_plan.generation,
                        tuple(report[layout.port] for layout in layouts),
                    )
                )
        return tuple(results)

    @staticmethod
    def _dispatch_failure(
        kind: DispatchFailureKind,
        error: Exception,
    ) -> DispatchFailure:
        """Snapshot an exception without requiring the exception to be picklable."""

        return DispatchFailure(
            kind,
            f"{type(error).__module__}.{type(error).__qualname__}",
            str(error),
            traceback.format_exc(),
        )

    def observe(self) -> WorkerSnapshot:
        """读取小型标量 audit；业务状态不会反向影响 Worker ABI。"""

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
            audit.update(
                (str(key), item)
                for key, item in value.items()
                if isinstance(item, (int, float, str))
            )
        return WorkerSnapshot(
            lifetime_calls=self.calls,
            pid=os.getpid(),
            rss_bytes=_rss_bytes(),
            audit=tuple(sorted(audit.items())),
        )

    @classmethod
    def _input_columns(
        cls,
        grain_plans: tuple[GrainPlan, ...],
        store: BlockStore,
    ) -> tuple[list[Any], ...]:
        width = len(grain_plans[0].inputs)
        if any(len(plan.inputs) != width for plan in grain_plans):
            raise WorkerContractError("Grain input arity changed inside batch")
        columns: list[list[Any]] = [[] for _ in range(width)]
        for grain_plan in grain_plans:
            for index, grain_input in enumerate(grain_plan.inputs):
                if isinstance(grain_input, MissingInput):
                    columns[index].append(MISSING)
                elif isinstance(grain_input, RowBinding):
                    columns[index].append(store.get(grain_input))
                elif isinstance(grain_input, GroupInput):
                    leaves = [
                        store.get(binding) for binding in grain_input.bindings
                    ]
                    try:
                        group = restore_group(
                            leaves,
                            grain_input.offsets_by_level,
                        )
                    except ValueError as error:
                        raise WorkerContractError(str(error)) from error
                    columns[index].append(group)
                else:  # pragma: no cover - 封闭联合类型的防御分支
                    raise WorkerContractError(
                        f"unsupported GrainInput: {grain_input!r}"
                    )
        return tuple(columns)

    @classmethod
    def _normalize_outputs(
        cls,
        raw: Any,
        layouts: tuple[CallOutputLayout, ...],
        grain_count: int,
    ) -> tuple[tuple[Any, ...], ...]:
        output_count = len(layouts)
        ports = (raw,) if output_count == 1 else cls._sequence(raw, "multi-output")
        if len(ports) != output_count:
            raise WorkerContractError(
                f"UDF expected {output_count} output ports, got {len(ports)}"
            )
        normalized = []
        for layout, column in zip(layouts, ports):
            values = cls._sequence(column, f"output column {layout.port!r}")
            if len(values) != grain_count:
                raise WorkerContractError(
                    f"output {layout.port!r} expected {grain_count} Grain rows, "
                    f"got {len(values)}"
                )
            normalized.append(values)
        return tuple(normalized)

    @staticmethod
    def _sequence(value: Any, label: str) -> tuple[Any, ...]:
        if not isinstance(value, (list, tuple)):
            raise WorkerContractError(f"{label} must be list or tuple")
        return tuple(value)


def _rss_bytes() -> int:
    """Best-effort current process RSS for benchmark diagnostics."""

    try:
        import psutil  # pyright: ignore[reportMissingModuleSource]

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return 0


__all__ = [
    "Worker",
    "WorkerSnapshot",
    "BlockStore",
    "WorkerContractError",
]
