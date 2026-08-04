"""本地执行与 Ray actor 共用的 Worker ABI；本模块本身不依赖 Ray。"""

from __future__ import annotations

from typing import Any, Protocol

from .model import MISSING
from .protocol import (
    CallFailureReport,
    CallReport,
    ExpandedRows,
    GroupTake,
    InputTake,
    InvocationPlan,
    MissingTake,
    OutputLayout,
    OutputReport,
    RecordFailure,
    RowBinding,
    ScalarTake,
    WorkerReport,
)


class WorkerContractError(RuntimeError):
    """UDF 返回值不满足已编译 Worker ABI。"""


class ValueStore(Protocol):
    """Worker 所需的最小块存储接口。"""

    def get(self, binding: RowBinding) -> Any:
        """读取一个物理行绑定对应的业务值。"""

        ...

    def put(self, values: tuple[Any, ...]):
        """写入一个粗粒度值块并返回逻辑块引用。"""

        ...



class LocalWorker:
    """持久 UDF 实例；对外只暴露 value-only、列式批处理 ABI。"""

    def __init__(
        self,
        target: Any,
        init_args: tuple[Any, ...] = (),
        init_kwargs: tuple[tuple[str, Any], ...] = (),
    ) -> None:
        kwargs = dict(init_kwargs)
        self.udf = target(*init_args, **kwargs) if isinstance(target, type) else target

    def execute(
        self,
        invocations: tuple[InvocationPlan, ...],
        layouts: tuple[OutputLayout, ...],
        store: ValueStore,
    ) -> tuple[WorkerReport, ...]:
        """执行一批 Grain，并让业务失败严格停留在对应记录。"""

        if not invocations:
            return ()
        columns = self._input_columns(invocations, store)
        raw = getattr(self.udf, "run", self.udf)(*columns)
        normalized = self._normalize_outputs(raw, len(layouts), len(invocations))

        # 任一输出列把某个位置标为 RecordFailure 时，该 Grain 的所有输出都
        # 不可见；这保持 multi-output Call 的逐 Grain 原子性。
        failures: list[RecordFailure | None] = [None] * len(invocations)
        for values in normalized:
            for index, value in enumerate(values):
                if isinstance(value, RecordFailure) and failures[index] is None:
                    failures[index] = value

        reports = [dict() for _ in invocations]
        for layout, values in zip(layouts, normalized):
            live_values = tuple(
                value for index, value in enumerate(values)
                if failures[index] is None
            )
            if layout.is_mask and any(type(value) is not bool for value in live_values):
                raise WorkerContractError("mask output must contain bool values")

            if layout.expanded_ports:
                groups = tuple(
                    () if failures[index] is not None
                    else self._sequence(value, "expanded value")
                    for index, value in enumerate(values)
                )
                # 同一逻辑输出只扁平化一次；失败位置不产生 provisional rows。
                flat = tuple(value for group in groups for value in group)
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
                            ExpandedRows(port, rows) for port in layout.expanded_ports
                        ),
                        control=values[index] if layout.is_mask else None,
                    )
                continue

            # 保留原 batch 行号，使每个成功 Grain 的 RowBinding 稳定对应 UDF
            # 返回位置；失败行虽然位于粗块中，但不会被任何 ItemRef 引用。
            block = store.put(values)
            for index, value in enumerate(values):
                if failures[index] is not None:
                    continue
                if value is MISSING:
                    raise WorkerContractError("MISSING cannot be a UDF output")
                reports[index][layout.port] = OutputReport(
                    layout.port,
                    scalar=RowBinding(block, index),
                    control=value if layout.is_mask else None,
                )

        results: list[WorkerReport] = []
        for invocation, report, failure in zip(invocations, reports, failures):
            if failure is not None:
                results.append(
                    CallFailureReport(
                        invocation.grain,
                        invocation.generation,
                        failure.cause,
                    )
                )
            else:
                results.append(
                    CallReport(
                        invocation.grain,
                        invocation.generation,
                        tuple(report[layout.port] for layout in layouts),
                    )
                )
        return tuple(results)

    @classmethod
    def _input_columns(
        cls,
        invocations: tuple[InvocationPlan, ...],
        store: ValueStore,
    ) -> tuple[list[Any], ...]:
        width = len(invocations[0].inputs)
        if any(len(invocation.inputs) != width for invocation in invocations):
            raise WorkerContractError("invocation input arity changed inside batch")
        columns: list[list[Any]] = [[] for _ in range(width)]
        for invocation in invocations:
            for index, take in enumerate(invocation.inputs):
                if isinstance(take, MissingTake):
                    columns[index].append(MISSING)
                elif isinstance(take, ScalarTake):
                    columns[index].append(store.get(take.binding))
                elif isinstance(take, GroupTake):
                    leaves = [store.get(binding) for binding in take.bindings]
                    columns[index].append(
                        cls.restore_group(leaves, take.offsets_by_level)
                    )
                else:  # pragma: no cover - 封闭联合类型的防御分支
                    raise WorkerContractError(f"unsupported InputTake: {take!r}")
        return tuple(columns)

    @staticmethod
    def restore_group(
        leaves: list[Any],
        offsets_by_level: tuple[tuple[int, ...], ...],
    ) -> list[Any]:
        """由最内层叶子向上应用 offsets，恢复唯一根 group。"""

        nodes: list[Any] = leaves
        for offsets in reversed(offsets_by_level):
            nodes = [
                nodes[offsets[index] : offsets[index + 1]]
                for index in range(len(offsets) - 1)
            ]
        if len(nodes) != 1:
            raise WorkerContractError("GroupShape does not have one root")
        return nodes[0]

    @classmethod
    def _normalize_outputs(
        cls,
        raw: Any,
        output_count: int,
        grain_count: int,
    ) -> tuple[tuple[Any, ...], ...]:
        ports = (raw,) if output_count == 1 else cls._sequence(raw, "multi-output")
        if len(ports) != output_count:
            raise WorkerContractError("UDF output port count mismatch")
        normalized = []
        for column in ports:
            values = cls._sequence(column, "output column")
            if len(values) != grain_count:
                raise WorkerContractError("UDF output grain count mismatch")
            normalized.append(values)
        return tuple(normalized)

    @staticmethod
    def _sequence(value: Any, label: str) -> tuple[Any, ...]:
        if not isinstance(value, (list, tuple)):
            raise WorkerContractError(f"{label} must be list or tuple")
        return tuple(value)


__all__ = [
    "GroupTake",
    "InputTake",
    "InvocationPlan",
    "LocalWorker",
    "MissingTake",
    "OutputLayout",
    "ScalarTake",
    "WorkerContractError",
]
