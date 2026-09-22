"""Execution-environment-independent Worker ABI with no Ray dependency."""

from __future__ import annotations

import traceback
from typing import Any, Protocol

from ..failures import GroupFailure, RecordFailure
from .._protocol import (
    BlockRef,
    GrainFailureReport,
    GrainReport,
    DispatchFailure,
    DispatchFailureKind,
    ExpandedRows,
    NestedGroupInput,
    CallInputLayout,
    GrainInvocation,
    CallOutputLayout,
    PortOutputReport,
    RowBinding,
    WorkerReport,
    WorkerDispatchResult,
    restore_nested_group,
)


class WorkerContractError(RuntimeError):
    """A UDF result violates the compiled Worker ABI."""


class BlockStore(Protocol):
    """Minimal block-storage interface required by a Worker."""

    def get(self, binding: RowBinding) -> Any:
        """Read the business value referenced by one physical row binding."""

        ...

    def put(self, values: tuple[Any, ...]) -> BlockRef:
        """Store one coarse value block and return its opaque reference."""

        ...


class Worker:
    """Persistent UDF instance exposing only a value-oriented columnar batch ABI."""

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
        invocations: tuple[GrainInvocation, ...],
        layouts: tuple[CallOutputLayout, ...],
        store: BlockStore,
        *,
        input_layout: CallInputLayout,
    ) -> WorkerDispatchResult:
        """Execute one Grain batch while keeping business failures row-local."""

        try:
            return self._execute(
                invocations,
                layouts,
                store,
                input_layout=input_layout,
            )
        except WorkerContractError as error:
            return self._dispatch_failure(
                DispatchFailureKind.CONTRACT_ERROR,
                error,
            )

    def _execute(
        self,
        invocations: tuple[GrainInvocation, ...],
        layouts: tuple[CallOutputLayout, ...],
        store: BlockStore,
        *,
        input_layout: CallInputLayout,
    ) -> WorkerDispatchResult:
        """Execute after the public boundary has installed contract capture."""

        if not invocations:
            return ()
        columns = self._input_columns(invocations, store)
        layout = input_layout
        if layout.input_count != len(columns):
            raise WorkerContractError(
                "input layout expected "
                f"{layout.input_count} columns, got {len(columns)}"
            )
        positional = columns[: layout.positional_count]
        keyword_columns = columns[layout.positional_count :]
        keywords = dict(zip(layout.keyword_names, keyword_columns))
        keywords.update(layout.static_kwargs)
        try:
            raw = getattr(self.udf, "run", self.udf)(*positional, **keywords)
        except Exception as error:
            return self._dispatch_failure(DispatchFailureKind.UDF_ERROR, error)
        normalized = self._normalize_outputs(raw, layouts, len(invocations))

        # All outputs at one row belong to one atomic Grain. GroupFailure takes
        # precedence over RecordFailure; the first highest-priority sentinel in
        # compiled layout order wins, independent of dict or block layout.
        failures: list[RecordFailure | GroupFailure | None] = [
            None
        ] * len(invocations)
        for values in normalized:
            for index, value in enumerate(values):
                if isinstance(value, GroupFailure):
                    if not isinstance(failures[index], GroupFailure):
                        failures[index] = value
                elif isinstance(value, RecordFailure) and failures[index] is None:
                    failures[index] = value

        reports = [dict() for _ in invocations]
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
                # Flatten each logical output once; failed rows create no
                # provisional bindings.
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
                    reports[index][layout.port] = PortOutputReport(
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

            # Preserve batch row numbers so each successful Grain maps to its
            # UDF result position. Failed rows remain unreferenced in the block.
            block = store.put(values)
            for index, value in enumerate(values):
                if failures[index] is not None:
                    continue
                reports[index][layout.port] = PortOutputReport(
                    layout.port,
                    scalar=RowBinding(block, index),
                    control=value if requires_control else None,
                )

        results: list[WorkerReport] = []
        for invocation, report, failure in zip(invocations, reports, failures):
            if failure is not None:
                results.append(
                    GrainFailureReport(
                        invocation.grain,
                        invocation.generation,
                        failure.cause,
                        suppress_siblings=isinstance(failure, GroupFailure),
                    )
                )
            else:
                results.append(
                    GrainReport(
                        invocation.grain,
                        invocation.generation,
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

    @classmethod
    def _input_columns(
        cls,
        invocations: tuple[GrainInvocation, ...],
        store: BlockStore,
    ) -> tuple[list[Any], ...]:
        width = len(invocations[0].inputs)
        if any(len(invocation.inputs) != width for invocation in invocations):
            raise WorkerContractError("Grain input arity changed inside batch")
        columns: list[list[Any]] = [[] for _ in range(width)]
        for invocation in invocations:
            for index, grain_input in enumerate(invocation.inputs):
                if isinstance(grain_input, RowBinding):
                    columns[index].append(store.get(grain_input))
                elif isinstance(grain_input, NestedGroupInput):
                    leaves = [
                        store.get(binding) for binding in grain_input.bindings
                    ]
                    try:
                        group = restore_nested_group(
                            leaves,
                            grain_input.offsets_by_level,
                        )
                    except ValueError as error:
                        raise WorkerContractError(str(error)) from error
                    columns[index].append(group)
                else:  # pragma: no cover - defensive closed-union branch
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


__all__ = [
    "Worker",
    "BlockStore",
    "WorkerContractError",
]
