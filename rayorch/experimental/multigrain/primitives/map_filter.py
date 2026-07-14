"""Record-preserving and record-filtering operator wrappers."""
from __future__ import annotations

from typing import Any, List, Sequence
import uuid

from rayorch.runtime.core import BadRecordError

from ._utils import (
    as_tuple,
    checked_output_lists,
    require_same_trace_grain,
    normalize_mask,
    output_name,
)
from .output import PortBatchBuilder, merge_aligned_inputs, select_filter_outputs
from ._binding import BoundPrimitive, PrimitiveBinding
from ..data.batch import (
    DeferredRecord,
    ErrorTrace,
    PortBatch,
    _align_by_identity,
    _as_columns,
    _call_user,
    _parent_display,
    _source_item,
    _without_index,
)
from ..ir.operations import FilterByMaskOp, FilterOp, MapOp
from ..ir.policy import (
    RecordRecoveryAction,
    RecoveryPolicy,
    RetryTiming,
    WorkerPoolSpec,
)
from ..ir.relations import SameAs, SubsetOf
from ..tracing import TracePort, ensure_trace_ports


class Map(BoundPrimitive):
    """Record-preserving logical operator."""

    def __init__(
        self,
        op_cls: Any,
        *args: Any,
        name: str | None = None,
        num_outputs: int = 1,
        workers: WorkerPoolSpec | None = None,
        recovery: RecoveryPolicy | None = None,
        **kwargs: Any,
    ):
        self._binding = PrimitiveBinding.create(
            op_cls,
            tuple(args),
            kwargs,
            name=name,
            num_outputs=num_outputs,
            workers=workers,
            recovery=recovery,
        )

    def __call__(
        self,
        *ports: PortBatch | TracePort,
    ) -> PortBatch | TracePort | tuple[PortBatch, ...] | tuple[TracePort, ...]:
        symbolic = ensure_trace_ports(ports)
        if symbolic is not None:
            self._binding.require_compilable("Map")
            grain = require_same_trace_grain(f"Map '{self.name}'", symbolic)
            return symbolic[0].tracer.add_node(
                name=self.name,
                inputs=symbolic,
                operation=MapOp(self.factory_spec),
                output_grains=(grain,) * self.num_outputs,
                relations=(SameAs(symbolic[0].ref),) * self.num_outputs,
                recovery=self.recovery,
                workers=self.workers,
            )
        aligned = _align_by_identity(ports)
        result, deferred = self.run_with_recovery(*aligned)
        if deferred:
            raise NotImplementedError(
                "deferred record retry requires MultigrainRayExecutor.execute_stream"
            )
        return result

    def run_with_recovery(
        self,
        *ports: PortBatch,
        force_inline: bool = False,
    ) -> tuple[
        PortBatch | tuple[PortBatch, ...],
        tuple[DeferredRecord, ...],
    ]:
        aligned = _align_by_identity(ports)
        columns = [_as_columns(port) for port in aligned]
        try:
            outputs = self._checked_outputs(_call_user(self.op, *columns))
            return self._make_outputs(self._merge_lineage(aligned), outputs), ()
        except BadRecordError as exc:
            if exc.index is None:
                raise
            return self._run_with_bad_index(
                aligned,
                exc,
                force_inline=force_inline,
            )

    def _make_outputs(
        self,
        base: PortBatch,
        outputs: Sequence[Sequence[Any]],
        *,
        errors: Sequence[ErrorTrace] = (),
    ) -> PortBatch | tuple[PortBatch, ...]:
        result = PortBatchBuilder.preserved(
            base,
            outputs,
            name=self.name,
            op_name=self.name,
            errors=errors,
            preserve_name=True,
        )
        return result[0] if len(result) == 1 else result

    @staticmethod
    def _merge_lineage(ports: Sequence[PortBatch]) -> PortBatch:
        """Merge metadata from every same-identity input of a diamond fan-in."""
        return merge_aligned_inputs(ports)

    def _trace_for(self, ports: Sequence[PortBatch], bad_index: int, error: str) -> ErrorTrace:
        base = ports[0]
        upstream_path: List[str] = []
        for port in ports:
            for step in port.lineage[bad_index]:
                if step not in upstream_path:
                    upstream_path.append(step)
        upstream_path.append(self.name)
        ancestors = dict(base.ancestors[bad_index])
        ancestors[base.name] = base.record_ids[bad_index]
        return ErrorTrace(
            source_item=_source_item(base, bad_index),
            logical_item=base.display_keys[bad_index],
            failed_op=self.name,
            grain=base.name,
            upstream_path=tuple(upstream_path),
            parent=_parent_display(base, bad_index),
            action="quarantined",
            error=error,
            ancestors=ancestors,
        )

    def _run_isolating(
        self,
        ports: Sequence[PortBatch],
        first_error: BadRecordError | None = None,
        *,
        force_inline: bool = False,
    ) -> tuple[
        Sequence[PortBatch],
        tuple[List[Any], ...],
        List[ErrorTrace],
        List[DeferredRecord],
    ]:
        """Recover attributable rows while retaining dense healthy execution.

        A retryable bad row is probed as a singleton up to its record budget.
        Every remaining row is still re-run as one dense batch. Successful pieces
        are merged in the original identity order, so retry never changes logical
        ordering or multi-output alignment.
        """
        if not ports or len(ports[0]) == 0:
            return ports, tuple([] for _ in range(self.num_outputs)), [], []
        columns = [_as_columns(port) for port in ports]
        try:
            if first_error is not None:
                raise first_error
            outputs = self._checked_outputs(_call_user(self.op, *columns))
            return ports, outputs, [], []
        except BadRecordError as exc:
            if exc.index is None:
                raise
            bad_index = int(exc.index)
            if bad_index < 0 or bad_index >= len(ports[0]):
                raise IndexError(f"bad record index out of range: {bad_index}") from exc
            good_ports = [_without_index(port, bad_index) for port in ports]
            clean, outputs, more, deferred = self._run_isolating(
                good_ports,
                force_inline=force_inline,
            )
            recovered_ports: Sequence[PortBatch] | None = None
            recovered_outputs: tuple[List[Any], ...] | None = None
            final_error = exc

            if exc.retryable:
                singleton = [port.take([bad_index]) for port in ports]
                if (
                    self.recovery.retry_timing is RetryTiming.DEFERRED
                    and not force_inline
                    and self.recovery.decide_record(
                        retryable=True,
                        attempt=0,
                    )
                    is RecordRecoveryAction.RETRY
                ):
                    deferred.insert(
                        0,
                        DeferredRecord(
                            token=uuid.uuid4().hex,
                            inputs=tuple(singleton),
                            failed_op=self.name,
                            error=str(exc),
                            target_rows=len(ports[0]),
                        ),
                    )
                    return clean, outputs, more, deferred
                attempt = 0
                while (
                    self.recovery.decide_record(
                        retryable=final_error.retryable,
                        attempt=attempt,
                    )
                    is RecordRecoveryAction.RETRY
                ):
                    try:
                        recovered_outputs = self._checked_outputs(
                            _call_user(
                                self.op,
                                *[_as_columns(port) for port in singleton],
                            )
                        )
                        recovered_ports = singleton
                        break
                    except BadRecordError as retry_error:
                        if retry_error.index is None:
                            raise
                        if int(retry_error.index) != 0:
                            raise IndexError(
                                "singleton retry reported a non-zero bad index"
                            ) from retry_error
                        final_error = retry_error
                        attempt += 1

            errors = list(more)
            if recovered_ports is None or recovered_outputs is None:
                errors.insert(
                    0,
                    self._trace_for(ports, bad_index, str(final_error)),
                )
                return clean, outputs, errors, deferred

            merged = self._merge_recovered(
                ports,
                clean,
                outputs,
                recovered_ports,
                recovered_outputs,
            )
            return (*merged, errors, deferred)

    def _checked_outputs(self, raw: Any) -> tuple[List[Any], ...]:
        return checked_output_lists(
            raw,
            expected=self.num_outputs,
            primitive="Map",
            name=self.name,
        )

    def _merge_recovered(
        self,
        original: Sequence[PortBatch],
        clean: Sequence[PortBatch],
        clean_outputs: tuple[List[Any], ...],
        recovered: Sequence[PortBatch],
        recovered_outputs: tuple[List[Any], ...],
    ) -> tuple[Sequence[PortBatch], tuple[List[Any], ...]]:
        survivor_ids = set(clean[0].record_ids) | set(recovered[0].record_ids)
        order = [
            index
            for index, record_id in enumerate(original[0].record_ids)
            if record_id in survivor_ids
        ]
        merged_ports = [port.take(order) for port in original]
        output_maps = [
            {
                **dict(zip(clean[0].record_ids, clean_values)),
                **dict(zip(recovered[0].record_ids, recovered_values)),
            }
            for clean_values, recovered_values in zip(
                clean_outputs,
                recovered_outputs,
            )
        ]
        merged_outputs = tuple(
            [values[record_id] for record_id in merged_ports[0].record_ids]
            for values in output_maps
        )
        return merged_ports, merged_outputs

    def _run_with_bad_index(
        self,
        ports: Sequence[PortBatch],
        exc: BadRecordError,
        *,
        force_inline: bool = False,
    ) -> tuple[
        PortBatch | tuple[PortBatch, ...],
        tuple[DeferredRecord, ...],
    ]:
        bad_index = int(exc.index)
        if bad_index < 0 or bad_index >= len(ports[0]):
            raise IndexError(f"bad record index out of range: {bad_index}") from exc
        clean, outputs, errors, deferred = self._run_isolating(
            ports,
            first_error=exc,
            force_inline=force_inline,
        )
        return (
            self._make_outputs(self._merge_lineage(clean), tuple(outputs), errors=errors),
            tuple(deferred),
        )


class Filter(BoundPrimitive):
    """Record-dropping logical operator that preserves kept identities."""

    def __init__(
        self,
        op_cls: Any,
        *args: Any,
        name: str | None = None,
        workers: WorkerPoolSpec | None = None,
        recovery: RecoveryPolicy | None = None,
        **kwargs: Any,
    ) -> None:
        self._binding = PrimitiveBinding.create(
            op_cls,
            tuple(args),
            kwargs,
            name=name,
            num_outputs=1,
            workers=workers,
            recovery=recovery,
        )

    def __call__(
        self,
        *ports: PortBatch | TracePort,
    ) -> PortBatch | TracePort | tuple[PortBatch, ...] | tuple[TracePort, ...]:
        symbolic = ensure_trace_ports(ports)
        if symbolic is not None:
            self._binding.require_compilable("Filter")
            grain = require_same_trace_grain(f"Filter '{self.name}'", symbolic)
            return symbolic[0].tracer.add_node(
                name=self.name,
                inputs=symbolic,
                operation=FilterOp(self.factory_spec),
                output_grains=(grain,) * len(symbolic),
                relations=tuple(SubsetOf(port.ref) for port in symbolic),
                recovery=self.recovery,
                workers=self.workers,
            )

        aligned = _align_by_identity(ports)
        mask = normalize_mask(
            _call_user(self.op, *[_as_columns(port) for port in aligned]),
            len(aligned[0]),
        )
        kept = [index for index, keep in enumerate(mask) if keep]
        outputs = PortBatchBuilder.filtered(
            aligned,
            kept,
            name=self.name,
            op_name=self.name,
            preserve_names=True,
        )
        return outputs[0] if len(outputs) == 1 else outputs


class Select(BoundPrimitive):
    """High-level score/annotate-then-filter API lowered to Map + Filter + Project."""

    def __init__(
        self,
        op_cls: Any,
        *args: Any,
        name: str | None = None,
        num_annotations: int = 1,
        workers: WorkerPoolSpec | None = None,
        recovery: RecoveryPolicy | None = None,
        **kwargs: Any,
    ) -> None:
        if (
            isinstance(num_annotations, bool)
            or not isinstance(num_annotations, int)
            or num_annotations < 0
        ):
            raise ValueError("num_annotations must be a non-negative integer")
        self.num_annotations = num_annotations
        self._binding = PrimitiveBinding.create(
            op_cls,
            tuple(args),
            kwargs,
            name=name,
            num_outputs=1 + num_annotations,
            workers=workers,
            recovery=recovery,
        )

    def __call__(
        self,
        *ports: PortBatch | TracePort,
    ) -> PortBatch | TracePort | tuple[PortBatch, ...] | tuple[TracePort, ...]:
        symbolic = ensure_trace_ports(ports)
        if symbolic is not None:
            self._binding.require_compilable("Select")
            grain = require_same_trace_grain(f"Select '{self.name}'", symbolic)
            annotate = symbolic[0].tracer.add_node(
                name=f"{self.name}__map",
                inputs=symbolic,
                operation=MapOp(self.factory_spec),
                output_grains=(grain,) * (1 + self.num_annotations),
                relations=(SameAs(symbolic[0].ref),)
                * (1 + self.num_annotations),
                recovery=self.recovery,
                workers=self.workers,
            )
            annotate_ports = as_tuple(annotate)
            filtered_sources = (*symbolic, *annotate_ports[1:])
            filtered = symbolic[0].tracer.add_node(
                name=f"{self.name}__filter",
                inputs=(*symbolic, *annotate_ports),
                operation=FilterByMaskOp(mask_input=len(symbolic)),
                output_grains=(grain,) * len(filtered_sources),
                relations=tuple(
                    SubsetOf(port.ref) for port in filtered_sources
                ),
            )
            return filtered

        aligned = _align_by_identity(ports)
        raw_outputs = checked_output_lists(
            _call_user(self.op, *[_as_columns(port) for port in aligned]),
            expected=1 + self.num_annotations,
            primitive="Select",
            name=self.name,
        )
        annotated = tuple(
            aligned[0].with_values(
                values,
                name=output_name(
                    f"{self.name}__map",
                    index,
                    len(raw_outputs),
                ),
                op_name=f"{self.name}__map",
            )
            for index, values in enumerate(raw_outputs)
        )
        outputs = select_filter_outputs(
            (*aligned, *annotated),
            mask_index=len(aligned),
            output_count=len(aligned) + self.num_annotations,
            op_name=f"{self.name}__filter",
        )
        return outputs[0] if len(outputs) == 1 else outputs


__all__ = ["Filter", "Map", "Select"]
