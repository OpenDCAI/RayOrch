"""Record-preserving and record-filtering operator wrappers."""
from __future__ import annotations

from typing import Any, List, Sequence
import uuid

from rayorch.runtime import BadRecordError

from ._op_utils import (
    LazyOp,
    as_tuple,
    check_symbolic_same_grain,
    lineage_union,
    normalize_mask,
    op_name,
    op_ref,
    output_name,
    take_with_lineage,
)
from .core import (
    DeferredRecord,
    ErrorTrace,
    PortBatch,
    _align_by_identity,
    _as_columns,
    _call_user,
    _normalize_output_lists,
    _parent_display,
    _source_item,
    _without_index,
)
from .graph import (
    NodeKind,
    OperatorProperties,
    OperatorRecipe,
    PhysicalHints,
    RecordRecoveryAction,
    RecoveryPolicy,
    RetryTiming,
    SymbolicPort,
    ensure_symbolic_ports,
)


class Map:
    """Record-preserving logical operator."""

    def __init__(
        self,
        op_cls: Any,
        *args: Any,
        name: str | None = None,
        num_outputs: int = 1,
        properties: OperatorProperties | None = None,
        physical: PhysicalHints | None = None,
        recovery: RecoveryPolicy | None = None,
        **kwargs: Any,
    ):
        self.name = op_name(op_cls, name)
        self.num_outputs = max(1, int(num_outputs))
        self.properties = properties or OperatorProperties()
        self.physical = physical or PhysicalHints()
        self.recovery = recovery or RecoveryPolicy()
        self.op_recipe = OperatorRecipe(
            cls_ref=op_ref(op_cls),
            args=tuple(args),
            kwargs=dict(kwargs),
        )
        self._lazy_op = LazyOp(op_cls, tuple(args), dict(kwargs))

    @property
    def op(self) -> Any:
        return self._lazy_op.get()

    def __call__(
        self,
        *ports: PortBatch | SymbolicPort,
    ) -> PortBatch | SymbolicPort | tuple[PortBatch, ...] | tuple[SymbolicPort, ...]:
        symbolic = ensure_symbolic_ports(ports)
        if symbolic is not None:
            grain = check_symbolic_same_grain(f"Map '{self.name}'", symbolic)
            return symbolic[0].tracer.add_node(
                name=self.name,
                kind=NodeKind.MAP,
                inputs=symbolic,
                num_outputs=self.num_outputs,
                output_grain=grain,
                op=self.op_recipe,
                properties=self.properties,
                physical=self.physical,
                recovery=self.recovery,
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
        result: List[PortBatch] = []
        count = len(outputs)
        for index, values in enumerate(outputs):
            batch = base.with_values(
                list(values),
                name=output_name(self.name, index, count),
                op_name=self.name,
            )
            batch.errors.extend(errors)
            result.append(batch)
        return result[0] if len(result) == 1 else tuple(result)

    @staticmethod
    def _merge_lineage(ports: Sequence[PortBatch]) -> PortBatch:
        """Merge metadata from every same-identity input of a diamond fan-in."""
        base = ports[0]
        if len(ports) == 1:
            return base

        ancestors: List[dict[str, str]] = []
        ancestor_display: List[dict[str, str]] = []
        ordinals: List[dict[str, int]] = []
        lineage: List[tuple[str, ...]] = []
        relations = []
        any_relations = any(port.relations for port in ports)
        for index in range(len(base)):
            item_ancestors: dict[str, str] = {}
            item_display: dict[str, str] = {}
            item_ordinals: dict[str, int] = {}
            item_relations = []
            for port in ports:
                item_ancestors.update(port.ancestors[index])
                item_display.update(port.ancestor_display[index])
                item_ordinals.update(port.ordinals[index])
                if port.relations:
                    for ref in port.relations[index]:
                        if ref not in item_relations:
                            item_relations.append(ref)
            ancestors.append(item_ancestors)
            ancestor_display.append(item_display)
            ordinals.append(item_ordinals)
            # ``with_values`` appends this Map's own op name afterwards.
            lineage.append(lineage_union([port.lineage[index] for port in ports]))
            if any_relations:
                relations.append(tuple(item_relations))

        errors = []
        for port in ports:
            for error in port.errors:
                if error not in errors:
                    errors.append(error)
        return PortBatch(
            name=base.name,
            values=list(base.values),
            record_ids=list(base.record_ids),
            display_keys=list(base.display_keys),
            ancestors=ancestors,
            ancestor_display=ancestor_display,
            ordinals=ordinals,
            lineage=lineage,
            relations=relations,
            errors=errors,
        )

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
        outputs = _normalize_output_lists(raw)
        if len(outputs) != self.num_outputs:
            raise ValueError(
                f"Map '{self.name}' expected {self.num_outputs} outputs, "
                f"got {len(outputs)}"
            )
        return outputs

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


class Filter:
    """Record-dropping logical operator that preserves kept identities."""

    def __init__(
        self,
        op_cls: Any,
        *args: Any,
        name: str | None = None,
        properties: OperatorProperties | None = None,
        physical: PhysicalHints | None = None,
        recovery: RecoveryPolicy | None = None,
        **kwargs: Any,
    ) -> None:
        self.name = op_name(op_cls, name)
        self.properties = properties or OperatorProperties()
        self.physical = physical or PhysicalHints()
        self.recovery = recovery or RecoveryPolicy()
        self.op_recipe = OperatorRecipe(
            cls_ref=op_ref(op_cls),
            args=tuple(args),
            kwargs=dict(kwargs),
        )
        self._lazy_op = LazyOp(op_cls, tuple(args), dict(kwargs))

    @property
    def op(self) -> Any:
        return self._lazy_op.get()

    def __call__(
        self,
        *ports: PortBatch | SymbolicPort,
    ) -> PortBatch | SymbolicPort | tuple[PortBatch, ...] | tuple[SymbolicPort, ...]:
        symbolic = ensure_symbolic_ports(ports)
        if symbolic is not None:
            grain = check_symbolic_same_grain(f"Filter '{self.name}'", symbolic)
            return symbolic[0].tracer.add_node(
                name=self.name,
                kind=NodeKind.FILTER,
                inputs=symbolic,
                num_outputs=len(symbolic),
                output_grain=grain,
                op=self.op_recipe,
                properties=self.properties,
                physical=self.physical,
                recovery=self.recovery,
            )

        aligned = _align_by_identity(ports)
        mask = normalize_mask(
            _call_user(self.op, *[_as_columns(port) for port in aligned]),
            len(aligned[0]),
        )
        kept = [index for index, keep in enumerate(mask) if keep]
        outputs = tuple(
            take_with_lineage(
                port,
                kept,
                name=output_name(self.name, index, len(aligned)),
                op_name=self.name,
            )
            for index, port in enumerate(aligned)
        )
        return outputs[0] if len(outputs) == 1 else outputs


class Select:
    """High-level score/annotate-then-filter API lowered to Map + Filter + Project."""

    def __init__(
        self,
        op_cls: Any,
        *args: Any,
        name: str | None = None,
        num_annotations: int = 1,
        properties: OperatorProperties | None = None,
        physical: PhysicalHints | None = None,
        recovery: RecoveryPolicy | None = None,
        **kwargs: Any,
    ) -> None:
        self.name = op_name(op_cls, name)
        self.num_annotations = max(0, int(num_annotations))
        self.properties = properties or OperatorProperties()
        self.physical = physical or PhysicalHints()
        self.recovery = recovery or RecoveryPolicy()
        self.op_recipe = OperatorRecipe(
            cls_ref=op_ref(op_cls),
            args=tuple(args),
            kwargs=dict(kwargs),
        )
        self._lazy_op = LazyOp(op_cls, tuple(args), dict(kwargs))

    @property
    def op(self) -> Any:
        return self._lazy_op.get()

    def __call__(
        self,
        *ports: PortBatch | SymbolicPort,
    ) -> PortBatch | SymbolicPort | tuple[PortBatch, ...] | tuple[SymbolicPort, ...]:
        symbolic = ensure_symbolic_ports(ports)
        if symbolic is not None:
            grain = check_symbolic_same_grain(f"Select '{self.name}'", symbolic)
            annotate = symbolic[0].tracer.add_node(
                name=f"{self.name}__map",
                kind=NodeKind.MAP,
                inputs=symbolic,
                num_outputs=1 + self.num_annotations,
                output_grain=grain,
                op=self.op_recipe,
                properties=self.properties,
                physical=self.physical,
                recovery=self.recovery,
            )
            annotate_ports = as_tuple(annotate)
            filtered = symbolic[0].tracer.add_node(
                name=f"{self.name}__filter",
                kind=NodeKind.FILTER,
                inputs=(*symbolic, *annotate_ports),
                num_outputs=len(symbolic) + self.num_annotations,
                output_grain=grain,
                op=OperatorRecipe(
                    cls_ref="rayorch.experimental.multigrain.SelectFilter",
                    provenance={"mask_input": str(len(symbolic))},
                ),
            )
            filtered_ports = as_tuple(filtered)
            return symbolic[0].tracer.add_node(
                name=f"{self.name}__project",
                kind=NodeKind.PROJECT,
                inputs=filtered_ports,
                num_outputs=len(filtered_ports),
                output_grain=grain,
                op=OperatorRecipe(cls_ref="rayorch.experimental.multigrain.Project"),
            )

        aligned = _align_by_identity(ports)
        raw_outputs = as_tuple(_call_user(self.op, *[_as_columns(port) for port in aligned]))
        if not raw_outputs:
            raise ValueError("Select operator must return at least a mask")
        mask = normalize_mask(raw_outputs[0], len(aligned[0]))
        annotations = raw_outputs[1:]
        if len(annotations) != self.num_annotations:
            raise ValueError(
                f"Select expected {self.num_annotations} annotation outputs, "
                f"got {len(annotations)}"
            )
        kept = [index for index, keep in enumerate(mask) if keep]
        output_batches: list[PortBatch] = []
        total = len(aligned) + len(annotations)
        for index, port in enumerate(aligned):
            output_batches.append(
                take_with_lineage(
                    port,
                    kept,
                    name=output_name(self.name, index, total),
                    op_name=self.name,
                )
            )
        for offset, values in enumerate(annotations):
            annotated = aligned[0].with_values(
                values,
                name=output_name(f"{self.name}_annotation", offset, len(annotations)),
                op_name=f"{self.name}__map",
            )
            output_batches.append(
                take_with_lineage(
                    annotated,
                    kept,
                    name=output_name(self.name, len(aligned) + offset, total),
                    op_name=f"{self.name}__filter",
                )
            )
        return output_batches[0] if len(output_batches) == 1 else tuple(output_batches)


__all__ = ["Filter", "Map", "Select"]
