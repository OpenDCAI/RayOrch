"""Record-preserving and record-filtering operator wrappers."""
from __future__ import annotations

from typing import Any, List, Sequence

from rayorch.runtime import BadRecordError

from ._op_utils import (
    LazyOp,
    as_tuple,
    check_symbolic_same_grain,
    normalize_mask,
    op_name,
    op_ref,
    output_name,
    take_with_lineage,
)
from .core import (
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
        **kwargs: Any,
    ):
        self.name = op_name(op_cls, name)
        self.num_outputs = max(1, int(num_outputs))
        self.properties = properties or OperatorProperties()
        self.physical = physical or PhysicalHints()
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
            )
        aligned = _align_by_identity(ports)
        columns = [_as_columns(port) for port in aligned]
        try:
            outputs = _normalize_output_lists(_call_user(self.op, *columns))
            return self._make_outputs(aligned[0], outputs)
        except BadRecordError as exc:
            if exc.index is None:
                raise
            return self._run_with_bad_index(aligned, exc)

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
    ) -> tuple[Sequence[PortBatch], tuple[List[Any], ...], List[ErrorTrace]]:
        """Re-run the remaining rows *as one batch*; peel off any further bad rows.

        Dropping a known-bad row and re-running the rest as a single batch keeps the
        recompute footprint minimal (each healthy row runs once) without serializing
        into one-row-at-a-time calls. If the batch surfaces another bad row we recurse
        to isolate it too, so multiple poison rows still degrade gracefully.
        """
        columns = [_as_columns(port) for port in ports]
        try:
            outputs = _normalize_output_lists(_call_user(self.op, *columns))
            return ports, outputs, []
        except BadRecordError as exc:
            if exc.index is None:
                raise
            bad_index = int(exc.index)
            if bad_index < 0 or bad_index >= len(ports[0]):
                raise IndexError(f"bad record index out of range: {bad_index}") from exc
            trace = self._trace_for(ports, bad_index, str(exc))
            good_ports = [_without_index(port, bad_index) for port in ports]
            clean, outputs, more = self._run_isolating(good_ports)
            return clean, outputs, [trace, *more]

    def _run_with_bad_index(
        self,
        ports: Sequence[PortBatch],
        exc: BadRecordError,
    ) -> PortBatch | tuple[PortBatch, ...]:
        bad_index = int(exc.index)
        if bad_index < 0 or bad_index >= len(ports[0]):
            raise IndexError(f"bad record index out of range: {bad_index}") from exc
        trace = self._trace_for(ports, bad_index, str(exc))
        good_ports = [_without_index(port, bad_index) for port in ports]
        clean, outputs, more = self._run_isolating(good_ports)
        return self._make_outputs(clean[0], tuple(outputs), errors=[trace, *more])


class Filter:
    """Record-dropping logical operator that preserves kept identities."""

    def __init__(
        self,
        op_cls: Any,
        *args: Any,
        name: str | None = None,
        properties: OperatorProperties | None = None,
        physical: PhysicalHints | None = None,
        **kwargs: Any,
    ) -> None:
        self.name = op_name(op_cls, name)
        self.properties = properties or OperatorProperties()
        self.physical = physical or PhysicalHints()
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
        **kwargs: Any,
    ) -> None:
        self.name = op_name(op_cls, name)
        self.num_annotations = max(0, int(num_annotations))
        self.properties = properties or OperatorProperties()
        self.physical = physical or PhysicalHints()
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
