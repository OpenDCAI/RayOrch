"""Expanding and reducing operator wrappers."""
from __future__ import annotations

from typing import Any, List, Sequence

from ._op_utils import LazyOp, op_name, op_ref, output_name
from .core import (
    ErrorTrace,
    Grouped,
    PortBatch,
    _align_by_identity,
    _as_columns,
    _call_user,
    _normalize_output_lists,
)
from .graph import (
    MissingChildPolicy,
    OperatorProperties,
    OperatorRecipe,
    PhysicalHints,
    RecoveryPolicy,
    SymbolicPort,
    ensure_symbolic_ports,
)


class Expand:
    """One parent record produces a child group."""

    def __init__(
        self,
        op_cls: Any,
        *args: Any,
        parent: int = 0,
        child_label: str | None = None,
        name: str | None = None,
        num_outputs: int = 1,
        properties: OperatorProperties | None = None,
        physical: PhysicalHints | None = None,
        recovery: RecoveryPolicy | None = None,
        **kwargs: Any,
    ) -> None:
        self.name = op_name(op_cls, name)
        self.parent = parent
        self.child_label = child_label or self.name
        self.num_outputs = max(1, int(num_outputs))
        self.properties = properties or OperatorProperties()
        self.physical = physical or PhysicalHints(prefer_rebatch=True)
        self.recovery = recovery or RecoveryPolicy()
        self.op_recipe = OperatorRecipe(
            cls_ref=op_ref(op_cls),
            args=tuple(args),
            kwargs=dict(kwargs),
            provenance={
                "parent": str(parent),
                "child_label": self.child_label,
            },
        )
        self._lazy_op = LazyOp(op_cls, tuple(args), dict(kwargs))

    @property
    def op(self) -> Any:
        return self._lazy_op.get()

    def __call__(
        self,
        *ports: PortBatch | SymbolicPort,
    ) -> PortBatch | SymbolicPort | tuple[PortBatch, ...] | tuple[SymbolicPort, ...]:
        if self.parent < 0 or self.parent >= len(ports):
            raise ValueError(f"parent input {self.parent} is out of range")
        symbolic = ensure_symbolic_ports(ports)
        if symbolic is not None:
            return symbolic[0].tracer.add_node(
                name=self.name,
                kind="EXPAND",
                inputs=symbolic,
                num_outputs=self.num_outputs,
                output_grain=self.name,
                parent_input=self.parent,
                op=self.op_recipe,
                properties=self.properties,
                physical=self.physical,
                recovery=self.recovery,
            )
        aligned = _align_by_identity(ports)
        parent_port = aligned[self.parent]
        outputs = _normalize_output_lists(
            _call_user(self.op, *[_as_columns(port) for port in aligned])
        )
        group_outputs = [self._normalize_groups(output, len(parent_port)) for output in outputs]
        self._validate_shared_groups(group_outputs)
        return self._make_outputs(parent_port, group_outputs)

    @staticmethod
    def _normalize_groups(output: Sequence[Any], parent_count: int) -> List[List[Any]]:
        if len(output) != parent_count:
            raise ValueError(
                f"expanded output has {len(output)} groups for {parent_count} parents"
            )
        groups: List[List[Any]] = []
        for group in output:
            if isinstance(group, list):
                groups.append(group)
            elif isinstance(group, tuple):
                groups.append(list(group))
            else:
                raise TypeError("expanded outputs must be nested lists or tuples")
        return groups

    @staticmethod
    def _validate_shared_groups(group_outputs: Sequence[Sequence[Sequence[Any]]]) -> None:
        if not group_outputs:
            return
        expected = [[len(group) for group in group_outputs[0]]]
        for output in group_outputs[1:]:
            current = [len(group) for group in output]
            if current != expected[0]:
                raise ValueError(
                    "multi-output Expand requires shared group lengths in the MVP"
                )

    def _make_outputs(
        self,
        parent_port: PortBatch,
        group_outputs: Sequence[Sequence[Sequence[Any]]],
    ) -> PortBatch | tuple[PortBatch, ...]:
        batches: List[PortBatch] = []
        count = len(group_outputs)
        for output_index, groups in enumerate(group_outputs):
            values: List[Any] = []
            record_ids: List[str] = []
            display_keys: List[str] = []
            ancestors: List[dict[str, str]] = []
            ancestor_display: List[dict[str, str]] = []
            ordinals: List[dict[str, int]] = []
            lineage: List[tuple[str, ...]] = []
            for parent_index, group in enumerate(groups):
                parent_id = parent_port.record_ids[parent_index]
                parent_key = parent_port.display_keys[parent_index]
                for child_index, value in enumerate(group):
                    child_id = f"{self.name}:{parent_id}:{child_index}"
                    child_key = f"{parent_key}/{self.child_label}={child_index}"
                    values.append(value)
                    record_ids.append(child_id)
                    display_keys.append(child_key)
                    child_ancestors = dict(parent_port.ancestors[parent_index])
                    child_ancestors[parent_port.name] = parent_id
                    ancestors.append(child_ancestors)
                    child_display = dict(parent_port.ancestor_display[parent_index])
                    child_display[parent_port.name] = parent_key
                    ancestor_display.append(child_display)
                    child_ordinals = dict(parent_port.ordinals[parent_index])
                    child_ordinals[parent_port.name] = child_index
                    ordinals.append(child_ordinals)
                    lineage.append(tuple((*parent_port.lineage[parent_index], self.name)))
            batches.append(
                PortBatch(
                    name=output_name(self.name, output_index, count),
                    values=values,
                    record_ids=record_ids,
                    display_keys=display_keys,
                    ancestors=ancestors,
                    ancestor_display=ancestor_display,
                    ordinals=ordinals,
                    lineage=lineage,
                    errors=list(parent_port.errors),
                )
            )
        return batches[0] if len(batches) == 1 else tuple(batches)


class Reduce:
    """Group descendants by an anchor port and return anchor-grain rows."""

    def __init__(
        self,
        op_cls: Any,
        *args: Any,
        name: str | None = None,
        num_outputs: int = 1,
        missing_child: "MissingChildPolicy | str" = MissingChildPolicy.FAIL_OPEN,
        properties: OperatorProperties | None = None,
        physical: PhysicalHints | None = None,
        recovery: RecoveryPolicy | None = None,
        **kwargs: Any,
    ):
        self.name = op_name(op_cls, name)
        self.num_outputs = max(1, int(num_outputs))
        # Recovery policy for a lost descendant: FAIL_OPEN assembles from whatever
        # survived (may be partial); FAIL_CLOSED suppresses the affected anchor's
        # output (the UDF is not called for it) and emits an anchor-grain error, so
        # a permanently-lost child cascades to a flagged, *not-written* result
        # instead of a silently-truncated one.
        self.missing = MissingChildPolicy(missing_child)
        self.properties = properties or OperatorProperties()
        self.physical = physical or PhysicalHints()
        self.recovery = recovery or RecoveryPolicy()
        self.op_recipe = OperatorRecipe(
            cls_ref=op_ref(op_cls),
            args=tuple(args),
            kwargs=dict(kwargs),
            provenance={"missing_child": self.missing.value},
        )
        self._lazy_op = LazyOp(op_cls, tuple(args), dict(kwargs))

    @property
    def op(self) -> Any:
        return self._lazy_op.get()

    def __call__(
        self,
        grouped: Grouped,
    ) -> PortBatch | SymbolicPort | tuple[PortBatch, ...] | tuple[SymbolicPort, ...]:
        if not isinstance(grouped, Grouped):
            raise TypeError("Reduce expects orch.group_by(anchor, *descendants)")
        anchor = grouped.anchor
        if isinstance(anchor, SymbolicPort):
            descendants = grouped.descendants
            if not all(isinstance(port, SymbolicPort) for port in descendants):
                raise TypeError("cannot mix symbolic and eager grouped ports")
            return anchor.tracer.add_node(
                name=self.name,
                kind="REDUCE",
                inputs=(anchor, *descendants),
                num_outputs=self.num_outputs,
                output_grain=anchor.grain,
                parent_input=0,
                grouped=True,
                op=self.op_recipe,
                properties=self.properties,
                physical=self.physical,
                recovery=self.recovery,
            )
        grouped_values: List[List[List[Any]]] = [
            self._groups_for(anchor, descendant)
            for descendant in grouped.descendants
        ]
        errors = list(anchor.errors)
        for descendant in grouped.descendants:
            errors.extend(descendant.errors)

        poisoned = self._poisoned_anchor_ids(anchor, errors)
        if poisoned:
            outputs, cascade = self._reduce_fail_closed(
                anchor, grouped_values, poisoned, errors
            )
            errors = errors + cascade
        else:
            outputs = _normalize_output_lists(
                _call_user(self.op, anchor.values, *grouped_values)
            )

        result: List[PortBatch] = []
        count = len(outputs)
        for output_index, values in enumerate(outputs):
            batch = anchor.with_values(
                values,
                name=output_name(self.name, output_index, count),
                op_name=self.name,
            )
            batch.errors.extend(errors)
            result.append(batch)
        return result[0] if len(result) == 1 else tuple(result)

    def _poisoned_anchor_ids(
        self,
        anchor: PortBatch,
        errors: Sequence[ErrorTrace],
    ) -> set[str]:
        """Anchor record ids that have a lost descendant (only under FAIL_CLOSED)."""
        if self.missing is not MissingChildPolicy.FAIL_CLOSED:
            return set()
        anchor_ids = set(anchor.record_ids)
        poisoned: set[str] = set()
        for err in errors:
            aid = err.ancestors.get(anchor.name)
            if aid in anchor_ids:
                poisoned.add(aid)
        return poisoned

    def _reduce_fail_closed(
        self,
        anchor: PortBatch,
        grouped_values: List[List[List[Any]]],
        poisoned: set[str],
        errors: Sequence[ErrorTrace],
    ) -> tuple[tuple[List[Any], ...], List[ErrorTrace]]:
        """Assemble only clean anchors; suppress (don't call the UDF for) poisoned ones.

        The poisoned anchors get a placeholder value and an anchor-grain error, so a
        permanently-lost child never yields a silently-truncated output document.
        """
        clean_idx = [
            i for i, rid in enumerate(anchor.record_ids) if rid not in poisoned
        ]
        clean_anchor_vals = [anchor.values[i] for i in clean_idx]
        clean_grouped = [[groups[i] for i in clean_idx] for groups in grouped_values]
        clean_outputs = _normalize_output_lists(
            _call_user(self.op, clean_anchor_vals, *clean_grouped)
        )
        count = len(clean_outputs)
        n = len(anchor.values)
        full: List[List[Any]] = [[None] * n for _ in range(count)]
        for out_i in range(count):
            for pos, ci in enumerate(clean_idx):
                full[out_i][ci] = clean_outputs[out_i][pos]

        cascade: List[ErrorTrace] = []
        for i, rid in enumerate(anchor.record_ids):
            if rid not in poisoned:
                continue
            lost = [
                err.logical_item
                for err in errors
                if err.ancestors.get(anchor.name) == rid
            ]
            placeholder = {
                "status": "incomplete",
                "reason": "descendant record(s) quarantined; output suppressed",
                "anchor": anchor.display_keys[i],
                "lost": lost,
            }
            for out_i in range(count):
                full[out_i][i] = placeholder
            cascade.append(
                ErrorTrace(
                    source_item=anchor.display_keys[i],
                    logical_item=anchor.display_keys[i],
                    failed_op=self.name,
                    grain=anchor.name,
                    upstream_path=(self.name,),
                    parent=None,
                    action="suppressed_incomplete",
                    error=f"{len(lost)} descendant record(s) lost: {lost}",
                    ancestors={anchor.name: rid},
                )
            )
        return tuple(full), cascade

    @staticmethod
    def _groups_for(anchor: PortBatch, descendant: PortBatch) -> List[List[Any]]:
        grouped: List[List[tuple[tuple[Any, ...], Any]]] = [
            [] for _ in anchor.values
        ]
        anchor_index = {
            record_id: index for index, record_id in enumerate(anchor.record_ids)
        }
        for row_index, value in enumerate(descendant.values):
            parent_id = descendant.ancestors[row_index].get(anchor.name)
            if parent_id is None:
                continue
            if parent_id not in anchor_index:
                continue
            ordinal_items = list(descendant.ordinals[row_index].items())
            anchor_position = next(
                (
                    index
                    for index, (grain, _) in enumerate(ordinal_items)
                    if grain == anchor.name
                ),
                None,
            )
            if anchor_position is None:
                ordinal_path: tuple[Any, ...] = (row_index,)
            else:
                # Nested Expand contributes one ordinal per level.  Sorting only
                # by the anchor's first child index restores page order but leaves
                # blocks within a page dependent on physical shard completion.
                ordinal_path = tuple(
                    ordinal for _, ordinal in ordinal_items[anchor_position:]
                )
            # A relation can legally emit multiple records at the same hierarchy
            # position.  Its content-addressed id is a deterministic final tie.
            order_key = (*ordinal_path, descendant.record_ids[row_index])
            grouped[anchor_index[parent_id]].append((order_key, value))
        return [
            [value for _, value in sorted(items, key=lambda item: item[0])]
            for items in grouped
        ]


__all__ = ["Expand", "Reduce"]
