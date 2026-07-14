"""Canonical primitive output construction and lineage transforms.

Purpose: keep PortBatch's parallel metadata columns aligned and make each
relation family's metadata policy explicit.  This module does not infer
cardinality or parent relations; callers must provide validated evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ._utils import (
    lineage_union,
    normalize_mask,
    output_name,
    take_with_lineage,
)
from ..data.batch import ErrorTrace, ParentRef, PortBatch


def _merge_consistent(
    target: dict[str, Any],
    source: Mapping[str, Any],
    *,
    field: str,
) -> None:
    for key, value in source.items():
        if key in target and target[key] != value:
            raise ValueError(
                f"conflicting {field} for '{key}': "
                f"{target[key]!r} != {value!r}"
            )
        target[key] = value


def dedupe_errors(errors: Sequence[ErrorTrace]) -> list[ErrorTrace]:
    result: list[ErrorTrace] = []
    for error in errors:
        if error not in result:
            result.append(error)
    return result


def merge_aligned_inputs(ports: Sequence[PortBatch]) -> PortBatch:
    """Merge metadata for already identity-aligned same-grain inputs."""
    base = ports[0]
    if len(ports) == 1:
        return base

    ancestors: list[dict[str, str]] = []
    ancestor_display: list[dict[str, str]] = []
    ordinals: list[dict[str, int]] = []
    lineage: list[tuple[str, ...]] = []
    relations: list[tuple[ParentRef, ...]] = []
    any_relations = any(port.relations for port in ports)
    for index in range(len(base)):
        item_ancestors: dict[str, str] = {}
        item_display: dict[str, str] = {}
        item_ordinals: dict[str, int] = {}
        item_relations: list[ParentRef] = []
        for port in ports:
            _merge_consistent(
                item_ancestors,
                port.ancestors[index],
                field="ancestor identity",
            )
            _merge_consistent(
                item_display,
                port.ancestor_display[index],
                field="ancestor display",
            )
            _merge_consistent(
                item_ordinals,
                port.ordinals[index],
                field="ordinal",
            )
            if port.relations:
                for ref in port.relations[index]:
                    if ref not in item_relations:
                        item_relations.append(ref)
        ancestors.append(item_ancestors)
        ancestor_display.append(item_display)
        ordinals.append(item_ordinals)
        lineage.append(lineage_union([port.lineage[index] for port in ports]))
        if any_relations:
            relations.append(tuple(item_relations))

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
        errors=dedupe_errors(
            [error for port in ports for error in port.errors]
        ),
    )


@dataclass(frozen=True)
class RelationOutput:
    value: Any
    record_id: str
    display_key: str
    ancestors: Mapping[str, str]
    ancestor_display: Mapping[str, str]
    ordinals: Mapping[str, int]
    lineage: tuple[str, ...]
    parents: tuple[ParentRef, ...]


class PortBatchBuilder:
    """Only primitive-level PortBatch materialization entry point."""

    @staticmethod
    def preserved(
        base: PortBatch,
        outputs: Sequence[Sequence[Any]],
        *,
        name: str,
        op_name: str,
        errors: Sequence[ErrorTrace] = (),
        preserve_name: bool = False,
    ) -> tuple[PortBatch, ...]:
        result: list[PortBatch] = []
        count = len(outputs)
        merged_errors = dedupe_errors([*base.errors, *errors])
        for index, values in enumerate(outputs):
            batch = base.with_values(
                list(values),
                name=base.name if preserve_name else output_name(name, index, count),
                op_name=op_name,
            )
            batch.errors = list(merged_errors)
            result.append(batch)
        return tuple(result)

    @staticmethod
    def filtered(
        ports: Sequence[PortBatch],
        indices: Sequence[int],
        *,
        name: str,
        op_name: str,
        preserve_names: bool = False,
    ) -> tuple[PortBatch, ...]:
        return tuple(
            take_with_lineage(
                port,
                indices,
                name=(
                    port.name
                    if preserve_names
                    else output_name(name, index, len(ports))
                ),
                op_name=op_name,
            )
            for index, port in enumerate(ports)
        )

    @staticmethod
    def expanded(
        parent: PortBatch,
        group_outputs: Sequence[Sequence[Sequence[Any]]],
        *,
        grain: str,
        op_name: str,
        child_label: str,
    ) -> tuple[PortBatch, ...]:
        batches: list[PortBatch] = []
        for groups in group_outputs:
            values: list[Any] = []
            record_ids: list[str] = []
            display_keys: list[str] = []
            ancestors: list[dict[str, str]] = []
            ancestor_display: list[dict[str, str]] = []
            ordinals: list[dict[str, int]] = []
            lineage: list[tuple[str, ...]] = []
            for parent_index, group in enumerate(groups):
                parent_id = parent.record_ids[parent_index]
                parent_key = parent.display_keys[parent_index]
                for child_index, value in enumerate(group):
                    values.append(value)
                    record_ids.append(f"{op_name}:{parent_id}:{child_index}")
                    display_keys.append(
                        f"{parent_key}/{child_label}={child_index}"
                    )
                    child_ancestors = dict(parent.ancestors[parent_index])
                    child_ancestors[parent.name] = parent_id
                    ancestors.append(child_ancestors)
                    child_display = dict(parent.ancestor_display[parent_index])
                    child_display[parent.name] = parent_key
                    ancestor_display.append(child_display)
                    child_ordinals = dict(parent.ordinals[parent_index])
                    child_ordinals[parent.name] = child_index
                    ordinals.append(child_ordinals)
                    lineage.append((*parent.lineage[parent_index], op_name))
            batches.append(
                PortBatch(
                    name=grain,
                    values=values,
                    record_ids=record_ids,
                    display_keys=display_keys,
                    ancestors=ancestors,
                    ancestor_display=ancestor_display,
                    ordinals=ordinals,
                    lineage=lineage,
                    errors=list(parent.errors),
                )
            )
        return tuple(batches)

    @staticmethod
    def related(
        rows: Sequence[RelationOutput],
        *,
        name: str,
        errors: Sequence[ErrorTrace] = (),
    ) -> PortBatch:
        return PortBatch(
            name=name,
            values=[row.value for row in rows],
            record_ids=[row.record_id for row in rows],
            display_keys=[row.display_key for row in rows],
            ancestors=[dict(row.ancestors) for row in rows],
            ancestor_display=[dict(row.ancestor_display) for row in rows],
            ordinals=[dict(row.ordinals) for row in rows],
            lineage=[tuple(row.lineage) for row in rows],
            relations=[tuple(row.parents) for row in rows],
            errors=dedupe_errors(errors),
        )


def select_filter_outputs(
    inputs: Sequence[PortBatch],
    *,
    mask_index: int,
    output_count: int,
    op_name: str,
) -> tuple[PortBatch, ...]:
    """Canonical Select lowering shared by eager and compiled execution."""
    if mask_index < 0 or mask_index >= len(inputs):
        raise ValueError(f"SelectFilter mask index {mask_index} out of range")
    mask_port = inputs[mask_index]
    mask = normalize_mask(mask_port.values, len(mask_port))
    kept = [index for index, keep in enumerate(mask) if keep]
    data_ports = [port for index, port in enumerate(inputs) if index != mask_index]
    if len(data_ports) != output_count:
        raise ValueError(
            f"SelectFilter has {len(data_ports)} data ports for "
            f"{output_count} outputs"
        )
    return PortBatchBuilder.filtered(
        data_ports,
        kept,
        name=op_name,
        op_name=op_name,
        preserve_names=True,
    )
