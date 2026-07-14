"""Many-to-many relation operator wrapper.

Three ways to declare an M:N relation, from cheapest to most flexible:

* ``on={role: field}`` — declarative equi-join. Records across roles whose
  extracted key matches are related. Pure data, serializable, optimizer-friendly.
  Covers the vast majority of real relations.
* ``relation_adapter="pkg.mod:fn"`` — by-reference imperative adapter for
  arbitrary (non equi-join) relations. Resolved at execute time, speaks only
  local indices, never internal ids.
* ``relation_fn=<callable>`` — live-object adapter (kept for local eager use;
  not serializable, so prefer ``relation_adapter`` for round-tripping IR).
"""
from __future__ import annotations

from importlib import import_module
from typing import Any, Callable, List, Mapping, Sequence

from ._utils import (
    lineage_union,
)
from ..data.batch import ParentRef, PortBatch, _as_columns, _call_user
from ._binding import BoundPrimitive, PrimitiveBinding
from .output import PortBatchBuilder, RelationOutput
from ..ir.model import (
    NodeKind,
    OperatorProperties,
    PhysicalHints,
    RecoveryPolicy,
    SymbolicPort,
    ensure_symbolic_ports,
)


def _resolve_adapter(ref: str) -> Callable[[Any], Any]:
    """Resolve a ``pkg.mod:fn`` dotted path into a callable."""
    if ":" not in ref:
        raise ValueError(
            f"relation_adapter '{ref}' must be a 'pkg.mod:fn' dotted path"
        )
    module_path, _, attr = ref.partition(":")
    module = import_module(module_path)
    fn = getattr(module, attr)
    if not callable(fn):
        raise TypeError(f"relation_adapter '{ref}' is not callable")
    return fn


def _extract_key(value: Any, field: str | Callable[[Any], Any]) -> Any:
    if callable(field):
        return field(value)
    if isinstance(value, Mapping):
        return value[field]
    return getattr(value, field)


class Relate(BoundPrimitive):
    """Arbitrary invocation-local relation primitive."""

    def __init__(
        self,
        op_cls: Any,
        *args: Any,
        name: str | None = None,
        output_grain: str | None = None,
        roles: Sequence[str] = (),
        on: Mapping[str, str | Callable[[Any], Any]] | None = None,
        relation_adapter: str | None = None,
        relation_fn: Any | None = None,
        num_outputs: int = 1,
        properties: OperatorProperties | None = None,
        physical: PhysicalHints | None = None,
        recovery: RecoveryPolicy | None = None,
        **kwargs: Any,
    ) -> None:
        if num_outputs != 1:
            raise ValueError(
                "Relate currently supports exactly one output; "
                "multi-output relation evidence is not defined"
            )
        resolved_name = name or getattr(op_cls, "__name__", type(op_cls).__name__)
        self.output_grain = output_grain or resolved_name
        self.on = dict(on) if on else None
        if self.on and not roles:
            roles = tuple(self.on.keys())
        self.roles = tuple(roles)
        self.relation_adapter = relation_adapter
        self.relation_fn = relation_fn
        provenance: dict[str, Any] = {}
        if self.on is not None:
            provenance["on"] = {
                role: (getattr(field, "__name__", "<callable>") if callable(field) else field)
                for role, field in self.on.items()
            }
        if relation_adapter is not None:
            provenance["relation_adapter"] = relation_adapter
        if relation_fn is not None:
            provenance["relation_fn"] = getattr(
                relation_fn, "__name__", type(relation_fn).__name__
            )
        self._binding = PrimitiveBinding.create(
            op_cls,
            tuple(args),
            kwargs,
            name=name,
            num_outputs=1,
            properties=properties,
            physical=physical,
            recovery=recovery,
            provenance=provenance,
        )

    def __call__(
        self,
        *ports: PortBatch | SymbolicPort,
    ) -> PortBatch | SymbolicPort | tuple[SymbolicPort, ...]:
        symbolic = ensure_symbolic_ports(ports)
        if symbolic is not None:
            self._binding.require_compilable("Relate")
            if self.on is not None and any(callable(field) for field in self.on.values()):
                raise TypeError(
                    "Relate compiled on= keys must be field names; "
                    "use relation_adapter='pkg.mod:fn' for custom extraction"
                )
            if self.relation_fn is not None:
                raise TypeError(
                    "Relate relation_fn is eager-only; compiled graphs require "
                    "relation_adapter='pkg.mod:fn'"
                )
            if self.roles and len(self.roles) != len(symbolic):
                raise ValueError("Relate roles must match the number of input ports")
            return symbolic[0].tracer.add_node(
                name=self.name,
                kind=NodeKind.RELATE,
                inputs=symbolic,
                num_outputs=self.num_outputs,
                output_grain=self.output_grain,
                op=self.op_recipe,
                properties=self.properties,
                physical=self.physical,
                recovery=self.recovery,
                relation_roles=self.roles,
            )
        if self.on is not None:
            if len(self.roles) != len(ports):
                raise ValueError("Relate `on` roles must match the number of input ports")
            return self._make_key_join_batch(ports)
        relation_fn = self.relation_fn
        if relation_fn is None and self.relation_adapter is not None:
            relation_fn = _resolve_adapter(self.relation_adapter)
        if relation_fn is None:
            raise NotImplementedError(
                "Relate eager execution needs on=, relation_adapter, or relation_fn"
            )
        if self.roles and len(self.roles) != len(ports):
            raise ValueError("Relate roles must match the number of input ports")
        role_names = self.roles or tuple(port.name for port in ports)
        raw_values = _call_user(self.op, *[_as_columns(port) for port in ports])
        evidence = relation_fn(raw_values)
        return self._make_relation_batch(raw_values, evidence, ports, role_names)

    def _make_key_join_batch(self, ports: Sequence[PortBatch]) -> PortBatch:
        assert self.on is not None
        role_names = self.roles
        role_to_port = dict(zip(role_names, ports))

        # Index each role's records by its declared join key.
        key_to_indices: dict[str, dict[Any, List[int]]] = {}
        for role in role_names:
            port = role_to_port[role]
            field = self.on[role]
            index: dict[Any, List[int]] = {}
            for local_index, value in enumerate(port.values):
                key = _extract_key(value, field)
                index.setdefault(key, []).append(local_index)
            key_to_indices[role] = index

        first_role = role_names[0]
        other_roles = role_names[1:]

        rows: List[RelationOutput] = []

        output_index = 0
        seen_keys: set[Any] = set()
        for base_value in role_to_port[first_role].values:
            key = _extract_key(base_value, self.on[first_role])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            if any(key not in key_to_indices[role] for role in other_roles):
                continue  # inner join: skip keys missing on any side
            # A key may occur more than once in *every* role.  Seed the product
            # with every first-role row as well; using only the first occurrence
            # silently degraded a true M:N join into 1:N for duplicate keys.
            combos = [
                [(first_role, base_index)]
                for base_index in key_to_indices[first_role][key]
            ]
            for role in other_roles:
                combos = [
                    combo + [(role, idx)]
                    for combo in combos
                    for idx in key_to_indices[role][key]
                ]
            for combo in combos:
                by_role = {role: role_to_port[role].values[idx] for role, idx in combo}
                value = _call_user(self.op, by_role) if self.op is not None else by_role

                merged_ancestors: dict[str, str] = {}
                merged_display: dict[str, str] = {}
                merged_ordinals: dict[str, int] = {}
                parent_refs: List[ParentRef] = []
                parent_lineage: List[tuple[str, ...]] = []
                key_parts: List[str] = []
                id_parts: List[str] = []
                for role, idx in combo:
                    port = role_to_port[role]
                    merged_ancestors.update(port.ancestors[idx])
                    merged_ancestors[port.name] = port.record_ids[idx]
                    merged_display.update(port.ancestor_display[idx])
                    merged_display[port.name] = port.display_keys[idx]
                    merged_ordinals.update(port.ordinals[idx])
                    parent_refs.append(
                        ParentRef(
                            role=role,
                            port=port.name,
                            record_id=port.record_ids[idx],
                            display_key=port.display_keys[idx],
                        )
                    )
                    parent_lineage.append(port.lineage[idx])
                    key_parts.append(f"{role}={port.display_keys[idx]}")
                    id_parts.append(f"{role}={port.record_ids[idx]}")

                # Content-addressed identity: a relation row is identified by its
                # matched parent tuple, not by emission order. Combos are distinct
                # by construction, so ids are unique AND invariant to upstream
                # reordering (see docs/todos/13-reordering-invariance-theorem.md).
                rows.append(
                    RelationOutput(
                        value=value,
                        record_id=f"{self.name}:{'|'.join(id_parts)}",
                        display_key="/".join(key_parts),
                        ancestors=merged_ancestors,
                        ancestor_display=merged_display,
                        ordinals=merged_ordinals,
                        lineage=lineage_union(parent_lineage, self.name),
                        parents=tuple(parent_refs),
                    )
                )
                output_index += 1

        return PortBatchBuilder.related(
            rows,
            name=self.name,
            errors=[error for port in ports for error in port.errors],
        )

    def _make_relation_batch(
        self,
        raw_values: Any,
        evidence: Any,
        ports: Sequence[PortBatch],
        role_names: Sequence[str],
    ) -> PortBatch:
        if not isinstance(evidence, list):
            raise TypeError("Relate relation_fn must return a list")

        rows: List[RelationOutput] = []
        role_to_port = dict(zip(role_names, ports))

        seen_ids: set[str] = set()
        for output_index, item in enumerate(evidence):
            if not isinstance(item, tuple) or len(item) not in (2, 3):
                raise TypeError(
                    "Relate relation_fn items must be "
                    "(value, {role: local_index}) or "
                    "(value, {role: local_index}, stable_key)"
                )
            value, parent_indexes = item[:2]
            stable_key = item[2] if len(item) == 3 else None
            if not isinstance(parent_indexes, dict):
                raise TypeError("Relate parent refs must be a dict of role -> index")

            parent_refs: List[ParentRef] = []
            item_ancestors: dict[str, str] = {}
            item_display: dict[str, str] = {}
            item_ordinals: dict[str, int] = {}
            parent_lineage: List[tuple[str, ...]] = []
            id_parts: List[str] = []
            for role, local_index in parent_indexes.items():
                if role not in role_to_port:
                    raise ValueError(f"Relate parent role '{role}' is not declared")
                if not isinstance(local_index, int):
                    raise TypeError("Relate parent indexes must be integers")
                parent_port = role_to_port[role]
                if local_index < 0 or local_index >= len(parent_port):
                    raise IndexError(
                        f"Relate parent index {local_index} for role '{role}' "
                        "is out of range"
                    )
                parent_refs.append(
                    ParentRef(
                        role=role,
                        port=parent_port.name,
                        record_id=parent_port.record_ids[local_index],
                        display_key=parent_port.display_keys[local_index],
                    )
                )
                item_ancestors.update(parent_port.ancestors[local_index])
                item_ancestors[parent_port.name] = parent_port.record_ids[local_index]
                item_display.update(parent_port.ancestor_display[local_index])
                item_display[parent_port.name] = parent_port.display_keys[local_index]
                item_ordinals.update(parent_port.ordinals[local_index])
                parent_lineage.append(parent_port.lineage[local_index])
                id_parts.append(f"{role}={parent_port.record_ids[local_index]}")

            record_id = f"{self.name}:{'|'.join(id_parts)}"
            if stable_key is not None:
                record_id += f"|key={stable_key}"
            if record_id in seen_ids:
                raise ValueError(
                    "Relate adapter emitted duplicate parent evidence; "
                    "provide a stable_key as the third tuple item"
                )
            seen_ids.add(record_id)
            rows.append(
                RelationOutput(
                    value=value,
                    record_id=record_id,
                    display_key=(
                        "/".join(
                            f"{ref.role}={ref.display_key}" for ref in parent_refs
                        )
                        or f"{self.name}:{output_index}"
                    ),
                    ancestors=item_ancestors,
                    ancestor_display=item_display,
                    ordinals=item_ordinals,
                    lineage=lineage_union(parent_lineage, self.name),
                    parents=tuple(parent_refs),
                )
            )

        if isinstance(raw_values, list) and len(raw_values) != len(rows):
            raise ValueError(
                "Relate relation_fn output length must match raw value length"
            )
        return PortBatchBuilder.related(
            rows,
            name=self.name,
            errors=[error for port in ports for error in port.errors],
        )


__all__ = ["Relate"]
