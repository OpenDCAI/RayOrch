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

import hashlib
from typing import Any, Callable, List, Mapping, Sequence

from ._utils import (
    lineage_union,
)
from ..data.batch import (
    IdentityDomain,
    ParentRef,
    PortBatch,
    _as_columns,
    _call_user,
)
from ._binding import BoundPrimitive, PrimitiveBinding
from .output import PortBatchBuilder, RelationOutput
from ..ir.operations import (
    KeyJoinSpec,
    RelateOp,
    RelationAdapterSpec,
    resolve_relation_adapter,
)
from ..ir.policy import RecoveryPolicy, WorkerPoolSpec
from ..ir.relations import RelatedFrom
from ..tracing import TracePort, ensure_trace_ports


def _extract_key(value: Any, field: str | Callable[[Any], Any]) -> Any:
    if callable(field):
        return field(value)
    if isinstance(value, Mapping):
        return value[field]
    return getattr(value, field)


def _relation_record_id(
    name: str,
    parents: Sequence[ParentRef],
    stable_key: Any = None,
) -> str:
    evidence = tuple(
        (parent.role, parent.identity_domain.token, parent.record_id)
        for parent in parents
    )
    digest = hashlib.sha256(repr((evidence, stable_key)).encode("utf-8")).hexdigest()
    return f"{name}:{digest}"


def _merge_relation_metadata(
    selected: Sequence[tuple[str, PortBatch, int]],
) -> tuple[
    dict[IdentityDomain, str],
    dict[str, str],
    dict[IdentityDomain, int],
    list[tuple[str, ...]],
]:
    """Keep only ancestry facts that agree across all role parents."""
    identity_values: dict[IdentityDomain, set[str]] = {}
    display_values: dict[str, set[str]] = {}
    ordinal_values: dict[IdentityDomain, set[int]] = {}
    lineages: list[tuple[str, ...]] = []
    for _, port, index in selected:
        if port.identity_domain is None:
            raise ValueError("Relate input has no identity domain")
        facts = dict(port.ancestors[index])
        facts[port.identity_domain] = port.record_ids[index]
        for domain, record_id in facts.items():
            identity_values.setdefault(domain, set()).add(record_id)
        displays = dict(port.ancestor_display[index])
        displays[port.name] = port.display_keys[index]
        for label, display in displays.items():
            display_values.setdefault(label, set()).add(display)
        for domain, ordinal in port.ordinals[index].items():
            ordinal_values.setdefault(domain, set()).add(ordinal)
        lineages.append(port.lineage[index])
    return (
        {
            domain: next(iter(values))
            for domain, values in identity_values.items()
            if len(values) == 1
        },
        {
            label: next(iter(values))
            for label, values in display_values.items()
            if len(values) == 1
        },
        {
            domain: next(iter(values))
            for domain, values in ordinal_values.items()
            if len(values) == 1
        },
        lineages,
    )


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
        workers: WorkerPoolSpec | None = None,
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
        if self.roles and (
            any(not role for role in self.roles)
            or len(set(self.roles)) != len(self.roles)
        ):
            raise ValueError("Relate roles must be unique non-empty names")
        if self.on is not None and set(self.on) != set(self.roles):
            raise ValueError("Relate `on` keys must match declared roles")
        self.relation_adapter = relation_adapter
        self.relation_fn = relation_fn
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
    ) -> PortBatch | TracePort | tuple[TracePort, ...]:
        symbolic = ensure_trace_ports(ports)
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
            role_names = self.roles or tuple(port.name for port in symbolic)
            if self.on is not None:
                matcher = KeyJoinSpec(
                    tuple(str(self.on[role]) for role in role_names)
                )
            elif self.relation_adapter is not None:
                matcher = RelationAdapterSpec(self.relation_adapter)
            else:
                raise TypeError(
                    "Relate compiled graphs require on= field names or "
                    "relation_adapter='pkg.mod:fn'"
                )
            return symbolic[0].tracer.add_node(
                name=self.name,
                inputs=symbolic,
                operation=RelateOp(self.factory_spec, matcher),
                output_grains=(self.output_grain,),
                relations=(
                    RelatedFrom(role_names),
                ),
                recovery=self.recovery,
                workers=self.workers,
            )
        if self.on is not None:
            if len(self.roles) != len(ports):
                raise ValueError("Relate `on` roles must match the number of input ports")
            return self._make_key_join_batch(ports)
        relation_fn = self.relation_fn
        if relation_fn is None and self.relation_adapter is not None:
            relation_fn = resolve_relation_adapter(
                RelationAdapterSpec(self.relation_adapter)
            )
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

                parent_refs: List[ParentRef] = []
                key_parts: List[str] = []
                selected: list[tuple[str, PortBatch, int]] = []
                for role, idx in combo:
                    port = role_to_port[role]
                    if port.identity_domain is None:
                        raise ValueError("Relate input has no identity domain")
                    selected.append((role, port, idx))
                    parent_refs.append(
                        ParentRef(
                            role=role,
                            port=port.name,
                            record_id=port.record_ids[idx],
                            display_key=port.display_keys[idx],
                            identity_domain=port.identity_domain,
                        )
                    )
                    key_parts.append(f"{role}={port.display_keys[idx]}")
                (
                    merged_ancestors,
                    merged_display,
                    merged_ordinals,
                    parent_lineage,
                ) = _merge_relation_metadata(selected)

                # Content-addressed identity: a relation row is identified by its
                # matched parent tuple, not by emission order. Combos are distinct
                # by construction, so ids are unique AND invariant to upstream
                # reordering (see docs/todos/13-reordering-invariance-theorem.md).
                rows.append(
                    RelationOutput(
                        value=value,
                        record_id=_relation_record_id(self.name, parent_refs),
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
            name=self.output_grain,
            identity_domain=IdentityDomain.derived(
                "related",
                self.name,
                *(
                    port.identity_domain
                    for port in ports
                    if port.identity_domain is not None
                ),
            ),
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
            if set(parent_indexes) != set(role_names):
                missing = sorted(set(role_names) - set(parent_indexes))
                extra = sorted(set(parent_indexes) - set(role_names))
                raise ValueError(
                    "Relate parent refs must provide exactly the declared roles; "
                    f"missing={missing}, extra={extra}"
                )

            parent_refs: List[ParentRef] = []
            selected: list[tuple[str, PortBatch, int]] = []
            for role in role_names:
                local_index = parent_indexes[role]
                if not isinstance(local_index, int):
                    raise TypeError("Relate parent indexes must be integers")
                parent_port = role_to_port[role]
                if parent_port.identity_domain is None:
                    raise ValueError("Relate input has no identity domain")
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
                        identity_domain=parent_port.identity_domain,
                    )
                )
                selected.append((role, parent_port, local_index))
            (
                item_ancestors,
                item_display,
                item_ordinals,
                parent_lineage,
            ) = _merge_relation_metadata(selected)
            record_id = _relation_record_id(self.name, parent_refs, stable_key)
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
            name=self.output_grain,
            identity_domain=IdentityDomain.derived(
                "related",
                self.name,
                *(
                    port.identity_domain
                    for port in ports
                    if port.identity_domain is not None
                ),
            ),
            errors=[error for port in ports for error in port.errors],
        )


__all__ = ["Relate"]
