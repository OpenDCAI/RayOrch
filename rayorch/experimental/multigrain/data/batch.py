"""Local multi-grain data model for the experimental MVP.

This module deliberately stays independent from Ray. It validates the core
semantics before we thread the model through RuntimeRayModule and Ray actors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence
import uuid


@dataclass(frozen=True, order=True)
class IdentityDomain:
    """One logical record-identity namespace, independent of grain and port."""

    token: str
    label: str

    @classmethod
    def fresh(cls, label: str) -> "IdentityDomain":
        return cls(uuid.uuid4().hex, label)

    @classmethod
    def named(cls, label: str) -> "IdentityDomain":
        return cls(f"named:{label}", label)

    @classmethod
    def derived(
        cls,
        kind: str,
        label: str,
        *parents: "IdentityDomain",
    ) -> "IdentityDomain":
        seed = repr((kind, label, tuple(parent.token for parent in parents)))
        return cls(uuid.uuid5(uuid.NAMESPACE_URL, seed).hex, label)


@dataclass(frozen=True)
class ErrorTrace:
    """User-facing explanation for one failed logical item."""

    source_item: str
    logical_item: str
    failed_op: str
    grain: str
    upstream_path: tuple[str, ...]
    parent: str | None
    action: str
    error: str
    # Ancestor identity domain -> record id for the failed item.
    # Lets a downstream Reduce map a lost descendant to the exact anchor it
    # belonged to, so failure can cascade to that anchor (fail-closed).
    ancestors: Dict[IdentityDomain, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_item": self.source_item,
            "logical_item": self.logical_item,
            "failed_op": self.failed_op,
            "grain": self.grain,
            "upstream_path": list(self.upstream_path),
            "parent": self.parent,
            "action": self.action,
            "error": self.error,
            "ancestors": [
                {
                    "token": domain.token,
                    "label": domain.label,
                    "record_id": record_id,
                }
                for domain, record_id in sorted(
                    self.ancestors.items(),
                    key=lambda item: (item[0].token, item[0].label),
                )
            ],
        }


@dataclass(frozen=True)
class ParentRef:
    """Invocation-local parent reference for relation-aware outputs."""

    role: str
    port: str
    record_id: str
    display_key: str
    identity_domain: IdentityDomain

    def to_dict(self) -> Dict[str, str]:
        return {
            "role": self.role,
            "port": self.port,
            "record_id": self.record_id,
            "display_key": self.display_key,
            "identity_domain": self.identity_domain.label,
            "identity_domain_token": self.identity_domain.token,
        }


@dataclass(frozen=True)
class DeferredRecord:
    """One attributable record retained for a later stage-epoch retry."""

    token: str
    inputs: tuple["PortBatch", ...]
    failed_op: str
    error: str
    target_rows: int


@dataclass
class PortBatch:
    """One logical port with one record set and one grain."""

    name: str
    values: List[Any]
    record_ids: List[str]
    display_keys: List[str]
    ancestors: List[Dict[IdentityDomain, str]]
    ancestor_display: List[Dict[str, str]]
    ordinals: List[Dict[IdentityDomain, int]]
    lineage: List[tuple[str, ...]]
    identity_domain: IdentityDomain | None = None
    relations: List[tuple[ParentRef, ...]] = field(default_factory=list)
    errors: List[ErrorTrace] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.identity_domain is None:
            self.identity_domain = IdentityDomain.named(self.name)
        # Keep hand-built MVP batches usable while all framework-produced
        # metadata uses typed domains.
        self.ancestors = [
            {
                (
                    key
                    if isinstance(key, IdentityDomain)
                    else (
                        self.identity_domain
                        if key == self.name
                        else IdentityDomain.named(str(key))
                    )
                ): value
                for key, value in item.items()
            }
            for item in self.ancestors
        ]
        self.ordinals = [
            {
                (
                    key
                    if isinstance(key, IdentityDomain)
                    else (
                        self.identity_domain
                        if key == self.name
                        else IdentityDomain.named(str(key))
                    )
                ): value
                for key, value in item.items()
            }
            for item in self.ordinals
        ]
        n = len(self.values)
        fields = {
            "record_ids": self.record_ids,
            "display_keys": self.display_keys,
            "ancestors": self.ancestors,
            "ancestor_display": self.ancestor_display,
            "ordinals": self.ordinals,
            "lineage": self.lineage,
        }
        for name, value in fields.items():
            if len(value) != n:
                raise ValueError(
                    f"{name} length {len(value)} does not match values length {n}"
                )
        if len(set(self.record_ids)) != n:
            raise ValueError("record_ids must be unique within a PortBatch")
        if self.relations and len(self.relations) != n:
            raise ValueError(
                f"relations length {len(self.relations)} does not match values length {n}"
            )

    def __len__(self) -> int:
        return len(self.values)

    @property
    def grain(self) -> str:
        return self.name

    @classmethod
    def source(
        cls,
        values: Sequence[Any],
        *,
        name: str = "source",
        display_key: Callable[[Any], str] | None = None,
        identity_domain: IdentityDomain | str | None = None,
    ) -> "PortBatch":
        domain = (
            identity_domain
            if isinstance(identity_domain, IdentityDomain)
            else (
                IdentityDomain.named(identity_domain)
                if isinstance(identity_domain, str)
                else IdentityDomain.named(name)
            )
        )
        keys = [
            str(display_key(value) if display_key is not None else value)
            for value in values
        ]
        record_ids = [f"{name}:{index}" for index in range(len(values))]
        return cls(
            name=name,
            values=list(values),
            record_ids=record_ids,
            display_keys=keys,
            ancestors=[
                {
                    domain: record_id,
                }
                for record_id in record_ids
            ],
            ancestor_display=[
                {
                    name: key,
                }
                for key in keys
            ],
            ordinals=[{} for _ in values],
            lineage=[() for _ in values],
            identity_domain=domain,
        )

    def take(self, indices: Sequence[int], *, name: str | None = None) -> "PortBatch":
        return PortBatch(
            name=name or self.name,
            values=[self.values[i] for i in indices],
            record_ids=[self.record_ids[i] for i in indices],
            display_keys=[self.display_keys[i] for i in indices],
            ancestors=[dict(self.ancestors[i]) for i in indices],
            ancestor_display=[dict(self.ancestor_display[i]) for i in indices],
            ordinals=[dict(self.ordinals[i]) for i in indices],
            lineage=[tuple(self.lineage[i]) for i in indices],
            identity_domain=self.identity_domain,
            relations=[
                tuple(self.relations[i]) for i in indices
            ] if self.relations else [],
            errors=list(self.errors),
        )

    def with_values(
        self,
        values: Sequence[Any],
        *,
        name: str,
        op_name: str,
    ) -> "PortBatch":
        if len(values) != len(self):
            raise ValueError(
                f"output length {len(values)} does not match input length {len(self)}"
            )
        return PortBatch(
            name=name,
            values=list(values),
            record_ids=list(self.record_ids),
            display_keys=list(self.display_keys),
            ancestors=[dict(item) for item in self.ancestors],
            ancestor_display=[dict(item) for item in self.ancestor_display],
            ordinals=[dict(item) for item in self.ordinals],
            lineage=[tuple((*path, op_name)) for path in self.lineage],
            identity_domain=self.identity_domain,
            relations=[tuple(item) for item in self.relations],
            errors=list(self.errors),
        )

    def trace_item(self, **selectors: Any) -> List[Dict[str, Any]]:
        """Return records whose display metadata contains all selectors."""
        expected = {key: str(value) for key, value in selectors.items()}
        matches: List[Dict[str, Any]] = []
        for index, display in enumerate(self.ancestor_display):
            combined = dict(display)
            combined[self.name] = self.display_keys[index]
            if all(combined.get(key) == value for key, value in expected.items()):
                matches.append(
                    {
                        "record_id": self.record_ids[index],
                        "display": combined,
                        "lineage": list(self.lineage[index]),
                        "relations": [
                            ref.to_dict()
                            for ref in (
                                self.relations[index] if self.relations else ()
                            )
                        ],
                        "value": self.values[index],
                    }
                )
        return matches


@dataclass(frozen=True)
class NodeExecution:
    """Internal node result plus records waiting for a deferred retry."""

    outputs: tuple[PortBatch, ...]
    deferred: tuple[DeferredRecord, ...] = ()


def source(
    values: Sequence[Any],
    *,
    name: str = "source",
    display_key: Callable[[Any], str] | None = None,
    identity_domain: IdentityDomain | str | None = None,
) -> PortBatch:
    return PortBatch.source(
        values,
        name=name,
        display_key=display_key,
        identity_domain=identity_domain,
    )


def dedupe_errors(errors: Sequence[ErrorTrace]) -> List[ErrorTrace]:
    result: List[ErrorTrace] = []
    for error in errors:
        if error not in result:
            result.append(error)
    return result


def concat(batches: Sequence[PortBatch], *, name: str | None = None) -> PortBatch:
    """Concatenate batches from the same logical port."""
    if not batches:
        return PortBatch(name or "empty", [], [], [], [], [], [], [])
    port_name = name or batches[0].name
    domain = batches[0].identity_domain
    if any(batch.identity_domain != domain for batch in batches[1:]):
        raise ValueError("cannot concat batches from different identity domains")
    source_name = batches[0].name
    if any(batch.name != source_name for batch in batches[1:]):
        raise ValueError("cannot concat batches from different logical ports")
    errors: List[ErrorTrace] = []
    values: List[Any] = []
    record_ids: List[str] = []
    display_keys: List[str] = []
    ancestors: List[Dict[str, str]] = []
    ancestor_display: List[Dict[str, str]] = []
    ordinals: List[Dict[str, int]] = []
    lineage: List[tuple[str, ...]] = []
    relations: List[tuple[ParentRef, ...]] = []
    for batch in batches:
        values.extend(batch.values)
        record_ids.extend(batch.record_ids)
        display_keys.extend(batch.display_keys)
        ancestors.extend(dict(item) for item in batch.ancestors)
        ancestor_display.extend(dict(item) for item in batch.ancestor_display)
        ordinals.extend(dict(item) for item in batch.ordinals)
        lineage.extend(tuple(item) for item in batch.lineage)
        if batch.relations:
            relations.extend(tuple(item) for item in batch.relations)
        errors.extend(batch.errors)
    if relations and len(relations) != len(values):
        raise ValueError("cannot concat mixed relation-aware and relation-less batches")
    return PortBatch(
        name=port_name,
        values=values,
        record_ids=record_ids,
        display_keys=display_keys,
        ancestors=ancestors,
        ancestor_display=ancestor_display,
        ordinals=ordinals,
        lineage=lineage,
        identity_domain=domain,
        relations=relations,
        errors=dedupe_errors(errors),
    )


def rebatch(batch: PortBatch, batch_size: int) -> List[PortBatch]:
    """Split a port into dense physical batches without changing identities."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [
        batch.take(range(start, min(start + batch_size, len(batch))))
        for start in range(0, len(batch), batch_size)
    ]


def _as_columns(batch: PortBatch) -> List[Any]:
    return list(batch.values)


def _call_user(op: Any, *args: Any) -> Any:
    if hasattr(op, "run"):
        return op.run(*args)
    return op(*args)


def _ensure_tuple(value: Any) -> tuple[Any, ...]:
    return value if isinstance(value, tuple) else (value,)


def _normalize_output_lists(value: Any) -> tuple[List[Any], ...]:
    outputs = _ensure_tuple(value)
    normalized: List[List[Any]] = []
    for output in outputs:
        if not isinstance(output, list):
            raise TypeError(
                "operator outputs must be lists for the local multigrain MVP"
            )
        normalized.append(output)
    return tuple(normalized)


def _align_by_identity(ports: Sequence[PortBatch]) -> List[PortBatch]:
    if not ports:
        raise ValueError("at least one input port is required")
    base = ports[0]
    base_ids = list(base.record_ids)
    aligned = [base]
    for port in ports[1:]:
        if port.identity_domain != base.identity_domain:
            raise ValueError(
                f"cannot align port '{port.name}' with '{base.name}' "
                "across identity domains"
            )
        if set(port.record_ids) != set(base_ids):
            raise ValueError(
                f"cannot align port '{port.name}' with '{base.name}' by identity"
            )
        index = {record_id: i for i, record_id in enumerate(port.record_ids)}
        aligned.append(port.take([index[record_id] for record_id in base_ids]))
    return aligned


def _source_item(batch: PortBatch, index: int) -> str:
    display = batch.ancestor_display[index]
    if display:
        first_key = next(iter(display))
        return display[first_key]
    return batch.display_keys[index]


def _parent_display(batch: PortBatch, index: int) -> str | None:
    display = batch.ancestor_display[index]
    return next(iter(display.values())) if display else None


def _without_index(batch: PortBatch, bad_index: int) -> PortBatch:
    return batch.take([i for i in range(len(batch)) if i != bad_index])


@dataclass(frozen=True)
class Grouped:
    anchor: Any
    descendants: tuple[Any, ...]
    roles: tuple[str | None, ...] = ()


@dataclass(frozen=True)
class Via:
    """Select one direct relation-parent role for Reduce grouping."""

    port: Any
    role: str

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError("via role must be non-empty")


def via(port: Any, *, role: str) -> Via:
    return Via(port=port, role=role)


def group_by(anchor: Any, *descendants: Any) -> Grouped:
    ports: list[Any] = []
    roles: list[str | None] = []
    for descendant in descendants:
        if isinstance(descendant, Via):
            ports.append(descendant.port)
            roles.append(descendant.role)
        else:
            ports.append(descendant)
            roles.append(None)
    return Grouped(
        anchor=anchor,
        descendants=tuple(ports),
        roles=tuple(roles),
    )


__all__ = [
    "IdentityDomain",
    "ErrorTrace",
    "Grouped",
    "Via",
    "dedupe_errors",
    "ParentRef",
    "PortBatch",
    "concat",
    "group_by",
    "rebatch",
    "source",
    "via",
]

