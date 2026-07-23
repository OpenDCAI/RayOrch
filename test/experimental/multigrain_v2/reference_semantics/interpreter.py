"""Executable, production-independent semantic reference primitives."""
from __future__ import annotations

from itertools import product
from typing import Mapping

from .models import ExpectedEntity, ExpectedFailure, ExpectedWorkUnit, SemanticCase


RoleRows = Mapping[str, tuple[tuple[str, str], ...]]


def _aligned_length(roles: RoleRows) -> int:
    lengths = {len(rows) for rows in roles.values()}
    if len(lengths) != 1:
        raise ValueError("reference aligned roles have different lengths")
    return next(iter(lengths), 0)


def interpret_map(
    name: str,
    roles: RoleRows,
    output_ports: tuple[str, ...],
) -> SemanticCase:
    count = _aligned_length(roles)
    role_names = tuple(roles)
    entities = tuple(
        ExpectedEntity(
            next(iter(roles.values()))[index][0],
            tuple((role, roles[role][index][1]) for role in role_names),
            output,
        )
        for index in range(count)
        for output in output_ports
    )
    work = tuple(
        ExpectedWorkUnit(
            "row",
            next(iter(roles.values()))[index][0],
            tuple(roles[role][index][1] for role in role_names),
        )
        for index in range(count)
    )
    return SemanticCase(name, "map", entities, work)


def interpret_select(
    name: str,
    roles: RoleRows,
    mask: tuple[bool, ...],
    target_ports: tuple[str, ...],
    annotation_ports: tuple[str, ...] = (),
    *,
    primitive: str = "select",
) -> SemanticCase:
    count = _aligned_length(roles)
    if len(mask) != count or len(target_ports) != len(roles):
        raise ValueError("reference Select shape mismatch")
    role_names = tuple(roles)
    entities = []
    work = []
    for index in range(count):
        identity = roles[role_names[0]][index][0]
        parents = tuple(
            (role, roles[role][index][1]) for role in role_names
        )
        work.append(
            ExpectedWorkUnit(
                "row",
                identity,
                tuple(parent for _, parent in parents),
            )
        )
        if not mask[index]:
            continue
        for port in target_ports + annotation_ports:
            entities.append(ExpectedEntity(identity, parents, port))
    return SemanticCase(name, primitive, tuple(entities), tuple(work))


def interpret_expand(
    name: str,
    parents: tuple[str, ...],
    child_counts: tuple[int, ...],
) -> SemanticCase:
    if len(parents) != len(child_counts):
        raise ValueError("reference Expand shape mismatch")
    entities = tuple(
        ExpectedEntity(
            f"{parent}/child:{ordinal}",
            (("parent", parent),),
        )
        for parent, count in zip(parents, child_counts)
        for ordinal in range(count)
    )
    work = tuple(
        ExpectedWorkUnit("parent", parent, (parent,))
        for parent in parents
    )
    return SemanticCase(name, "expand", entities, work)


def interpret_reduce(
    name: str,
    anchors: tuple[str, ...],
    members: Mapping[str, Mapping[str, tuple[str, ...]]],
    missing_anchors: frozenset[str] = frozenset(),
    missing_causes: Mapping[str, tuple[str, ...]] | None = None,
) -> SemanticCase:
    entities = []
    work = []
    failures = []
    for anchor in anchors:
        closure = (anchor,) + tuple(
            member
            for role in members.values()
            for member in role.get(anchor, ())
        )
        if anchor in missing_anchors:
            failures.append(
                ExpectedFailure(
                    anchor,
                    "suppressed",
                    (
                        missing_causes[anchor]
                        if missing_causes is not None
                        else closure[1:] or (anchor,)
                    ),
                )
            )
            continue
        work.append(ExpectedWorkUnit("fiber", anchor, closure))
        entities.append(
            ExpectedEntity(
                anchor,
                (("anchor", anchor),)
                + tuple(
                    (role, member)
                    for role, groups in members.items()
                    for member in groups.get(anchor, ())
                ),
            )
        )
    return SemanticCase(
        name,
        "reduce",
        tuple(entities),
        tuple(work),
        tuple(failures),
    )


def interpret_key_relate(
    name: str,
    roles: Mapping[str, tuple[tuple[str, object], ...]],
) -> SemanticCase:
    role_names = tuple(roles)
    primary_keys = []
    for _, key in roles[role_names[0]]:
        if key not in primary_keys:
            primary_keys.append(key)
    entities = []
    work = []
    for key in primary_keys:
        groups = tuple(
            tuple(entity for entity, value in roles[role] if value == key)
            for role in role_names
        )
        if any(not group for group in groups):
            continue
        members = tuple(entity for group in groups for entity in group)
        work.append(ExpectedWorkUnit("join-key", f"key:{key}", members))
        for parents in product(*groups):
            entities.append(
                ExpectedEntity(
                    f"key:{key}/" + "/".join(parents),
                    tuple(zip(role_names, parents)),
                )
            )
    return SemanticCase(name, "relate-key", tuple(entities), tuple(work))


def interpret_custom_relate(
    name: str,
    roles: Mapping[str, tuple[str, ...]],
    parent_indexes: Mapping[str, tuple[int, ...]],
) -> SemanticCase:
    role_names = tuple(roles)
    if set(parent_indexes) != set(role_names):
        raise ValueError("reference Custom Relate role mismatch")
    lengths = {len(parent_indexes[role]) for role in role_names}
    if len(lengths) != 1:
        raise ValueError("reference Custom Relate parent lengths differ")
    entities = tuple(
        ExpectedEntity(
            f"relation:{index}",
            tuple(
                (role, roles[role][parent_indexes[role][index]])
                for role in role_names
            ),
        )
        for index in range(next(iter(lengths), 0))
    )
    members = tuple(f"{role}:*" for role in role_names)
    return SemanticCase(
        name,
        "relate-custom",
        entities,
        (
            ExpectedWorkUnit(
                "whole-invocation",
                name,
                members,
            ),
        ),
    )


def interpret_writer(
    name: str,
    roles: RoleRows,
) -> SemanticCase:
    mapped = interpret_map(name, roles, ())
    return SemanticCase(name, "map-writer", (), mapped.work_units)


def interpret_diamond(
    name: str,
    roles: RoleRows,
    surviving_indexes: tuple[int, ...],
    suppressed: Mapping[str, tuple[str, ...]],
) -> SemanticCase:
    count = _aligned_length(roles)
    role_names = tuple(roles)
    entities = []
    work = []
    failures = []
    for index in range(count):
        identity = roles[role_names[0]][index][0]
        parents = tuple(
            (role, roles[role][index][1]) for role in role_names
        )
        if identity in suppressed:
            failures.append(
                ExpectedFailure(identity, "suppressed", suppressed[identity])
            )
            continue
        if index not in surviving_indexes:
            continue
        entities.append(ExpectedEntity(identity, parents))
        work.append(
            ExpectedWorkUnit(
                "row",
                identity,
                tuple(parent for _, parent in parents),
            )
        )
    return SemanticCase(
        name,
        "diamond",
        tuple(entities),
        tuple(work),
        tuple(failures),
    )
