"""Plain-data relation projection independent of production graph classes."""
from __future__ import annotations


def ordered_unique(items):
    result = []
    for item in items:
        if item not in result:
            result.append(item)
    return tuple(result)


def project_relation(spec: dict, output_slot: int) -> dict:
    kind = spec["kind"]
    if kind == "map":
        return {
            "kind": "same_as",
            "roles": tuple(spec["roles"]),
            "parents": tuple(spec["inputs"]),
        }
    if kind == "expand":
        return {
            "kind": "children_of",
            "parent": spec["parent"],
            "context": tuple(
                port for port in spec["inputs"] if port != spec["parent"]
            ),
        }
    if kind == "reduce":
        return {
            "kind": "aggregate_of",
            "anchor": spec["anchor"],
            "roles": tuple(role for role, _ in spec["members"]),
            "members": tuple(port for _, port in spec["members"]),
        }
    if kind == "relate":
        return {
            "kind": "related_from",
            "roles": tuple(role["name"] for role in spec["roles"]),
            "parents": tuple(role["value"] for role in spec["roles"]),
        }
    if kind == "filter":
        targets = tuple(spec["targets"])
        annotations = spec.get("annotation_arity", 0)
        if output_slot >= len(targets) + annotations:
            raise IndexError(output_slot)
        target = targets[output_slot] if output_slot < len(targets) else targets[0]
        return {
            "kind": "subset_of",
            "target": target,
            "controls": ordered_unique(
                port for port in spec["controls"] if port != target
            ),
        }
    raise ValueError(kind)
