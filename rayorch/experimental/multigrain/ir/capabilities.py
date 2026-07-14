"""Capabilities derived from passive relation contracts.

Capabilities describe execution requirements, not primitive names.  They are
computed from existing IR data and therefore do not add another serialized
source of truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .model import IRNode, RelationKind


class RelationEvidenceFamily(str, Enum):
    ALIGNED = "aligned"
    PARENT = "parent"
    ANCHOR = "anchor"
    ROLE = "role"
    INTERNAL = "internal"


@dataclass(frozen=True)
class PrimitiveCapabilities:
    identity_alignment: bool
    group_completion: bool
    row_partitionable: bool
    relation_evidence: RelationEvidenceFamily


def capabilities_for(node: IRNode) -> PrimitiveCapabilities:
    relations = {relation.relation for relation in node.contract.relations}
    if not relations:
        return PrimitiveCapabilities(False, False, False, RelationEvidenceFamily.INTERNAL)

    if relations <= {RelationKind.PRESERVE, RelationKind.FILTER}:
        evidence = RelationEvidenceFamily.ALIGNED
    elif relations == {RelationKind.EXPAND}:
        evidence = RelationEvidenceFamily.PARENT
    elif relations == {RelationKind.REDUCE}:
        evidence = RelationEvidenceFamily.ANCHOR
    elif relations == {RelationKind.RELATE}:
        evidence = RelationEvidenceFamily.ROLE
    else:
        evidence = RelationEvidenceFamily.INTERNAL

    return PrimitiveCapabilities(
        identity_alignment=relations <= {
            RelationKind.PRESERVE,
            RelationKind.FILTER,
        },
        group_completion=RelationKind.REDUCE in relations,
        row_partitionable=relations <= {
            RelationKind.PRESERVE,
            RelationKind.FILTER,
            RelationKind.EXPAND,
        },
        relation_evidence=evidence,
    )
