"""Frozen Phase 0 ports and snapshot normalization helpers."""

from __future__ import annotations

from typing import Any

from .models import (
    Emission,
    Failed,
    GrainId,
    GrainRecord,
    ItemRef,
    PortId,
    RoleItems,
    Success,
    Suppressed,
)


RUN_SALT = bytes.fromhex("00112233445566778899aabbccddeeff")
SOURCE_PORT = PortId(0, 0)
EXPAND_PORT_A = PortId(10, 0)
EXPAND_PORT_B = PortId(10, 1)
MAP_PORT = PortId(20, 0)
FILTER_PORT = PortId(30, 0)
REDUCE_PORT = PortId(40, 0)
RELATE_PORT = PortId(50, 0)


def snapshot_item(item: ItemRef) -> dict[str, Any]:
    return {
        "port": [item.port.node, item.port.slot],
        "entity": item.entity.hex(),
    }


def snapshot_cause(cause: ItemRef | GrainId) -> dict[str, Any]:
    if isinstance(cause, GrainId):
        return {"grain": cause.hex()}
    return {"item": snapshot_item(cause)}


def snapshot_roles(roles: tuple[RoleItems, ...]) -> list[dict[str, Any]]:
    return [
        {
            "role": role.role,
            "items": [snapshot_item(item) for item in role.items],
        }
        for role in roles
    ]


def snapshot_emission(emission: Emission) -> dict[str, Any]:
    return {
        "item": snapshot_item(emission.item),
        "ordinal": emission.ordinal,
    }


def snapshot_record(record: GrainRecord) -> dict[str, Any]:
    if isinstance(record.outcome, Success):
        outcome: dict[str, Any] = {
            "kind": "success",
            "ports": [
                [snapshot_emission(emission) for emission in emissions]
                for emissions in record.outcome.emissions_by_port
            ],
        }
    elif isinstance(record.outcome, Failed):
        outcome = {
            "kind": "failed",
            "failure_kind": record.outcome.failure.kind,
            "message": record.outcome.failure.message,
            "causes": [
                snapshot_cause(cause)
                for cause in record.outcome.failure.direct_causes
            ],
        }
    elif isinstance(record.outcome, Suppressed):
        outcome = {
            "kind": "suppressed",
            "causes": [
                snapshot_cause(cause)
                for cause in record.outcome.direct_causes
            ],
        }
    else:  # pragma: no cover
        raise AssertionError(type(record.outcome))

    return {
        "id": record.id.hex(),
        "node": record.node,
        "inputs": snapshot_roles(record.inputs),
        "output_slots": [
            snapshot_item(item) for item in record.output_slots
        ],
        "outcome": outcome,
    }
