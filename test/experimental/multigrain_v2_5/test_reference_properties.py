from __future__ import annotations

import random

from .reference_semantics.interpreter import ReferenceInterpreter
from .reference_semantics.models import (
    GrainId,
    ItemRef,
    MemberSettlement,
    PortId,
    RoleItems,
    Suppressed,
)
from .reference_semantics.semantic_cases import (
    EXPAND_PORT_A,
    MAP_PORT,
    REDUCE_PORT,
    RUN_SALT,
    SOURCE_PORT,
    snapshot_record,
)


def test_fixed_seed_retry_and_completion_reordering_properties():
    randomizer = random.Random(25_032_025)
    interpreter = ReferenceInterpreter(RUN_SALT)

    for position in range(80):
        source = interpreter.source(SOURCE_PORT, position)
        anchor = source.output_slots[0]
        inputs = (RoleItems("parent", (anchor,)),)
        count = randomizer.randrange(0, 8)

        first = interpreter.expand(10, inputs, (EXPAND_PORT_A,), count)
        retry = interpreter.expand(10, inputs, (EXPAND_PORT_A,), count)
        assert first == retry

        emissions = first.outcome.emissions_by_port[0]  # type: ignore[union-attr]
        members = tuple(
            ItemRef(MAP_PORT, emission.item.entity) for emission in emissions
        )
        settlements = []
        expected_causes = []
        for ordinal, member in enumerate(members):
            draw = randomizer.randrange(5)
            if draw == 0:
                settlements.append(MemberSettlement.dropped(ordinal))
            elif draw == 1:
                receipt = GrainId(
                    ordinal.to_bytes(1, "big") + bytes([position]) + b"x" * 14
                )
                settlements.append(
                    MemberSettlement.failed(ordinal, member, receipt)
                )
                expected_causes.append((ordinal, receipt))
            else:
                settlements.append(
                    MemberSettlement.present(ordinal, member)
                )

        shuffled = list(settlements)
        randomizer.shuffle(shuffled)
        canonical = interpreter.reduce(
            40,
            anchor,
            MAP_PORT,
            (REDUCE_PORT,),
            expected=count,
            settlements=settlements,
        )
        reordered = interpreter.reduce(
            40,
            anchor,
            MAP_PORT,
            (REDUCE_PORT,),
            expected=count,
            settlements=shuffled,
        )

        assert snapshot_record(canonical) == snapshot_record(reordered)
        if expected_causes:
            assert isinstance(canonical.outcome, Suppressed)
            assert canonical.outcome.direct_causes == tuple(
                receipt for _, receipt in sorted(expected_causes)
            )


def test_source_identity_depends_on_logical_position_not_arena_partition():
    interpreter = ReferenceInterpreter(RUN_SALT)
    positions = tuple(range(30))
    partition_a = (positions[:3], positions[3:17], positions[17:])
    partition_b = (positions[:11], positions[11:12], positions[12:])

    ids_a = {
        position: interpreter.source(SOURCE_PORT, position).id
        for arena in partition_a
        for position in arena
    }
    ids_b = {
        position: interpreter.source(SOURCE_PORT, position).id
        for arena in partition_b
        for position in arena
    }
    assert ids_a == ids_b
