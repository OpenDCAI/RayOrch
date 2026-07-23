"""Actor-capacity regressions for reservation and generation fencing."""

from __future__ import annotations

from dataclasses import replace

import pytest
import ray

from rayorch.experimental.multigrain_v2_5.executor import Arena, Executor
from rayorch.experimental.multigrain_v2_5.graph import (
    BindingReceipt,
    UdfRecipe,
    admit_source,
    compile_graph,
    plan_map,
)
from rayorch.experimental.multigrain_v2_5.grain import GrainPhase

from ..reference_semantics.semantic_cases import RUN_SALT
from ..support import map_node, source_node


pytestmark = pytest.mark.usefixtures("ray_cluster")


class Echo:
    def run(self, values):
        return list(values)


def test_actor_saturation_does_not_reserve_or_increment_waiting_grain():
    """A full actor mailbox leaves later grains READY with generation zero."""

    source_spec = source_node(0)
    map_spec = map_node(1, source_spec.output_ports[0])
    assert map_spec.execution is not None
    map_spec = replace(
        map_spec,
        udf_recipe=UdfRecipe(Echo),
        execution=replace(
            map_spec.execution,
            batch_size=1,
            replicas=1,
            max_batch_wait_ms=0,
        ),
    )
    graph = compile_graph((source_spec, map_spec))
    arena = Arena(1, graph, RUN_SALT)
    transport = Executor(graph).ray_transport(max_pending_per_actor=1)
    try:
        sources = tuple(
            admit_source(source_spec, RUN_SALT, position)
            for position in range(2)
        )
        arena.admit_source_batch(sources, ray.put(("a", "b")))
        records = []
        for source in sources:
            records.extend(
                arena.ensure_plans(
                    (
                        plan_map(
                            map_spec,
                            RUN_SALT,
                            (
                                BindingReceipt.present(
                                    "primary",
                                    source.output_slots[0],
                                ),
                            ),
                        ),
                    )
                )
            )

        first = arena.reserve_dispatch(1, admission_closed=False)
        assert first is not None
        assert transport.submit(arena, first)
        assert not transport.can_submit(1)

        waiting = records[1]
        assert waiting.phase is GrainPhase.READY
        assert waiting.generation == 0
        assert waiting.active is None

        transport.drain()
        assert transport.can_submit(1)
        second = arena.reserve_dispatch(1, admission_closed=False)
        assert second is not None
        assert second.entries[0].token.grain == waiting.id
        assert waiting.phase is GrainPhase.IN_FLIGHT
        assert waiting.generation == 1
        assert transport.submit(arena, second)
        transport.drain()
    finally:
        transport.shutdown()
