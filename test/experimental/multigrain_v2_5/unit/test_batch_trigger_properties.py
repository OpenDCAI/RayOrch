"""Fixed-seed schedule properties for physical batch triggering."""

from __future__ import annotations

import random

from .test_batch_trigger import _arena, _enqueue


def test_random_arrival_schedules_dispatch_every_grain_once_with_bounded_batches():
    """Random trickle/burst schedules preserve exact membership and batch bounds."""

    randomizer = random.Random(25_032_026)
    for _case in range(40):
        batch_size = randomizer.randint(2, 16)
        count = randomizer.randint(1, 80)
        wait_ms = randomizer.choice((0.0, 1.0, 2.0, 5.0))
        clock, arena, map_spec, sources = _arena(
            batch_size=batch_size,
            wait_ms=wait_ms,
            count=count,
        )
        records = []
        cursor = 0
        plans = []
        while cursor < count:
            chunk = randomizer.randint(1, min(12, count - cursor))
            records.extend(
                _enqueue(
                    arena,
                    map_spec,
                    sources[cursor : cursor + chunk],
                )
            )
            cursor += chunk
            clock.advance_ms(randomizer.random() * 3.0)
            while True:
                plan = arena.reserve_dispatch(
                    1,
                    admission_closed=False,
                )
                if plan is None:
                    break
                plans.append(plan)

        while arena.ready_count(1):
            plan = arena.reserve_dispatch(1, admission_closed=True)
            assert plan is not None
            plans.append(plan)

        dispatched = [
            entry.token.grain for plan in plans for entry in plan.entries
        ]
        assert len(dispatched) == count
        assert len(set(dispatched)) == count
        assert set(dispatched) == {record.id for record in records}
        assert all(1 <= len(plan.entries) <= batch_size for plan in plans)
