"""Real Ray actor lifetime, retry fencing and record-level failure boundaries."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import rayorch.experimental.multigrain_v3_5 as mg
from rayorch.experimental.multigrain_v3_5.executor import Executor
from rayorch.experimental.multigrain_v3_5.model import ItemOutcome
from rayorch.experimental.multigrain_v3_5.protocol import RecordFailure


@pytest.fixture(scope="module", autouse=True)
def ray_runtime():
    import ray

    started_here = not ray.is_initialized()
    if started_here:
        ray.init(
            address=os.environ.get("RAY_ADDRESS", "local"),
            include_dashboard=False,
        )
    yield
    if started_here:
        ray.shutdown()


class PersistentIdentity:
    def __init__(self) -> None:
        self.identity = os.getpid()

    def run(self, values):
        return [(value, self.identity) for value in values]


class IdentityPipeline(mg.Pipeline):
    def __init__(self) -> None:
        self.identity = mg.RayModule(PersistentIdentity).ray_options(
            batch_size=3,
            replicas=1,
            num_cpus=0,
        )

    def forward(self, values):
        return self.identity(values)


def test_persistent_actor_is_shared_by_multiple_arenas():
    with Executor(IdentityPipeline()) as executor:
        result = executor.run(range(15), arena_size=4, max_in_flight=3)

    assert [value for value, _ in result.outputs] == list(range(15))
    assert len({identity for _, identity in result.outputs}) == 1
    assert result.actor_count == 1
    assert result.max_active_arenas == 3
    assert len(result.arenas) == 4
    assert all(arena.is_complete() for arena in result.arenas)
    assert all(not arena.state.values for arena in result.arenas)


class KeywordOnlyAdd:
    def run(self, values, *, increments):
        return [
            value + increment
            for value, increment in zip(values, increments)
        ]


class KeywordPipeline(mg.Pipeline):
    def __init__(self) -> None:
        self.call = mg.RayModule(KeywordOnlyAdd).ray_options(
            batch_size=4,
            num_cpus=0,
        )

    def forward(self, values, increments):
        return self.call(values, increments=increments)


def test_keyword_input_layout_survives_actor_boundary():
    with Executor(KeywordPipeline()) as executor:
        result = executor.run([1, 2, 3], [10, 20, 30])

    assert result.outputs == [11, 22, 33]


class CrashOnce:
    def __init__(self, marker: str) -> None:
        self.marker = Path(marker)

    def run(self, values):
        if not self.marker.exists():
            self.marker.write_text("crashed", encoding="utf-8")
            os._exit(17)
        return [value * 2 for value in values]


class CrashPipeline(mg.Pipeline):
    def __init__(self, marker: str) -> None:
        self.call = (
            mg.RayModule(CrashOnce)
            .pre_init(marker)
            .ray_options(
                batch_size=4,
                replicas=1,
                max_retries=1,
                num_cpus=0,
                max_restarts=0,
            )
        )

    def forward(self, values):
        return self.call(values)


def test_actor_crash_replaces_actor_and_replays_same_grains(tmp_path):
    marker = tmp_path / "v35-crash-once"
    with Executor(CrashPipeline(str(marker))) as executor:
        result = executor.run(range(4), arena_size=4)

    metrics = next(iter(result.calls.values()))
    assert result.outputs == [0, 2, 4, 6]
    assert metrics.actor_starts == 2
    assert metrics.retries == 4
    assert metrics.rpcs == 2
    assert {
        grain.generation for grain in result.arenas[0].state.grains.values()
    } == {1}


class Render:
    def run(self, parents):
        return [[(parent, ordinal) for ordinal in range(3)] for parent in parents]


class FailOneLeaf:
    def run(self, leaves):
        return [
            RecordFailure(f"bad leaf {leaf}") if leaf == (1, 1) else leaf
            for leaf in leaves
        ]


class Assemble:
    def run(self, parents, groups):
        return list(zip(parents, groups))


class BadLeafPipeline(mg.Pipeline):
    def __init__(self) -> None:
        self.render = mg.RayModule(Render).ray_options(batch_size=8, num_cpus=0)
        self.transform = mg.RayModule(FailOneLeaf).ray_options(
            batch_size=16,
            num_cpus=0,
        )
        self.assemble = mg.RayModule(Assemble).ray_options(
            batch_size=8,
            num_cpus=0,
        )

    def forward(self, parents):
        leaves = mg.F.expand(self.render(parents))
        values = self.transform(leaves)
        return self.assemble(parents, mg.F.reduce(values))


def test_record_failure_suppresses_only_its_parent():
    with Executor(BadLeafPipeline()) as executor:
        result = executor.run([0, 1, 2])

    assert result.outputs[0] == (0, [(0, 0), (0, 1), (0, 2)])
    assert result.outputs[1] is ItemOutcome.SUPPRESSED
    assert result.outputs[2] == (2, [(2, 0), (2, 1), (2, 2)])
    transform_call = next(
        call
        for call, spec in result.arenas[0].plan.calls.items()
        if spec.kernel.target is FailOneLeaf
    )
    assert result.calls[transform_call].grains == 9


class MultiOutputFail:
    def run(self, values):
        return (
            [value * 10 for value in values],
            [
                RecordFailure("second output failed") if value == 2 else value * 100
                for value in values
            ],
        )


class MultiOutputPipeline(mg.Pipeline):
    def __init__(self) -> None:
        self.call = mg.RayModule(MultiOutputFail, num_outputs=2).ray_options(
            batch_size=8,
            num_cpus=0,
        )

    def forward(self, values):
        return self.call(values)


def test_record_failure_makes_all_outputs_of_one_grain_failed():
    with Executor(MultiOutputPipeline()) as executor:
        left, right = executor.run([1, 2, 3]).outputs

    assert left == [10, ItemOutcome.FAILED, 30]
    assert right == [100, ItemOutcome.FAILED, 300]
