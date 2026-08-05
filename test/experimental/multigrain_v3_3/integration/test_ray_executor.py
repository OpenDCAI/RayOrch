"""Ray actor、持久实例、多 Arena overlap 与 generation retry 回归。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import rayorch.experimental.multigrain_v3_3 as mg
from rayorch.experimental.multigrain_v3_3.benchmark.dummy import DummyPipeline
from rayorch.experimental.multigrain_v3_3.executor import LocalExecutor
from rayorch.experimental.multigrain_v3_3.ray_executor import RayExecutor


class PersistentIdentity:
    """每个 actor 构造一次 identity，用于验证跨 batch/Arena 的实例复用。"""

    def __init__(self) -> None:
        self.identity = os.getpid()

    def run(self, values):
        """返回 value 与 actor identity，保持列式 batch ABI。"""

        return [(value, self.identity) for value in values]


class IdentityPipeline(mg.Pipeline):
    """只含一个 Call 的最小持久 actor Pipeline。"""

    def __init__(self) -> None:
        self.identity = (
            mg.RayModule(PersistentIdentity)
            .ray_options(batch_size=3, replicas=1, num_cpus=0)
        )

    def forward(self, values):
        """声明唯一计算边界。"""

        return self.identity(values)


class FailDuringInit:
    """构造时失败，用于验证 Executor 不把启动错误推迟到首个 batch。"""

    def __init__(self) -> None:
        raise RuntimeError("intentional startup failure")

    def run(self, values):
        """该方法不应被调用。"""

        return values


class StartupFailurePipeline(mg.Pipeline):
    """只含一个构造失败 Call 的启动屏障测试 Pipeline。"""

    def __init__(self) -> None:
        self.call = mg.RayModule(FailDuringInit).ray_options(num_cpus=0)

    def forward(self, values):
        """声明唯一 Call。"""

        return self.call(values)


class CrashOnce:
    """第一次执行直接结束 actor，替换 actor 后正常返回。"""

    def __init__(self, marker: str) -> None:
        self.marker = Path(marker)

    def run(self, values):
        """以共享 marker 保证整个测试只触发一次进程级 crash。"""

        if not self.marker.exists():
            self.marker.write_text("crashed", encoding="utf-8")
            os._exit(17)
        return [value * 2 for value in values]


class CrashPipeline(mg.Pipeline):
    """验证 actor replacement、generation fence 与局部 replay。"""

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
        """声明单 Call replay 场景。"""

        return self.call(values)


@pytest.fixture(scope="module", autouse=True)
def ray_runtime():
    """优先遵循 RAY_ADDRESS；未配置时启动进程自有的本地集群。"""

    import ray

    if not ray.is_initialized():
        ray.init(address=os.environ.get("RAY_ADDRESS", "local"))
    yield


def test_actor_startup_failure_is_reported_before_run():
    """初始 UDF 构造失败必须在 RayExecutor 构造阶段同步暴露。"""

    import ray

    with pytest.raises(ray.exceptions.RayActorError):
        RayExecutor(StartupFailurePipeline())


def test_persistent_actor_is_shared_by_multiple_arenas():
    with RayExecutor(IdentityPipeline()) as executor:
        result = executor.run(range(15), arena_size=4, max_in_flight=3)

    assert [value for value, _ in result.outputs] == list(range(15))
    assert len({identity for _, identity in result.outputs}) == 1
    assert result.actor_count == 1
    assert result.max_active_arenas == 3
    assert len(result.arenas) == 4
    assert all(arena.is_complete() for arena in result.arenas)
    assert result.released_values > 0
    assert all(not arena.state.values for arena in result.arenas)


def test_reusing_ray_executor_does_not_accumulate_metrics():
    """actor 可跨 run 复用，但上一份 RunResult 不得被后续运行修改。"""

    with RayExecutor(IdentityPipeline()) as executor:
        first = executor.run(range(7), arena_size=7)
        first_metrics = next(iter(first.calls.values()))
        first_batches = tuple(first_metrics.batch_sizes)

        second = executor.run(range(2), arena_size=2)
        second_metrics = next(iter(second.calls.values()))

    assert first_metrics.rpcs == 3
    assert tuple(first_metrics.batch_sizes) == first_batches == (3, 3, 1)
    assert second_metrics.rpcs == 1
    assert second_metrics.batch_sizes == [2]
    assert first.actor_count == second.actor_count == 1


def test_ray_dummy_matches_local_and_structural_relations_have_no_actors():
    pipeline = DummyPipeline(mode="elastic", batch_size=8, rounds=1)
    local = LocalExecutor(pipeline).run(range(48))
    with RayExecutor(pipeline) as executor:
        remote = executor.run(
            range(48),
            arena_size=16,
            max_in_flight=2,
        )

    assert remote.outputs == local.outputs
    assert remote.actor_count == 4  # 仅四个 RayModule Call；F.* 均无 actor。
    assert remote.rpc_count > 0


def test_actor_crash_replaces_actor_and_replays_same_grain(tmp_path):
    marker = tmp_path / "v33-crash-once"
    with RayExecutor(CrashPipeline(str(marker))) as executor:
        result = executor.run(
            range(4),
            arena_size=4,
            max_in_flight=1,
        )

    metrics = next(iter(result.calls.values()))
    assert result.outputs == [0, 2, 4, 6]
    assert metrics.actor_starts == 2
    assert metrics.retries == 4  # 一个失败 batch 中的四个 Grain 各 replay 一次。
    assert metrics.rpcs == 2
    assert {
        grain.generation for grain in result.arenas[0].state.grains.values()
    } == {1}
