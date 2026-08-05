"""V3 actor outstanding credit 与 generation 收口的 Ray-free 测试。"""

from __future__ import annotations

import sys
from collections import deque
from types import SimpleNamespace

import pytest

from rayorch.experimental.multigrain_v3 import Map, Pipeline
from rayorch.experimental.multigrain_v3.contracts import ExecutionError
from rayorch.experimental.multigrain_v3.executor import Executor
from rayorch.experimental.multigrain_v3.execution import (
    ActorCreditWindow,
    ExecutionPool,
    PendingRPC,
    StageExecutor,
)


class _Identity:
    """用于编译 Stage execution options 的最小 UDF。"""

    def __call__(self, value):
        """原样返回输入。"""

        return value


class _WindowPipeline(Pipeline):
    """声明带 Stage 级 outstanding override 的最小 Pipeline。"""

    def __init__(self) -> None:
        """配置一个执行并发二、outstanding 三的 Stage。"""

        self.map = Map(_Identity).ray_options(
            max_concurrency=2,
            max_outstanding_per_actor=3,
        )

    def forward(self, source):
        """把 source 连接到唯一 Map Stage。"""

        return self.map(source)


def test_credit_window_bounds_and_releases_each_slot() -> None:
    """credit ledger 不超发，并在终止时精确释放一次。"""

    window = ActorCreditWindow(actor_count=2, max_outstanding_per_actor=1)

    assert window.try_acquire(start=0, candidates=(0, 1)) == 0
    assert window.try_acquire(start=0, candidates=(0, 1)) == 1
    assert window.try_acquire(start=0, candidates=(0, 1)) is None
    assert not window.has_capacity()

    window.release(0)
    assert window.has_capacity()
    assert window.try_acquire(start=0, candidates=(0,)) == 0
    window.release(0)
    with pytest.raises(ExecutionError, match="released twice"):
        window.release(0)


def test_stage_override_is_static_execution_metadata() -> None:
    """Stage override 编译进 ExecutionSpec，不泄漏到通用 Ray options。"""

    stage = _WindowPipeline().compile().dag.stages[1]

    assert stage.execution is not None
    assert stage.execution.max_outstanding_per_actor == 3
    assert dict(stage.execution.ray_options)["max_concurrency"] == 2


def test_executor_accepts_only_one_outstanding_spelling() -> None:
    """Executor 规范化旧名称，并拒绝新旧参数同时出现。"""

    executor = Executor(_WindowPipeline(), max_pending_per_actor=3)
    assert executor.max_outstanding_per_actor == 3
    assert executor.max_pending_per_actor == 3

    with pytest.raises(ValueError, match="use only"):
        Executor(
            _WindowPipeline(),
            max_outstanding_per_actor=3,
            max_pending_per_actor=3,
        )


def test_fail_generation_releases_all_credits_and_replaces_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """actor 故障一次摘除同 generation 全部 RPC，并只重建一次。"""

    killed: list[object] = []
    monkeypatch.setitem(
        sys.modules,
        "ray",
        SimpleNamespace(kill=lambda actor: killed.append(actor)),
    )
    executor = object.__new__(StageExecutor)
    executor.actors = ["old-actor"]
    executor.worker_generations = [7]
    executor.credits = ActorCreditWindow(1, 3)
    for _ in range(3):
        assert executor.credits.try_acquire(start=0, candidates=(0,)) == 0
    executor.pending = {
        f"ref-{index}": PendingRPC(
            SimpleNamespace(arena_id=index, call=SimpleNamespace()),
            1,
            0,
            7,
            f"ref-{index}",
            (),
            0.0,
        )
        for index in range(3)
    }
    executor._spawn = lambda: "new-actor"

    failed = executor.fail_generation(0, 7)

    assert len(failed) == 3
    assert executor.pending == {}
    assert executor.pending_by_slot == [0]
    assert executor.worker_generations == [8]
    assert executor.actors == ["new-actor"]
    assert killed == ["old-actor"]


def test_pool_buffers_every_failure_from_a_dead_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """poll 将故障 generation 的每个 RPC 转成可路由事件后再允许回收。"""

    monkeypatch.setitem(
        sys.modules,
        "ray",
        SimpleNamespace(kill=lambda actor: None),
    )
    executor = object.__new__(StageExecutor)
    executor.actors = ["old-actor"]
    executor.worker_generations = [2]
    executor.credits = ActorCreditWindow(1, 2)
    for _ in range(2):
        assert executor.credits.try_acquire(start=0, candidates=(0,)) == 0
    executor.pending = {
        f"ref-{arena_id}": PendingRPC(
            SimpleNamespace(
                arena_id=arena_id,
                call=SimpleNamespace(dispatch=100 + arena_id),
            ),
            1,
            0,
            2,
            f"ref-{arena_id}",
            (),
            0.0,
        )
        for arena_id in (10, 11)
    }
    executor._spawn = lambda: "new-actor"
    pool = object.__new__(ExecutionPool)
    pool.executors = {1: executor}
    pool.buffered_events = deque()

    class _FailingRay:
        """模拟 wait 命中后 ray.get 报基础设施失败。"""

        @staticmethod
        def wait(refs, *, num_returns, timeout):
            """返回第一个 ready ref。"""

            del num_returns, timeout
            return [refs[0]], refs[1:]

        @staticmethod
        def get(ref):
            """对命中的 ref 抛出 actor failure。"""

            raise RuntimeError(f"actor died while resolving {ref}")

    pool.ray = _FailingRay()

    first = pool.poll()
    assert first is not None and first.arena_id == 10
    assert pool.pending_count == 1
    assert pool.has_outstanding(11)
    second = pool.poll()
    assert second is not None and second.arena_id == 11
    assert pool.pending_count == 0
    assert not pool.has_outstanding(11)
    assert executor.pending_by_slot == [0]
    assert executor.worker_generations == [3]
