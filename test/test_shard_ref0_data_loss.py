from __future__ import annotations

from typing import List

import pytest
import ray

from rayorch import OverlappedPipeline, RayModule
from rayorch.dispatch_mode import collect_concat, dispatch_shard_all_args_mod


@pytest.fixture
def ray_session():
    ray.init(ignore_reinit_error=True, num_cpus=8)
    yield
    if ray.is_initialized():
        ray.shutdown()


def _kill_modules(*modules: RayModule) -> None:
    for m in modules:
        for actor in getattr(m, "actors", []):
            try:
                ray.kill(actor)
            except Exception:
                pass


class IdentityBatchOp:
    def run(self, xs: List[int]) -> List[int]:
        return list(xs)


class CountOp:
    def run(self, xs: List[int]) -> int:
        return len(xs)


def test_shard_correct_semantics_vs_ref0_shortcut(ray_session):
    """
    同一测例里对照“理论正确路径”与“refs[0]捷径错误路径”：
    - 正确：shard 前后语义应保持 batch 完整（10 -> 10）
    - 错误：只拿 refs[0] 会把下游输入退化成第一个分片（10 -> 4）
    """
    ws = 3
    sharder = RayModule(
        IdentityBatchOp,
        replicas=ws,
        dispatch_fn=dispatch_shard_all_args_mod,
        collect_fn=collect_concat,
    ).pre_init()
    counter = RayModule(CountOp, replicas=1).pre_init()
    try:
        batch = list(range(10))  # ws=3 => shard sizes [4,3,3]

        # 理论正确基准：按 shard 执行后再 gather，语义应与原 batch 一致。
        direct_full = sharder(batch)
        assert direct_full == batch

        upstream = sharder.remote(batch)

        # 正确异步路径：remote 后 gather，再下游处理完整 batch
        full_batch = sharder.gather(upstream)
        assert full_batch == batch
        good_count = counter(full_batch)
        assert good_count == len(batch) == 10

        # 错误路径：只拿 refs[0] 当作“整个 batch”
        bad = counter.remote(upstream.completion_refs()[0])
        bad_count = counter.gather(bad)
        assert bad_count == 4
        assert bad_count == len(batch[:4])
        assert bad_count < good_count
        assert bad_count != len(batch)
    finally:
        _kill_modules(sharder, counter)


def test_overlapped_rejects_multi_ref_future_for_shard_chain(ray_session):
    """
    Overlapped 链路下，stage1 为 shard 多 refs 时，不能默认 refs[0] 透传到 stage2。
    期望：抛错，避免 silent data loss。
    """

    class Pipe(OverlappedPipeline):
        def __init__(self):
            self.s1 = RayModule(
                IdentityBatchOp,
                replicas=2,
                dispatch_fn=dispatch_shard_all_args_mod,
                collect_fn=collect_concat,
            ).pre_init()
            self.s2 = RayModule(CountOp, replicas=1).pre_init()
            super().__init__(max_inflight=2)

        def forward(self, x):
            return self.s2(self.s1(x))

    p = Pipe()
    try:
        with pytest.raises(ValueError, match="multiple completion refs"):
            p([list(range(8))])
    finally:
        _kill_modules(p.s1, p.s2)


def test_target_behavior_overlapped_shard_chain_auto_join_reduce(ray_session):
    """
    监督用“目标行为”测试（当前预期会失败）：
    - 不要求用户手工 gather
    - Overlapped 在 stage 边界自动完成 join/reduce
    - 下游应看到完整 batch（而不是 refs[0] 的首 shard）
    """

    class Pipe(OverlappedPipeline):
        def __init__(self):
            self.s1 = RayModule(
                IdentityBatchOp,
                replicas=3,
                dispatch_fn=dispatch_shard_all_args_mod,
                collect_fn=collect_concat,
            ).pre_init()
            self.s2 = RayModule(CountOp, replicas=1).pre_init()
            super().__init__(max_inflight=2)

        def forward(self, x):
            # 这是你希望最终支持的写法：只重载 forward，不手工 gather。
            return self.s2(self.s1(x))

    p = Pipe()
    try:
        out = p([list(range(10))])
        # 目标语义：完整 batch 长度=10
        assert out == [10]
    finally:
        _kill_modules(p.s1, p.s2)

