"""
RayModule 与 dispatch / collect 的配合：端到端 map-reduce 行为。

- Map：dispatch_fn 把一次调用的 *args/**kwargs 切成每 replica 一份，经 actor 上 op.run。
- Reduce：各 replica 返回值列表交给 collect_fn，得到对外的单一结果。

RayOrch 的 dispatch_fn 采用“列式（按参数槽位）”协议：
- per_args: tuple，每个元素是 len=replicas 的 list/sequence，表示该参数在每个 replica 上的值
- per_kwargs: dict，每个 value 是 len=replicas 的 list/sequence

RayModule 负责把列式 per_args/per_kwargs 转置成每个 actor 的 (args_i, kwargs_i)。
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import pytest

import ray

from rayorch import RayModule
from rayorch.dispatch_mode import (
    collect_identity,
    collect_concat,
    dispatch_per_replica,
    dispatch_broadcast,
    dispatch_shard_all_args_mod,
)

pytestmark = pytest.mark.usefixtures("ray_cluster")


def _kill_modules(*modules: RayModule) -> None:
    for m in modules:
        for actor in getattr(m, "actors", []):
            try:
                ray.kill(actor)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Ops：尽量小，只表达 map 语义
# ---------------------------------------------------------------------------


class ScaleSumOp:
    """run(x, scale) -> x * scale；用于 one-to_all 各副本相同参数。"""

    def __init__(self) -> None:
        pass

    def run(self, x: int, scale: int) -> int:
        return x * scale


class VarPackOp:
    """原样返回 (args, kwargs)，检查 broadcast 是否正确。"""

    def __init__(self) -> None:
        pass

    def run(self, *args: Any, **kwargs: Any) -> Any:
        return (args, kwargs)


class PairAddOp:
    def __init__(self) -> None:
        pass

    def run(self, a: int, b: int) -> int:
        return a + b


class ChunkAccOp:
    """run(chunk, bias) -> [sum(chunk)+bias] 便于 collect_concat 拼 list。"""

    def __init__(self) -> None:
        pass

    def run(self, chunk: list, bias: int) -> list:
        return [sum(chunk) + bias]


class DictShardOp:
    """每 rank 返回 {'items': [...]}，reduce 后 items 按 rank 顺序 extend。"""

    def __init__(self) -> None:
        pass

    def run(self, chunk: list) -> dict:
        return {"items": list(chunk)}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_one_to_all_map_identical_reduce_list():
    """Map：每 replica 同一 (x, scale)；Reduce：collect 原样列表。"""
    ws = 3
    m = RayModule(
        ScaleSumOp,
        replicas=ws,
        dispatch_fn=dispatch_broadcast,
        collect_fn=collect_identity,
    ).pre_init()
    try:
        got = m(7, 4)
        assert got == [28, 28, 28]
    finally:
        _kill_modules(m)


def test_one_to_all_many_args_kwargs_each_replica_sees_same():
    ws = 4
    m = RayModule(
        VarPackOp,
        replicas=ws,
        dispatch_fn=dispatch_broadcast,
        collect_fn=collect_identity,
    ).pre_init()
    try:
        p_args = (1, "hi", (3, 4), None)
        p_kw = {"a": 7, "flag": True}
        got = m(*p_args, **p_kw)
        assert isinstance(got, list) and len(got) == ws
        for item in got:
            ga, gkw = item
            assert ga == p_args
            assert gkw == p_kw
    finally:
        _kill_modules(m)


def test_one_to_all_custom_reduce_sum():
    """Reduce：对 map 输出做 sum（各副本标量相同也可验 reduce 被调用）。"""
    ws = 3
    m = RayModule(
        ScaleSumOp,
        replicas=ws,
        dispatch_fn=dispatch_broadcast,
        collect_fn=lambda rm, outs: sum(outs),
    ).pre_init()
    try:
        assert m(5, 2) == 30
    finally:
        _kill_modules(m)


def test_all_to_all_map_per_rank_reduce_list():
    """每 rank 不同 (a,b)；reduce 保留 list。"""
    ws = 3
    m = RayModule(
        PairAddOp,
        replicas=ws,
        dispatch_fn=dispatch_per_replica,
        collect_fn=collect_identity,
    ).pre_init()
    try:
        got = m((1, 2, 3), (10, 100, 1000))
        assert got == [11, 102, 1003]
    finally:
        _kill_modules(m)


def test_shard_map_slice_reduce_concat_list():
    """Map：batch 切到各 rank；Reduce：collect_concat 按 rank 拼 batch。"""
    ws = 2
    m = RayModule(
        ChunkAccOp,
        replicas=ws,
        dispatch_fn=dispatch_shard_all_args_mod,
        collect_fn=collect_concat,
    ).pre_init()
    try:
        out = m(list(range(4)), 100)
        # rank0 [0,1] sum+100=101 -> [101]; rank1 [2,3] sum+100=105 -> [105] => [101,105]
        assert out == [101, 105]
    finally:
        _kill_modules(m)


def test_shard_matches_manual_serial_reduce():
    """与手写分片 + 串行 op + 拼接对齐（map-reduce 数值预期）。"""
    ws = 3
    batch = list(range(10))
    bias = 1
    # 手写分片（与 dispatch_shard_all_args_mod 相同的区间策略）再串行执行
    base = len(batch) // ws
    rem = len(batch) % ws
    sizes = [base + (1 if i < rem else 0) for i in range(ws)]
    ranges = []
    start = 0
    for sz in sizes:
        end = start + sz
        ranges.append((start, end))
        start = end

    manual: list = []
    op = ChunkAccOp()
    for (s, e) in ranges:
        manual.extend(op.run(batch[s:e], bias))

    m = RayModule(
        ChunkAccOp,
        replicas=ws,
        dispatch_fn=dispatch_shard_all_args_mod,
        collect_fn=collect_concat,
    ).pre_init()
    try:
        assert m(batch, bias) == manual
    finally:
        _kill_modules(m)


def test_shard_dict_outputs_collect_concat():
    ws = 2
    m = RayModule(
        DictShardOp,
        replicas=ws,
        dispatch_fn=dispatch_shard_all_args_mod,
        collect_fn=collect_concat,
    ).pre_init()
    try:
        got = m([1, 2, 3, 4])
        assert got == {"items": [1, 2, 3, 4]}
    finally:
        _kill_modules(m)


def test_remote_gather_same_as_call():
    ws = 2
    m = RayModule(
        PairAddOp,
        replicas=ws,
        dispatch_fn=dispatch_per_replica,
        collect_fn=collect_identity,
    ).pre_init()
    try:
        direct = m((3, 4), (30, 40))
        fut = m.remote((3, 4), (30, 40))
        via_gather = m.gather(fut)
        assert direct == via_gather == [33, 44]
    finally:
        _kill_modules(m)


def test_shard_length_mismatch_raises_before_actors():
    ws = 2
    m = RayModule(
        ChunkAccOp,
        replicas=ws,
        dispatch_fn=dispatch_shard_all_args_mod,
        collect_fn=collect_concat,
    ).pre_init()
    try:
        with pytest.raises(ValueError, match="Shardable"):
            m([1, 2], [1, 2, 3])
    finally:
        _kill_modules(m)
