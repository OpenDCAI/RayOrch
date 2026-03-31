"""
RayModule 与 dispatch / collect 的配合：端到端 map-reduce 行为。

- Map：dispatch_fn 把一次调用的 *args/**kwargs 切成每 replica 一份，经 actor 上 op.run。
- Reduce：各 replica 返回值列表交给 collect_fn，得到对外的单一结果。

``dispatch_all_to_all`` / ``dispatch_shard_all_args_mod`` 返回的是 shard 风格（按参数位列），
需 zip 成 RayModule 要求的 per-replica 两个 list；本文件用 ``zip_shard_dispatch`` 包装，属真实场景中常见的 adapter。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Tuple

import pytest

import ray

from rayorch import RayModule
from rayorch.dispatch_mode import (
    collect_all_to_all,
    collect_concat,
    dispatch_all_to_all,
    dispatch_one_to_all,
    dispatch_shard_all_args_mod,
)


def zip_shard_dispatch(rm: RayModule, *args: Any, **kwargs: Any):
    """把 shard 列式 dispatch 输出 zip 成 RayModule 协议。"""
    ws = rm._replicas
    per_args, per_kw = dispatch_shard_all_args_mod(rm, *args, **kwargs)
    for j, col in enumerate(per_args):
        if len(col) != ws:
            raise ValueError(f"dispatch shard: arg slot {j} len {len(col)} != replicas {ws}")
    for k, col in per_kw.items():
        if len(col) != ws:
            raise ValueError(f"dispatch shard: kw {k!r} len {len(col)} != replicas {ws}")
    ra = [tuple(per_args[j][i] for j in range(len(per_args))) for i in range(ws)]
    rk = [{kk: per_kw[kk][i] for kk in per_kw} for i in range(ws)]
    return ra, rk


def zip_all_to_all_dispatch(rm: RayModule, *args: Any, **kwargs: Any):
    per_args, per_kw = dispatch_all_to_all(rm, *args, **kwargs)
    return zip_shard_style_columns(rm._replicas, per_args, per_kw)


def zip_shard_style_columns(
    ws: int,
    per_args: Tuple[Sequence[Any], ...],
    per_kw: Dict[str, Any],
) -> Tuple[List[Tuple[Any, ...]], List[Dict[str, Any]]]:
    for j, col in enumerate(per_args):
        if len(col) != ws:
            raise ValueError(f"arg slot {j} len {len(col)} != {ws}")
    for k, col in per_kw.items():
        if len(col) != ws:
            raise ValueError(f"kw {k!r} len {len(col)} != {ws}")
    ra = [tuple(per_args[j][i] for j in range(len(per_args))) for i in range(ws)]
    rk = [{kk: per_kw[kk][i] for kk in per_kw} for i in range(ws)]
    return ra, rk


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


@pytest.fixture
def ray_session():
    ray.init(ignore_reinit_error=True, num_cpus=16)
    yield
    if ray.is_initialized():
        ray.shutdown()


def test_one_to_all_map_identical_reduce_list(ray_session):
    """Map：每 replica 同一 (x, scale)；Reduce：collect 原样列表。"""
    ws = 3
    m = RayModule(
        ScaleSumOp,
        replicas=ws,
        dispatch_fn=dispatch_one_to_all,
        collect_fn=collect_all_to_all,
    ).pre_init()
    try:
        got = m(7, 4)
        assert got == [28, 28, 28]
    finally:
        _kill_modules(m)


def test_one_to_all_many_args_kwargs_each_replica_sees_same(ray_session):
    ws = 4
    m = RayModule(
        VarPackOp,
        replicas=ws,
        dispatch_fn=dispatch_one_to_all,
        collect_fn=collect_all_to_all,
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


def test_one_to_all_custom_reduce_sum(ray_session):
    """Reduce：对 map 输出做 sum（各副本标量相同也可验 reduce 被调用）。"""
    ws = 3
    m = RayModule(
        ScaleSumOp,
        replicas=ws,
        dispatch_fn=dispatch_one_to_all,
        collect_fn=lambda rm, outs: sum(outs),
    ).pre_init()
    try:
        assert m(5, 2) == 30
    finally:
        _kill_modules(m)


def test_all_to_all_map_per_rank_reduce_list(ray_session):
    """每 rank 不同 (a,b)；reduce 保留 list。"""
    ws = 3
    m = RayModule(
        PairAddOp,
        replicas=ws,
        dispatch_fn=zip_all_to_all_dispatch,
        collect_fn=collect_all_to_all,
    ).pre_init()
    try:
        got = m((1, 2, 3), (10, 100, 1000))
        assert got == [11, 102, 1003]
    finally:
        _kill_modules(m)


def test_shard_map_slice_reduce_concat_list(ray_session):
    """Map：batch 切到各 rank；Reduce：collect_concat 按 rank 拼 batch。"""
    ws = 2
    m = RayModule(
        ChunkAccOp,
        replicas=ws,
        dispatch_fn=zip_shard_dispatch,
        collect_fn=collect_concat,
    ).pre_init()
    try:
        out = m(list(range(4)), 100)
        # rank0 [0,1] sum+100=101 -> [101]; rank1 [2,3] sum+100=105 -> [105] => [101,105]
        assert out == [101, 105]
    finally:
        _kill_modules(m)


def test_shard_matches_manual_serial_reduce(ray_session):
    """与手写分片 + 串行 op + 拼接对齐（map-reduce 数值预期）。"""
    ws = 3
    batch = list(range(10))
    bias = 1
    pa, pk = dispatch_shard_all_args_mod(SimpleNamespace(_replicas=ws), batch, bias)
    ra, _ = zip_shard_style_columns(ws, pa, pk)

    manual: list = []
    for args_i in ra:
        manual.extend(ChunkAccOp().run(*args_i))

    m = RayModule(
        ChunkAccOp,
        replicas=ws,
        dispatch_fn=zip_shard_dispatch,
        collect_fn=collect_concat,
    ).pre_init()
    try:
        assert m(batch, bias) == manual
    finally:
        _kill_modules(m)


def test_shard_dict_outputs_collect_concat(ray_session):
    ws = 2
    m = RayModule(
        DictShardOp,
        replicas=ws,
        dispatch_fn=zip_shard_dispatch,
        collect_fn=collect_concat,
    ).pre_init()
    try:
        got = m([1, 2, 3, 4])
        assert got == {"items": [1, 2, 3, 4]}
    finally:
        _kill_modules(m)


def test_remote_gather_same_as_call(ray_session):
    ws = 2
    m = RayModule(
        PairAddOp,
        replicas=ws,
        dispatch_fn=zip_all_to_all_dispatch,
        collect_fn=collect_all_to_all,
    ).pre_init()
    try:
        direct = m((3, 4), (30, 40))
        fut = m.remote((3, 4), (30, 40))
        via_gather = m.gather(fut)
        assert direct == via_gather == [33, 44]
    finally:
        _kill_modules(m)


def test_shard_length_mismatch_raises_before_actors(ray_session):
    ws = 2
    m = RayModule(
        ChunkAccOp,
        replicas=ws,
        dispatch_fn=zip_shard_dispatch,
        collect_fn=collect_concat,
    ).pre_init()
    try:
        with pytest.raises(ValueError, match="Shardable"):
            m([1, 2], [1, 2, 3])
    finally:
        _kill_modules(m)
