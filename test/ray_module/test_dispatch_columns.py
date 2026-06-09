from __future__ import annotations

from typing import Any, List, Tuple

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


class EchoPairOp:
    """
    期望每个 replica 收到的都是：
    - images: List[Any]
    - texts:  List[str]
    """

    def run(self, images: List[Any], texts: List[str]) -> Tuple[List[Any], List[str]]:
        assert isinstance(images, list)
        assert isinstance(texts, list)
        assert len(images) == len(texts)
        return images, texts


class PackOp:
    """
    用于 shard + collect_concat：
    每个 replica 返回一个 “batch list”，元素是 (image, text, replica_id)。
    """

    def run(self, images: List[Any], texts: List[str], replica_id: int) -> List[tuple]:
        assert isinstance(images, list)
        assert isinstance(texts, list)
        assert len(images) == len(texts)
        return [(images[i], texts[i], replica_id) for i in range(len(images))]


class KwShardOp:
    """
    shardable 在 kwargs 里；args 里是广播参数。
    返回 batch list，元素为 (idx, payload, tag)。
    """

    def run(self, tag: str, idx: List[int], payload: List[str]) -> List[tuple]:
        assert isinstance(idx, list)
        assert isinstance(payload, list)
        assert len(idx) == len(payload)
        return [(idx[i], payload[i], tag) for i in range(len(idx))]


class DictOutOp:
    """
    每 rank 返回 dict，其中包含 batch list 字段；collect_concat 需要递归拼接。
    """

    def run(self, xs: List[int]) -> dict:
        return {"items": list(xs), "count": len(xs)}


class TupleOutOp:
    """
    每 rank 返回 (batch_list, meta_dict)，collect_concat 需要对 tuple 按槽位递归 merge。
    """

    def run(self, xs: List[int], prefix: str) -> tuple:
        return ([f"{prefix}{x}" for x in xs], {"prefix": prefix})


class IdentityOp:
    def run(self, x: Any) -> Any:
        return x


def test_column_dispatch_broadcast_accepts_list_batches():
    """
    验证 RayModule 能直接对接 `dispatch_broadcast` 的列式输出：
    - 输入是 list_of_images/list_of_texts
    - 每个 replica 都收到同一份 lists
    - reduce 使用 collect_identity，返回每个 replica 的输出列表
    """
    ws = 3
    m = RayModule(
        EchoPairOp,
        replicas=ws,
        dispatch_fn=dispatch_broadcast,
        collect_fn=collect_identity,
    ).pre_init()
    try:
        images = [{"id": i} for i in range(5)]
        texts = [f"t{i}" for i in range(5)]
        outs = m(images, texts)
        assert isinstance(outs, list) and len(outs) == ws
        for (got_images, got_texts) in outs:
            assert got_images == images
            assert got_texts == texts
    finally:
        _kill_modules(m)


def test_column_dispatch_shard_all_args_mod_shards_and_collects():
    """
    验证 RayModule 能直接对接 `dispatch_shard_all_args_mod` 的列式输出：
    - 输入是等长的 list_of_images/list_of_texts
    - 每个 replica 收到连续分片
    - collect_concat 会按 rank 顺序拼回整体顺序
    """
    ws = 3
    m = RayModule(
        PackOp,
        replicas=ws,
        dispatch_fn=dispatch_shard_all_args_mod,
        collect_fn=collect_concat,
    ).pre_init()
    try:
        n = 10
        images = [{"id": i} for i in range(n)]
        texts = [f"t{i}" for i in range(n)]
        out = m(images, texts, 999)  # replica_id 是广播参数

        assert isinstance(out, list) and len(out) == n
        # 顺序必须与输入一致（collect_concat 按 rank 顺序拼接连续区间）
        assert [img for (img, _txt, _rid) in out] == images
        assert [txt for (_img, txt, _rid) in out] == texts
        # 广播参数对每条记录一致
        assert all(rid == 999 for (_img, _txt, rid) in out)
    finally:
        _kill_modules(m)


def test_column_dispatch_shard_supports_kwargs_shard_and_args_broadcast():
    """
    - shardable 参数放在 kwargs（idx/payload）
    - args 中 tag 是广播参数
    - reduce 用 collect_concat 拼 batch
    """
    ws = 4
    m = RayModule(
        KwShardOp,
        replicas=ws,
        dispatch_fn=dispatch_shard_all_args_mod,
        collect_fn=collect_concat,
    ).pre_init()
    try:
        idx = list(range(9))
        payload = [f"p{i}" for i in range(9)]
        out = m("TAG", idx=idx, payload=payload)
        assert out == [(i, f"p{i}", "TAG") for i in range(9)]
    finally:
        _kill_modules(m)


def test_column_dispatch_shard_length_mismatch_raises():
    """
    shardable 参数长度不一致要在 dispatch 阶段报错（不应进入 actor.run）。
    """
    ws = 2
    m = RayModule(
        PackOp,
        replicas=ws,
        dispatch_fn=dispatch_shard_all_args_mod,
        collect_fn=collect_concat,
    ).pre_init()
    try:
        with pytest.raises(ValueError, match="Shardable"):
            m([{"id": 1}], ["t0", "t1"], 0)
    finally:
        _kill_modules(m)


def test_column_collect_concat_merges_dict_batch_fields():
    """
    collect_concat 对 dict 输出：
    - items: batch list 要按 rank 顺序拼接回完整 list
    - count: 标量默认保持第一个（通常是广播/常量）；这里验证不会报错
    """
    ws = 3
    m = RayModule(
        DictOutOp,
        replicas=ws,
        dispatch_fn=dispatch_shard_all_args_mod,
        collect_fn=collect_concat,
    ).pre_init()
    try:
        out = m(list(range(10)))
        assert out["items"] == list(range(10))
        assert "count" in out
    finally:
        _kill_modules(m)


def test_column_collect_concat_merges_tuple_outputs():
    ws = 3
    m = RayModule(
        TupleOutOp,
        replicas=ws,
        dispatch_fn=dispatch_shard_all_args_mod,
        collect_fn=collect_concat,
    ).pre_init()
    try:
        out_list, meta = m(list(range(7)), "x")
        assert out_list == [f"x{i}" for i in range(7)]
        assert meta == {"prefix": "x"}
    finally:
        _kill_modules(m)


def test_column_dispatch_all_to_all_column_style():
    """
    dispatch_per_replica 约定：输入已经是列式 per-replica 形式。
    这里用一个最小 identity op 验证每个 replica 各自拿到自己的值。
    """
    ws = 3
    m = RayModule(
        IdentityOp,
        replicas=ws,
        dispatch_fn=dispatch_per_replica,
        collect_fn=collect_identity,
    ).pre_init()
    try:
        out = m([10, 11, 12])
        assert out == [10, 11, 12]
    finally:
        _kill_modules(m)


def test_column_remote_gather_path():
    """
    remote()/gather() 路径也必须兼容列式 dispatch。
    """
    ws = 2
    m = RayModule(
        PackOp,
        replicas=ws,
        dispatch_fn=dispatch_shard_all_args_mod,
        collect_fn=collect_concat,
    ).pre_init()
    try:
        images = [{"id": i} for i in range(6)]
        texts = [f"t{i}" for i in range(6)]
        fut = m.remote(images, texts, 7)
        out = m.gather(fut)
        assert [img for (img, _t, _rid) in out] == images
        assert [t for (_img, t, _rid) in out] == texts
        assert all(rid == 7 for (_img, _t, rid) in out)
    finally:
        _kill_modules(m)

