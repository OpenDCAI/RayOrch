"""Docling Native 四 GPU runner 的无 GPU 单元测试。"""

from __future__ import annotations

import pytest

from rayorch.experimental.multigrain_v3.benchmark.document_docling.core_native import (
    NativeCoreConfig,
)
from rayorch.experimental.multigrain_v3.benchmark.document_docling.core_native_multi import (
    _aggregate_worker_payloads,
    _partition_indices_lpt,
    _validate_inputs,
)


def test_lpt_partition_is_deterministic_page_balanced_and_source_sorted() -> None:
    """LPT 应优先均衡页数、稳定打破平局，并在各 shard 恢复 source 顺序。"""

    partitions = _partition_indices_lpt(
        page_counts=(8, 8, 8, 8, 4, 4, 4, 4),
        sizes=(80, 70, 60, 50, 40, 30, 20, 10),
    )

    assert partitions == ((0, 7), (1, 6), (2, 5), (3, 4))
    assert [
        sum((8, 8, 8, 8, 4, 4, 4, 4)[index] for index in shard)
        for shard in partitions
    ] == [12, 12, 12, 12]


def test_lpt_uses_bytes_only_for_equal_page_loads() -> None:
    """页数相同时，较大的文件应优先入队并以字节负载稳定打破 worker 平局。"""

    partitions = _partition_indices_lpt(
        page_counts=(1, 1, 1, 1, 1),
        sizes=(10, 50, 40, 30, 20),
    )

    assert partitions == ((1,), (2,), (3,), (0, 4))


def test_multi_gpu_configuration_validation() -> None:
    """四 GPU baseline 必须拒绝不完整 GPU 列表、错误元数据和 CPU 配置。"""

    config = NativeCoreConfig(device="cuda")
    with pytest.raises(ValueError, match="exactly four"):
        _validate_inputs(["a.pdf"], [1], [1], config, (0, 1, 2))
    with pytest.raises(ValueError, match="unique"):
        _validate_inputs(["a.pdf"], [1], [1], config, (0, 1, 2, 2))
    with pytest.raises(ValueError, match="same length"):
        _validate_inputs(["a.pdf"], [], [1], config, (0, 1, 2, 3))
    with pytest.raises(ValueError, match="CUDA"):
        _validate_inputs(
            ["a.pdf"],
            [1],
            [1],
            NativeCoreConfig(device="cpu"),
            (0, 1, 2, 3),
        )


def test_fake_worker_payloads_aggregate_to_original_source_order() -> None:
    """聚合层只依赖可序列化 worker payload，可由 fake worker/monkeypatch 独立测试。"""

    payloads = [
        {
            "gpu_slot": 2,
            "source_indices": [2],
            "doc_count": 1,
            "documents": [{"source_index": 2, "document": {"markdown": "C"}}],
        },
        {
            "gpu_slot": 0,
            "source_indices": [0],
            "doc_count": 1,
            "documents": [{"source_index": 0, "document": {"markdown": "A"}}],
        },
        {
            "gpu_slot": 3,
            "source_indices": [],
            "doc_count": 0,
            "documents": [],
        },
        {
            "gpu_slot": 1,
            "source_indices": [1],
            "doc_count": 1,
            "documents": [{"source_index": 1, "document": {"markdown": "B"}}],
        },
    ]

    documents, records = _aggregate_worker_payloads(
        ["a.pdf", "b.pdf", "c.pdf"],
        payloads,
    )

    assert [document["markdown"] for document in documents] == ["A", "B", "C"]
    assert [record["gpu_slot"] for record in records] == [0, 1, 2, 3]


def test_fake_worker_payload_rejects_missing_source() -> None:
    """聚合必须检测 fake 或真实 worker 的漏文档结果。"""

    payloads = [
        {
            "gpu_slot": slot,
            "source_indices": [],
            "doc_count": 0,
            "documents": [],
        }
        for slot in range(4)
    ]

    with pytest.raises(RuntimeError, match="cover"):
        _aggregate_worker_payloads(["a.pdf"], payloads)
