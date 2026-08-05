"""Docling core-stage 四臂矩阵的无模型配置与 correctness 测试。"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from rayorch.experimental.multigrain_v3.benchmark.document_docling.core_compare import (
    ARM_ORDER,
    CoreMatrixConfig,
    _compare_documents,
    _timeline_summary,
    _write_timeline,
    build_matrix_plan,
    counterbalanced_arm_order,
)
from rayorch.experimental.multigrain_v3.benchmark.document_docling.core_native import (
    NativeCoreConfig,
)


def test_four_arm_plan_freezes_only_document_concurrency_and_scope() -> None:
    """四臂计划必须固定 heavy-stage cap，且 V3 两臂只切 batch_scope。"""

    config = CoreMatrixConfig(
        device="cuda",
        ocr_device="cpu",
        stage_batch_size=4,
        native_tuned_doc_concurrency=4,
    )

    plan = build_matrix_plan(12, config)
    by_name = {arm.name: arm for arm in plan}

    assert tuple(arm.name for arm in plan) == ARM_ORDER
    assert by_name["native_default"].options["doc_batch_size"] == 1
    assert by_name["native_default"].options["doc_batch_concurrency"] == 1
    assert by_name["native_tuned"].options["doc_batch_size"] == 12
    assert by_name["native_tuned"].options["doc_batch_concurrency"] == 4

    parent = dict(by_name["v3_parent_bound"].options)
    elastic = dict(by_name["v3_elastic"].options)
    assert parent.pop("batch_scope") == "parent_bound"
    assert elastic.pop("batch_scope") == "elastic"
    assert parent == elastic
    assert parent["layout_batch_size"] == 4
    assert parent["ocr_batch_size"] == 4
    assert parent["table_batch_size"] == 4


def test_table_encoder_acceleration_is_wired_into_both_v3_arms() -> None:
    """Table P1 参数必须进入 V3 plan，且不改变 parent/elastic 对照合同。"""

    config = CoreMatrixConfig(
        table_batch_mode="encoder_accelerated",
        table_batch_max_jobs=12,
    )
    plan = build_matrix_plan(12, config)
    parent = dict(plan[2].options)
    elastic = dict(plan[3].options)

    assert parent["table_batch_mode"] == "encoder_accelerated"
    assert parent["table_batch_max_jobs"] == 12
    assert parent.pop("batch_scope") == "parent_bound"
    assert elastic.pop("batch_scope") == "elastic"
    assert parent == elastic


def test_table_core_batch_size_is_independent_in_both_v3_arms() -> None:
    """TableCore 可以缩小物理 RPC，而 Page stages 继续使用共享 cap。"""

    plan = build_matrix_plan(
        12,
        CoreMatrixConfig(stage_batch_size=16, table_core_batch_size=4),
    )
    parent = dict(plan[2].options)
    elastic = dict(plan[3].options)

    assert parent["table_batch_size"] == 16
    assert parent["table_core_batch_size"] == 4
    assert parent.pop("batch_scope") == "parent_bound"
    assert elastic.pop("batch_scope") == "elastic"
    assert parent == elastic


def test_table_encoder_batching_requires_single_actor_concurrency() -> None:
    """临时 encoder hook 不得与同 actor 的并发 RPC 交错。"""

    with pytest.raises(ValueError, match="table_actor_concurrency"):
        CoreMatrixConfig(
            table_batch_mode="encoder_accelerated",
            table_actor_concurrency=2,
        )


def test_four_gpu_plan_uses_multi_process_native_and_four_static_v3_gpus() -> None:
    """四卡矩阵必须让每个 arm 内部使用四卡，而不是四个 arm 各跑一张卡。"""

    config = CoreMatrixConfig(
        device="cuda",
        ocr_device="cpu",
        stage_batch_size=8,
        layout_replicas=1,
        ocr_replicas=4,
        table_replicas=3,
        reduce_replicas=4,
        layout_num_gpus=1.0,
        table_num_gpus=1.0,
        four_gpu_native=True,
    )

    plan = build_matrix_plan(48, config)
    by_name = {arm.name: arm for arm in plan}

    assert by_name["native_default"].kind == "native_multi"
    assert by_name["native_tuned"].kind == "native_multi"
    parent = by_name["v3_parent_bound"].options
    assert parent["layout_replicas"] == 1
    assert parent["table_replicas"] == 3
    assert parent["ocr_replicas"] == 4
    assert parent["reduce_replicas"] == 4
    assert (
        parent["layout_replicas"] * parent["layout_num_gpus"]
        + parent["table_replicas"] * parent["table_num_gpus"]
    ) == 4.0


def test_tuned_native_requires_document_batch_to_cover_threads() -> None:
    """Docling concurrent conversion 配置不能产生永远用不满的非法 batch。"""

    with pytest.raises(ValueError, match="doc_batch_size"):
        NativeCoreConfig(doc_batch_size=2, doc_batch_concurrency=4)

    config = CoreMatrixConfig(native_tuned_doc_concurrency=4)
    with pytest.raises(ValueError, match="smaller than concurrency"):
        config.resolved_native_tuned_batch_size(3)


def test_correctness_gate_compares_markdown_and_structure() -> None:
    """矩阵 JSON 必须将 Markdown 与 page/text/table/picture 分开报告。"""

    native = (
        {
            "markdown": "A B C",
            "pages": 2,
            "texts": 3,
            "tables": 1,
            "pictures": 0,
        },
    )
    matching = (
        {
            "markdown": "C B A",
            "pages": 2,
            "texts": 3,
            "tables": 1,
            "pictures": 0,
        },
    )
    mismatch = (
        {
            "markdown": "A B",
            "pages": 2,
            "texts": 4,
            "tables": 1,
            "pictures": 0,
        },
    )

    assert _compare_documents(native, matching) == {
        "document_count_matches": True,
        "markdown_jaccard": (1.0,),
        "structure_exact": (True,),
    }
    assert _compare_documents(native, mismatch) == {
        "document_count_matches": True,
        "markdown_jaccard": (0.66666667,),
        "structure_exact": (False,),
    }


def test_matrix_counterbalances_arm_order_without_changing_plan() -> None:
    """不同 repeat 应轮转首臂，避免固定 GPU/file-cache 顺序偏差。"""

    plan = build_matrix_plan(12, CoreMatrixConfig())

    assert tuple(
        arm.name for arm in counterbalanced_arm_order(plan, 0)
    ) == ARM_ORDER
    assert tuple(
        arm.name for arm in counterbalanced_arm_order(plan, 1)
    ) == (
        "native_tuned",
        "v3_elastic",
        "v3_parent_bound",
        "native_default",
    )
    assert tuple(
        arm.name for arm in counterbalanced_arm_order(plan, 4)
    ) == ARM_ORDER

    four_orders = [
        tuple(arm.name for arm in counterbalanced_arm_order(plan, index))
        for index in range(4)
    ]
    assert sum(
        order.index("v3_elastic") < order.index("v3_parent_bound")
        for order in four_orders
    ) == 2


def test_timeline_summary_keeps_page_stage_batch_histograms() -> None:
    """Docling 报告必须能区分 Layout/OCR/Table 的实际 batch shape。"""

    from types import SimpleNamespace

    events = [
        SimpleNamespace(
            stage=2,
            grains=4,
            flush_reason="full",
            worker_started_at=1.0,
            worker_finished_at=2.0,
        ),
        SimpleNamespace(
            stage=2,
            grains=2,
            flush_reason="timeout",
            worker_started_at=1.5,
            worker_finished_at=2.5,
        ),
        SimpleNamespace(
            stage=3,
            grains=3,
            flush_reason="timeout",
            worker_started_at=2.5,
            worker_finished_at=3.5,
        ),
    ]

    summary = _timeline_summary(events)

    assert summary["layout"]["rpc_count"] == 2
    assert summary["layout"]["grains_per_rpc"] == 3.0
    assert summary["layout"]["batch_histogram"] == {2: 1, 4: 1}
    assert summary["layout"]["peak_concurrency"] == 2
    assert summary["ocr"]["flush_reasons"] == {"timeout": 1}


def test_write_timeline_preserves_raw_event_fields(tmp_path) -> None:
    """原始 timeline artifact 必须保留 arena 与 worker 时间字段。"""

    @dataclass(frozen=True)
    class Event:
        arena_id: int
        stage: int
        grains: int
        flush_reason: str
        worker_started_at: float
        worker_finished_at: float

    output = tmp_path / "timeline.jsonl"
    _write_timeline(
        str(output),
        (
            Event(
                arena_id=3,
                stage=2,
                grains=16,
                flush_reason="full",
                worker_started_at=1.25,
                worker_finished_at=2.5,
            ),
        ),
    )

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "arena_id": 3,
        "stage": 2,
        "grains": 16,
        "flush_reason": "full",
        "worker_started_at": 1.25,
        "worker_finished_at": 2.5,
    }
