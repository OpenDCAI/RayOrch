"""TableFormer 保守 batching 作业层的无依赖测试。"""

from __future__ import annotations

from rayorch.experimental.multigrain_v3.benchmark.document_docling.table_batching import (
    EncoderBatchedKernelAdapter,
    TableBatchExecutor,
    TableBatchPlanner,
    TableJob,
    TablePrediction,
    TablePredictionComparator,
    restore_table_order,
)


def _job(
    page: int,
    table: int,
    order: int,
    payload: str,
    *,
    bucket: str = "default",
) -> TableJob:
    return TableJob(
        page=page,
        table=table,
        order=order,
        payload=payload,
        metadata={"batch_key": bucket},
    )


def test_planner_stably_buckets_and_splits_jobs() -> None:
    """同 bucket 保持输入顺序，bucket 按首次出现顺序且可稳定切分。"""

    first_a = _job(0, 0, 0, "a-1", bucket="a")
    only_b = _job(0, 1, 0, "b-1", bucket="b")
    second_a = _job(1, 0, 0, "a-2", bucket="a")
    third_a = _job(1, 1, 0, "a-3", bucket="a")

    batches = TableBatchPlanner(max_batch_size=2).plan(
        [first_a, only_b, second_a, third_a]
    )

    assert [batch.bucket_key for batch in batches] == ["a", "a", "b"]
    assert [[job.payload for job in batch.jobs] for batch in batches] == [
        ["a-1", "a-2"],
        ["a-3"],
        ["b-1"],
    ]


def test_executor_restores_document_order_after_bucket_execution() -> None:
    """执行可按 bucket 重排，但最终输出必须按 page/table/order 恢复。"""

    jobs = [
        _job(1, 0, 0, "page-1", bucket="late"),
        _job(0, 1, 0, "page-0-table-1", bucket="early"),
        _job(0, 0, 1, "page-0-table-0-order-1", bucket="late"),
        _job(0, 0, 0, "page-0-table-0-order-0", bucket="early"),
    ]
    batch_payloads: list[list[str]] = []

    def batch_callable(batch_jobs: list[TableJob]) -> list[TablePrediction]:
        batch_payloads.append([job.payload for job in batch_jobs])
        return [TablePrediction(job.payload, {"runtime_ms": 1}) for job in batch_jobs]

    executor = TableBatchExecutor(
        TableBatchPlanner(),
        reference_callable=lambda job: TablePrediction(job.payload, {"path": "reference"}),
        batch_callable=batch_callable,
    )
    results = executor.run(jobs)

    assert batch_payloads == [
        ["page-1", "page-0-table-0-order-1"],
        ["page-0-table-1", "page-0-table-0-order-0"],
    ]
    assert [result.job.order_key for result in results] == [
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (1, 0, 0),
    ]
    assert [result.prediction.value for result in results] == [
        "page-0-table-0-order-0",
        "page-0-table-0-order-1",
        "page-0-table-1",
        "page-1",
    ]
    assert restore_table_order(list(reversed(results))) == results


def test_shadow_semantic_compare_ignores_metadata_and_falls_back_per_job() -> None:
    """元数据不影响语义比较，只有漂移的 job 应回退到 reference。"""

    reference_calls: list[str] = []
    jobs = [_job(0, 0, 0, "stable"), _job(0, 0, 1, "drift")]

    def reference(job: TableJob) -> TablePrediction:
        reference_calls.append(job.payload)
        return TablePrediction(
            {"cells": [[job.payload]]},
            {"path": "decoder-batch-1", "runtime_ms": 9},
        )

    def batch(batch_jobs: list[TableJob]) -> list[TablePrediction]:
        assert [job.payload for job in batch_jobs] == ["stable", "drift"]
        return [
            TablePrediction({"cells": [["stable"]]}, {"runtime_ms": 1}),
            TablePrediction({"cells": [["wrong"]]}, {"runtime_ms": 1}),
        ]

    results = TableBatchExecutor(
        TableBatchPlanner(),
        reference,
        batch_callable=batch,
    ).run(jobs)

    assert reference_calls == ["stable", "drift"]
    assert [result.source for result in results] == ["batch", "shadow_fallback"]
    assert [result.prediction.value for result in results] == [
        {"cells": [["stable"]]},
        {"cells": [["drift"]]},
    ]
    assert results[0].batched is not None
    assert results[0].reference is not None
    assert results[0].batched.metadata != results[0].reference.metadata
    assert results[0].used_fallback is False
    assert results[1].used_fallback is True


def test_batch_exception_uses_reference_per_job() -> None:
    """候选 batch 异常时不丢失 job，逐个回退到原 batch=1 reference 路径。"""

    reference_calls: list[str] = []
    jobs = [_job(0, 0, 0, "first"), _job(0, 0, 1, "second")]

    def reference(job: TableJob) -> str:
        reference_calls.append(job.payload)
        return "reference-" + job.payload

    def failing_batch(batch_jobs: list[TableJob]) -> list[str]:
        raise RuntimeError("verified kernel unavailable")

    results = TableBatchExecutor(
        TableBatchPlanner(),
        reference,
        batch_callable=failing_batch,
    ).run(jobs)

    assert reference_calls == ["first", "second"]
    assert [result.source for result in results] == ["reference", "reference"]
    assert [result.prediction.value for result in results] == [
        "reference-first",
        "reference-second",
    ]
    assert all(
        result.batch_error == "RuntimeError: verified kernel unavailable"
        for result in results
    )


def test_encoder_adapter_preserves_payload_order_and_can_disable_shadow() -> None:
    """encoder adapter 只转交 payload；关闭 shadow 时不调用 reference。"""

    jobs = [_job(0, 0, 0, "one"), _job(0, 0, 1, "two")]
    kernel_payloads: list[list[str]] = []

    def kernel(payloads: list[str]) -> list[str]:
        kernel_payloads.append(payloads)
        return ["encoded-" + payload for payload in payloads]

    def unexpected_reference(job: TableJob) -> str:
        raise AssertionError("strict_shadow_fallback=False 时不应调用 reference")

    results = TableBatchExecutor(
        TableBatchPlanner(),
        unexpected_reference,
        batch_callable=EncoderBatchedKernelAdapter(kernel),
        comparator=TablePredictionComparator(),
        strict_shadow_fallback=False,
    ).run(jobs)

    assert kernel_payloads == [["one", "two"]]
    assert [result.prediction.value for result in results] == ["encoded-one", "encoded-two"]
    assert [result.source for result in results] == ["batch", "batch"]
