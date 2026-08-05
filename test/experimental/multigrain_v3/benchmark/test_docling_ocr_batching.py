"""RapidOCR P1 recognizer batching helper 的 Ray-free 测试。"""

from __future__ import annotations

from dataclasses import dataclass

from rayorch.experimental.multigrain_v3.benchmark.document_docling.ocr_batching import (
    OcrCropJob,
    OcrRecognition,
    RapidOCRRecognitionBatcher,
    RecognitionBatchPlanner,
    RecognitionComparator,
    RecognizerCallableAdapter,
    RecognizerInput,
    restore_page_rect_order,
)


@dataclass(frozen=True)
class FakeTensor:
    """不依赖 NumPy 的 normalized tensor fake。"""

    token: bytes
    shape: tuple[int, ...]
    dtype: str = "float32"


def _job(
    page: int,
    rect: int,
    crop: int,
    local_order: int,
    image: bytes,
) -> OcrCropJob:
    return OcrCropJob(
        page=page,
        rect=rect,
        crop=crop,
        local_order=local_order,
        image=image,
        metadata={"source": "test"},
    )


def test_planner_only_merges_actual_normalized_shape_and_key() -> None:
    """跨 page/rect 只可合并实际 normalized tensor key 完全相同的 crop。"""

    seen_images: list[bytes] = []

    def preprocess(job: OcrCropJob) -> RecognizerInput:
        seen_images.append(job.image)
        if job.image == b"narrow":
            return RecognizerInput(
                FakeTensor(job.image, shape=(3, 48, 96)),
                key=("latin", 96),
            )
        if job.image == b"other-key":
            return RecognizerInput(
                FakeTensor(job.image, shape=(3, 48, 160)),
                key=("cjk", 160),
            )
        return RecognizerInput(
            FakeTensor(job.image, shape=(3, 48, 160)),
            key=("latin", 160),
        )

    page_0 = _job(0, 1, 0, 0, b"wide-page-0")
    page_1 = _job(1, 0, 0, 0, b"wide-page-1")
    narrow = _job(1, 1, 0, 0, b"narrow")
    same_shape_other_key = _job(2, 0, 0, 0, b"other-key")

    batches = RecognitionBatchPlanner(preprocess).plan(
        [page_0, narrow, page_1, same_shape_other_key]
    )

    assert seen_images == [
        b"wide-page-0",
        b"narrow",
        b"wide-page-1",
        b"other-key",
    ]
    assert [batch.key.shape for batch in batches] == [
        (3, 48, 160),
        (3, 48, 96),
        (3, 48, 160),
    ]
    assert [batch.key.normalization_key for batch in batches] == [
        ("latin", 160),
        ("latin", 96),
        ("cjk", 160),
    ]
    assert [[item.job.image for item in batch.jobs] for batch in batches] == [
        [b"wide-page-0", b"wide-page-1"],
        [b"narrow"],
        [b"other-key"],
    ]


def test_runner_restores_page_rect_order_and_preserves_crop_bytes() -> None:
    """bucket 执行顺序可改变，但最终结果必须恢复 page/rect 顺序且不重编码输入。"""

    received_batch_tokens: list[list[bytes]] = []
    received_single_tokens: list[bytes] = []

    def preprocess(job: OcrCropJob) -> RecognizerInput:
        assert isinstance(job.image, bytes)
        return RecognizerInput(FakeTensor(job.image, shape=(3, 48, 128)), key="fixed")

    def recognize_batch(tensors: list[FakeTensor]):
        received_batch_tokens.append([tensor.token for tensor in tensors])
        return [(tensor.token.decode(), 0.9) for tensor in tensors]

    def recognize_one(tensor: FakeTensor):
        received_single_tokens.append(tensor.token)
        return (tensor.token.decode(), 0.9)

    jobs = [
        _job(1, 0, 0, 0, b"page-1"),
        _job(0, 1, 0, 0, b"page-0-rect-1"),
        _job(0, 0, 1, 1, b"page-0-rect-0-crop-1"),
        _job(0, 0, 0, 0, b"page-0-rect-0-crop-0"),
    ]
    runner = RapidOCRRecognitionBatcher(
        RecognitionBatchPlanner(preprocess),
        RecognizerCallableAdapter(recognize_batch, single_callable=recognize_one),
    )

    results = runner.run(jobs)

    assert received_batch_tokens == [[job.image for job in jobs]]
    assert received_single_tokens == [job.image for job in jobs]
    assert [result.job.order_key for result in results] == [
        (0, 0, 0, 0),
        (0, 0, 1, 1),
        (0, 1, 0, 0),
        (1, 0, 0, 0),
    ]
    assert [result.recognition.text for result in results] == [
        "page-0-rect-0-crop-0",
        "page-0-rect-0-crop-1",
        "page-0-rect-1",
        "page-1",
    ]
    assert all(result.source == "batch" for result in results)
    assert restore_page_rect_order(list(reversed(results))) == results


def test_shadow_compare_uses_score_tolerance_and_per_job_fallback() -> None:
    """score 小漂移可通过，文本漂移只回退对应 job，而不是整批回退。"""

    fallback_calls: list[bytes] = []

    def preprocess(job: OcrCropJob) -> RecognizerInput:
        return RecognizerInput(FakeTensor(job.image, shape=(3, 48, 128)), key="fixed")

    def batch(tensors: list[FakeTensor]):
        assert [tensor.token for tensor in tensors] == [b"stable", b"drift"]
        return [("stable", 0.9004), ("incorrect", 0.7)]

    def original_single(tensor: FakeTensor):
        if tensor.token == b"stable":
            return ("stable", 0.9)
        return ("expected", 0.7)

    def fallback_single(tensor: FakeTensor):
        fallback_calls.append(tensor.token)
        return ("fallback-" + tensor.token.decode(), 0.42)

    recognizer = RecognizerCallableAdapter(batch, single_callable=original_single)
    fallback = RecognizerCallableAdapter(
        lambda tensors: [fallback_single(tensor) for tensor in tensors],
        single_callable=fallback_single,
    )
    runner = RapidOCRRecognitionBatcher(
        RecognitionBatchPlanner(preprocess),
        recognizer,
        comparator=RecognitionComparator(score_atol=0.001),
        fallback_recognizer=fallback,
    )

    results = runner.run(
        [
            _job(0, 0, 0, 0, b"stable"),
            _job(0, 0, 1, 1, b"drift"),
        ]
    )

    assert [result.source for result in results] == ["batch", "fallback"]
    assert [result.recognition for result in results] == [
        OcrRecognition("stable", 0.9004, ("stable", 0.9004)),
        OcrRecognition("fallback-drift", 0.42, ("fallback-drift", 0.42)),
    ]
    assert fallback_calls == [b"drift"]
    assert results[0].shadow == OcrRecognition("stable", 0.9, ("stable", 0.9))
    assert results[1].shadow == OcrRecognition("expected", 0.7, ("expected", 0.7))
    assert not results[0].used_fallback
    assert results[1].used_fallback


def test_adapter_accepts_rapidocr_style_results_and_batch_failure_falls_back() -> None:
    """``(results, elapsed)`` 返回可解析，batch 异常也必须逐 job 回退。"""

    def preprocess(job: OcrCropJob) -> RecognizerInput:
        return RecognizerInput(FakeTensor(job.image, shape=(3, 48, 128)), key="fixed")

    rapidocr_style = RecognizerCallableAdapter(
        lambda tensors: (
            [(tensor.token.decode(), 0.8) for tensor in tensors],
            0.01,
        ),
        single_callable=lambda tensor: (tensor.token.decode(), 0.8),
    )
    normal_jobs = [
        _job(0, 0, 0, 0, b"first"),
        _job(0, 0, 1, 1, b"second"),
    ]
    assert [
        result.recognition.text
        for result in RapidOCRRecognitionBatcher(
            RecognitionBatchPlanner(preprocess),
            rapidocr_style,
        ).run(normal_jobs)
    ] == ["first", "second"]

    failing = RecognizerCallableAdapter(
        lambda tensors: (_ for _ in ()).throw(RuntimeError("batch unavailable")),
        single_callable=lambda tensor: (tensor.token.decode(), 0.6),
    )
    results = RapidOCRRecognitionBatcher(
        RecognitionBatchPlanner(preprocess),
        failing,
    ).run(normal_jobs)

    assert [result.source for result in results] == ["fallback", "fallback"]
    assert [result.recognition.text for result in results] == ["first", "second"]
    assert all(result.batch_error == "RuntimeError: batch unavailable" for result in results)
