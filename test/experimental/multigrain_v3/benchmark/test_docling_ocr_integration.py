"""Docling OCR recognition_shadow 接入的 Ray-free 契约测试。"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy
import pytest

from rayorch.experimental.multigrain_v3.benchmark.document_docling.core_stages import (
    DoclingOcrPages,
    rapidocr_outputs_semantically_equal,
    select_rapidocr_rect_output,
)
from rayorch.experimental.multigrain_v3.benchmark.document_docling.core_values import (
    OcrCropJob,
)
from rayorch.experimental.multigrain_v3.benchmark.document_docling.core_v3 import (
    DoclingCoreV3Pipeline,
)


@dataclass
class FakeRapidOCROutput:
    """仅覆盖 strict shadow 比较要求的 RapidOCROutput 语义字段。"""

    boxes: object | None
    txts: tuple[str, ...] | None
    scores: tuple[float, ...] | None
    elapse_list: tuple[float, ...] = ()


def _output(
    *,
    boxes: list[list[list[int]]] | None = None,
    txts: tuple[str, ...] | None = ("text",),
    scores: tuple[float, ...] | None = (0.9,),
    elapse_list: tuple[float, ...] = (),
) -> FakeRapidOCROutput:
    """构造带 NumPy boxes 的独立 fake 输出。"""

    return FakeRapidOCROutput(
        boxes=None if boxes is None else numpy.asarray(boxes),
        txts=txts,
        scores=scores,
        elapse_list=elapse_list,
    )


def _ocr_stage_options(pipeline: DoclingCoreV3Pipeline) -> dict:
    """从编译 DAG 取 OCR stage 的 Ray execution 配置。"""

    for stage in pipeline.compile().dag.stages:
        if stage.udf is not None and stage.udf.target.__name__ == "DoclingOcrPages":
            return dict(stage.execution.ray_options)
    raise AssertionError("DoclingOcrPages stage is missing")


def test_recognition_shadow_mode_validation_is_opt_in() -> None:
    """reference 保持默认，非法模式和非法 bucket 上限必须立即失败。"""

    pipeline = DoclingCoreV3Pipeline()
    assert _ocr_stage_options(pipeline)["max_concurrency"] == 1

    with pytest.raises(ValueError, match="ocr_batch_mode"):
        DoclingCoreV3Pipeline(ocr_batch_mode="batched")
    with pytest.raises(ValueError, match="ocr_recognition_batch_size"):
        DoclingCoreV3Pipeline(ocr_recognition_batch_size=0)
    DoclingCoreV3Pipeline(ocr_batch_mode="recognition_accelerated")


def test_recognition_shadow_requires_single_actor_concurrency() -> None:
    """shadow 要求 stage actor 单并发，并把该约束编译为 Ray option。"""

    with pytest.raises(ValueError, match="ocr_actor_concurrency must be 1"):
        DoclingCoreV3Pipeline(
            ocr_batch_mode="recognition_shadow",
            ocr_actor_concurrency=2,
        )

    with pytest.raises(ValueError, match="ocr_actor_concurrency must be 1"):
        DoclingCoreV3Pipeline(
            ocr_batch_mode="recognition_accelerated",
            ocr_actor_concurrency=2,
        )

    options = _ocr_stage_options(
        DoclingCoreV3Pipeline(ocr_batch_mode="recognition_shadow")
    )
    assert options["max_concurrency"] == 1


def test_rect_candidate_semantic_contract_uses_reference_on_drift() -> None:
    """候选只要 boxes/text/scores 任一漂移，就只回退该 rect 的 reference。"""

    reference = _output(boxes=[[[1, 2], [3, 4], [5, 6], [7, 8]]])
    same_semantics = _output(
        boxes=[[[1, 2], [3, 4], [5, 6], [7, 8]]],
        elapse_list=(0.01, 0.02),
    )
    text_drift = _output(
        boxes=[[[1, 2], [3, 4], [5, 6], [7, 8]]],
        txts=("different",),
    )
    score_drift = _output(
        boxes=[[[1, 2], [3, 4], [5, 6], [7, 8]]],
        scores=(0.91,),
    )
    box_drift = _output(boxes=[[[9, 2], [3, 4], [5, 6], [7, 8]]])

    assert rapidocr_outputs_semantically_equal(reference, same_semantics)
    selected, fallback = select_rapidocr_rect_output(
        reference,
        same_semantics,
    )
    assert selected is same_semantics
    assert fallback is False

    for candidate in (text_drift, score_drift, box_drift, None):
        selected, fallback = select_rapidocr_rect_output(reference, candidate)
        assert selected is reference
        assert fallback is True


def test_v3_ocr_batch_calls_rapidocr_recognize_txt_facade() -> None:
    """V3 只按声明宽度分组，模型执行仍调用 facade 并按 lineage scatter。"""

    class FakeTextRecOutput:
        def __init__(
            self,
            *,
            imgs,
            txts,
            scores,
            word_results,
            elapse,
            viser,
        ) -> None:
            self.imgs = imgs
            self.txts = txts
            self.scores = scores
            self.word_results = word_results
            self.elapse = elapse
            self.viser = viser

    seen_batches = []

    class FakeRecognizer:
        rec_image_shape = (3, 48, 320)

    class FakeReader:
        text_rec = FakeRecognizer()
        return_word_box = False

        @staticmethod
        def recognize_txt(images):
            seen_batches.append(images)
            return FakeTextRecOutput(
                imgs=images,
                txts=tuple(f"crop-{index}" for index in range(len(images))),
                scores=[0.9] * len(images),
                word_results=(None,) * len(images),
                elapse=0.01,
                viser=None,
            )

    stage = object.__new__(DoclingOcrPages)
    stage.ocr_recognition_batch_size = 8
    stage.last_ocr_batch_audit = {
        "jobs": 0,
        "batches": 0,
        "rect_fallbacks": 0,
        "batch_errors": 0,
    }
    images = [
        numpy.zeros((20, 40, 3), dtype=numpy.uint8),
        numpy.zeros((30, 180, 3), dtype=numpy.uint8),
        numpy.zeros((20, 200, 3), dtype=numpy.uint8),
    ]
    jobs = [
        OcrCropJob(
            page=0,
            rect=0,
            crop=index,
            local_order=index,
            image=image,
            metadata={"work_index": 0},
        )
        for index, image in enumerate(images)
    ]
    work = SimpleNamespace(recognitions={}, failed=False)

    stage._run_text_recognition_batch(
        FakeReader(),
        jobs,
        [work],
        text_rec_output=FakeTextRecOutput,
    )

    assert seen_batches == [images[:2], images[2:]]
    assert [work.recognitions[index].txts for index in range(3)] == [
        ("crop-0",),
        ("crop-1",),
        ("crop-0",),
    ]
    assert stage.last_ocr_batch_audit["batches"] == 2
    assert stage.last_ocr_batch_audit["batch_errors"] == 0
