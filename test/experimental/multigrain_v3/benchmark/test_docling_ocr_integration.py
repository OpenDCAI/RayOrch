"""Docling OCR recognition_shadow 接入的 Ray-free 契约测试。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy
import pytest

from rayorch.experimental.multigrain_v3.benchmark.document_docling.core_stages import (
    rapidocr_outputs_semantically_equal,
    select_rapidocr_rect_output,
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
