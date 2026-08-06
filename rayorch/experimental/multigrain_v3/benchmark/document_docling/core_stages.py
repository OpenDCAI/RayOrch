"""Docling 核心模型 stage 的 V3 value-only adapter。

本模块不调用 per-page ``DocumentConverter``，也不在 Stage 之间传递可变 ``Page``。
Expand 使用 Docling PDF backend 物化 immutable page source；Layout/OCR/Table actors 各自
构造临时 Page、调用 Docling 原模型，再只返回该 Stage 的值结果。
"""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

from .core_values import (
    DoclingOcrResult,
    OcrCropJob,
    DoclingPageAssembly,
    DoclingPostprocessedPage,
    DoclingPageSource,
    DoclingTableJob,
    PdfRenderCache,
    page_from_source,
)
from .table_batching import (
    TableBatchPlanner,
    TableJob as EncoderTableJob,
)


@dataclass
class _RapidOcrRectWork:
    """一个 OCR rect 的候选重建上下文。

    detector/classifier 保持每个 rect 单独运行；``crops`` 中的 TextRecognizer
    输入会在随后跨 rect/page gather。``reference`` 始终是原 RapidOCR reader 的完整
    输出，候选任一步出错或语义不同都会选用它。
    """

    page_index: int
    rect_index: int
    ocr_rect: Any
    original_image: Any
    op_record: dict[str, Any]
    det_res: Any
    cls_res: Any
    cropped_images: list[Any]
    reference: Any
    recognitions: dict[int, Any] = field(default_factory=dict)
    candidate: Any = None
    failed: bool = False


def _png_bytes(image: Any) -> bytes:
    """把 PIL image 压缩为可序列化 PNG bytes。"""

    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def _input_document(
    path: str,
    document_hash: str,
    page_count: int,
):
    """构造不打开 backend 的 Docling InputDocument context。"""

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.document import InputDocument
    from docling.datamodel.settings import DocumentLimits

    return InputDocument.model_construct(
        file=Path(path),
        document_hash=document_hash,
        valid=True,
        backend_options=None,
        limits=DocumentLimits(),
        format=InputFormat.PDF,
        filesize=Path(path).stat().st_size,
        page_count=page_count,
    )


def _conversion_result(
    path: str,
    document_hash: str,
    pages: list[Any] | None = None,
):
    """为 Docling model 构造最小 ConversionResult context。"""

    from docling.datamodel.document import ConversionResult

    return ConversionResult(
        input=_input_document(
            path,
            document_hash,
            len(pages or ()),
        ),
        pages=list(pages or ()),
    )


def _group_indices_by_document(
    sources: list[DoclingPageSource],
) -> list[tuple[str, str, list[int]]]:
    """按 document 分组输入下标，同时保持原 batch 的首次出现顺序。"""

    groups: dict[tuple[str, str], list[int]] = {}
    order: list[tuple[str, str]] = []
    for index, source in enumerate(sources):
        key = (source.document_path, source.document_hash)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(index)
    return [
        (path, document_hash, groups[(path, document_hash)])
        for path, document_hash in order
    ]


def rapidocr_outputs_semantically_equal(left: Any, right: Any) -> bool:
    """严格比较 RapidOCR 最终输出的 boxes/text/scores 语义字段。

    ``RapidOCROutput`` 的耗时、可视化器和原始图像均不是 Docling OCR cell 的业务
    语义，故不参与比较。缺失字段也要两边同时缺失才视为相同。
    """

    try:
        import numpy

        left_boxes = getattr(left, "boxes", None)
        right_boxes = getattr(right, "boxes", None)
        if (left_boxes is None) != (right_boxes is None):
            return False
        if left_boxes is not None and not numpy.array_equal(
            left_boxes,
            right_boxes,
        ):
            return False
        for field_name in ("txts", "scores"):
            left_value = getattr(left, field_name, None)
            right_value = getattr(right, field_name, None)
            if (left_value is None) != (right_value is None):
                return False
            if left_value is not None and tuple(left_value) != tuple(right_value):
                return False
    except Exception:
        return False
    return True


def select_rapidocr_rect_output(reference: Any, candidate: Any) -> tuple[Any, bool]:
    """选择 rect 的严格输出，并返回是否必须回退到原 reader。

    此小 helper 不依赖 Docling/RapidOCR，供 Ray-free 测试覆盖“候选漂移只影响该
    rect，最终仍使用 reference”的契约。
    """

    if candidate is not None and rapidocr_outputs_semantically_equal(
        reference,
        candidate,
    ):
        return candidate, False
    return reference, True


class ExpandDoclingPages:
    """使用 Docling PDF backend 一次打开一个文档并产生 page source values。"""

    def __init__(
        self,
        image_scales: tuple[float, ...] = (1.0, 2.0, 3.0),
    ) -> None:
        """初始化 Docling preprocess，并保存需物化的 image scales。"""

        from docling.models.stages.page_preprocessing.page_preprocessing_model import (
            PagePreprocessingModel,
            PagePreprocessingOptions,
        )

        self.preprocess = PagePreprocessingModel(
            PagePreprocessingOptions(images_scale=1.0)
        )
        self.image_scales = image_scales

    def run(self, paths: list[str]) -> list[list[DoclingPageSource]]:
        """物化 segmented cells 和模型所需 1×/2×/3× page images。"""

        from docling.backend.docling_parse_backend import (
            DoclingParseDocumentBackend,
        )
        from docling.datamodel.base_models import InputFormat, Page
        from docling.datamodel.document import ConversionResult, InputDocument

        outputs = []
        for value in paths:
            path = str(Path(value).resolve())
            in_doc = InputDocument(
                path_or_stream=Path(path),
                format=InputFormat.PDF,
                backend=DoclingParseDocumentBackend,
            )
            conv_res = ConversionResult(input=in_doc)
            sources = []
            try:
                for page_index in range(in_doc.page_count):
                    backend = in_doc._backend.load_page(page_index)
                    page = Page(
                        page_no=page_index + 1,
                        size=backend.get_size(),
                    )
                    page._backend = backend
                    page = list(self.preprocess(conv_res, [page]))[0]
                    source = DoclingPageSource(
                        document_path=path,
                        document_hash=in_doc.document_hash,
                        page_no=page.page_no,
                        size=page.size,
                        segmented_page=copy.deepcopy(page.parsed_page),
                        images_by_scale={
                            scale: _png_bytes(
                                backend.get_page_image(scale=scale)
                            )
                            for scale in self.image_scales
                        },
                    )
                    backend.unload()
                    sources.append(source)
            finally:
                in_doc._backend.unload()
            outputs.append(sources)
        return outputs


class DoclingLayoutPages:
    """持有 Docling LayoutModel，并调用其真实跨 page batch API。"""

    def __init__(
        self,
        *,
        device: str = "cpu",
        num_threads: int = 4,
    ) -> None:
        """按 Docling 默认 LayoutOptions 初始化模型一次。"""

        from docling.datamodel.accelerator_options import AcceleratorOptions
        from docling.datamodel.pipeline_options import LayoutOptions
        from docling.models.stages.layout.layout_model import LayoutModel

        self.model = LayoutModel(
            artifacts_path=None,
            accelerator_options=AcceleratorOptions(
                device=device,
                num_threads=num_threads,
            ),
            options=LayoutOptions(),
        )
        self.render_cache = PdfRenderCache()

    def run(self, sources: list[DoclingPageSource]) -> list[Any]:
        """返回与输入 page 一一对应的 Docling LayoutPrediction。"""

        from docling_core.types.doc import BoundingBox, DocItemLabel
        from docling.datamodel.base_models import Cluster, LayoutPrediction

        pages = [
            page_from_source(
                source,
                render_cache=self.render_cache,
            )
            for source in sources
        ]
        predictions = self.model.layout_predictor.predict_batch(
            [page.get_image(scale=1.0) for page in pages]
        )
        return [
            LayoutPrediction(
                clusters=[
                    Cluster(
                        id=index,
                        label=DocItemLabel(
                            prediction["label"]
                            .lower()
                            .replace(" ", "_")
                            .replace("-", "_")
                        ),
                        confidence=prediction["confidence"],
                        bbox=BoundingBox.model_validate(prediction),
                        cells=[],
                    )
                    for index, prediction in enumerate(page_predictions)
                ]
            )
            for page_predictions in predictions
        ]


class DoclingOcrPages:
    """直接适配 Docling RapidOcrModel，并只返回 OCR 后 segmented page。"""

    def __init__(
        self,
        *,
        device: str = "cpu",
        num_threads: int = 4,
        ocr_batch_mode: str = "reference",
        ocr_recognition_batch_size: int = 6,
    ) -> None:
        """初始化与原生 auto 路径一致的 RapidOCR/ONNX Runtime backend。

        ``ocr_batch_mode="reference"`` 完全保留原始 ``RapidOcrModel`` 路径。
        ``"recognition_shadow"`` 是保守实验模式：每个 OCR rect 均先执行原 reader
        作为 shadow reference；候选路径只跨 rect/page 批处理 ``TextRecognizer`` 的
        crop，detector/classifier 仍逐 rect 调用。候选 ``RapidOCROutput`` 的
        boxes/text/scores 与 reference 有任何差异、或候选 batch 失败时，都会按 rect
        回退 reference。该模式要求 actor ``max_concurrency=1``，由 ``core_v3`` 在
        编排时校验，避免 RapidOCR 可变 reader 配置和 actor 内审计状态发生重入。
        """

        if ocr_batch_mode not in {
            "reference",
            "recognition_shadow",
            "recognition_accelerated",
        }:
            raise ValueError(
                "ocr_batch_mode must be reference, recognition_shadow, "
                "or recognition_accelerated"
            )
        if ocr_recognition_batch_size <= 0:
            raise ValueError("ocr_recognition_batch_size must be positive")

        from docling.datamodel.accelerator_options import AcceleratorOptions
        from docling.datamodel.pipeline_options import RapidOcrOptions
        from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel

        self.model = RapidOcrModel(
            enabled=True,
            artifacts_path=None,
            options=RapidOcrOptions(
                backend="onnxruntime",
                rapidocr_params={
                    "Rec.rec_batch_num": ocr_recognition_batch_size,
                },
            ),
            accelerator_options=AcceleratorOptions(
                device=device,
                num_threads=num_threads,
            ),
        )
        self.render_cache = PdfRenderCache()
        self.ocr_batch_mode = ocr_batch_mode
        self.ocr_recognition_batch_size = ocr_recognition_batch_size
        self.last_ocr_batch_audit: dict[str, int | str] = {
            "jobs": 0,
            "batches": 0,
            "rect_fallbacks": 0,
            "batch_errors": 0,
        }
        self.total_ocr_batch_audit: dict[str, int | str] = {
            "jobs": 0,
            "batches": 0,
            "rect_fallbacks": 0,
            "batch_errors": 0,
        }

    def run(
        self,
        sources: list[DoclingPageSource],
        layouts: list[Any],
    ) -> list[DoclingOcrResult]:
        """按 document context 调用 OCR，同时保持原输入 row 顺序。"""

        self.last_ocr_batch_audit = {
            "jobs": 0,
            "batches": 0,
            "rect_fallbacks": 0,
            "batch_errors": 0,
        }
        outputs: list[DoclingOcrResult | None] = [None] * len(sources)
        if self.ocr_batch_mode == "reference":
            for path, document_hash, indices in _group_indices_by_document(
                sources
            ):
                pages = self._pages_for_indices(indices, sources, layouts)
                processed = list(
                    self.model(
                        _conversion_result(path, document_hash, pages),
                        pages,
                    )
                )
                for index, page in zip(indices, processed):
                    outputs[index] = DoclingOcrResult(
                        segmented_page=copy.deepcopy(page.parsed_page)
                    )
        else:
            processed = self._recognition_batch_pages(
                sources,
                layouts,
                strict_shadow=self.ocr_batch_mode == "recognition_shadow",
            )
            for index, page in enumerate(processed):
                outputs[index] = DoclingOcrResult(
                    segmented_page=copy.deepcopy(page.parsed_page)
                )
        assert all(output is not None for output in outputs)
        self._accumulate_ocr_audit()
        return [output for output in outputs if output is not None]

    def batch_audit(self) -> dict[str, int | str]:
        """返回小型 OCR shadow 审计，不携带 crop/page 业务数据。"""

        return dict(self.total_ocr_batch_audit)

    def _accumulate_ocr_audit(self) -> None:
        """把当前物理 UDF batch 的审计累加到 actor 生命周期摘要。"""

        for key in ("jobs", "batches", "rect_fallbacks", "batch_errors"):
            self.total_ocr_batch_audit[key] = int(
                self.total_ocr_batch_audit[key]
            ) + int(self.last_ocr_batch_audit[key])

    def _pages_for_indices(
        self,
        indices: list[int],
        sources: list[DoclingPageSource],
        layouts: list[Any],
    ) -> list[Any]:
        """从 immutable input 重建独占 Page，避免 candidate/reference 互相污染。"""

        return [
            page_from_source(
                sources[index],
                layout=layouts[index],
                render_cache=self.render_cache,
            )
            for index in indices
        ]

    def _reference_pages_across_documents(
        self,
        sources: list[DoclingPageSource],
        layouts: list[Any],
    ) -> list[Any]:
        """逐 document 运行原 OCR，并恢复到物理 batch 的输入顺序。"""

        outputs: list[Any | None] = [None] * len(sources)
        for path, document_hash, indices in _group_indices_by_document(sources):
            pages = self._pages_for_indices(indices, sources, layouts)
            pages = list(
                self.model(
                    _conversion_result(path, document_hash, pages),
                    pages,
                )
            )
            for index, page in zip(indices, pages):
                outputs[index] = page
        assert all(page is not None for page in outputs)
        return [page for page in outputs if page is not None]

    def _recognition_batch_pages(
        self,
        sources: list[DoclingPageSource],
        layouts: list[Any],
        *,
        strict_shadow: bool,
    ) -> list[Any]:
        """跨当前物理 V3 batch 聚合 OCR crops，并按输入顺序恢复 pages。

        shadow 模式仍执行完整 reference 和 page gate；accelerated 模式只在候选
        batch/rect 失败时延迟调用原 reader。两种模式都不再按 document 拆散当前物理
        batch，因此 elastic 可以真正形成跨 document recognizer batch。
        """

        reference_pages = (
            self._reference_pages_across_documents(sources, layouts)
            if strict_shadow
            else None
        )
        rapid_model = self.model
        if not self._is_rapidocr_engine(rapid_model):
            self._record_ocr_batch_error(
                "RapidOcrModel does not expose the required facade; "
                "recognition batching kept the reference path"
            )
            return reference_pages or self._reference_pages_across_documents(
                sources,
                layouts,
            )

        candidate_pages = self._pages_for_indices(
            list(range(len(sources))),
            sources,
            layouts,
        )
        conv_res_by_page: list[Any | None] = [None] * len(sources)
        for path, document_hash, indices in _group_indices_by_document(sources):
            conv_res = _conversion_result(
                path,
                document_hash,
                [candidate_pages[index] for index in indices],
            )
            for index in indices:
                conv_res_by_page[index] = conv_res
        assert all(value is not None for value in conv_res_by_page)
        try:
            candidate_pages = self._run_recognition_candidate(
                rapid_model,
                [value for value in conv_res_by_page if value is not None],
                candidate_pages,
                strict_shadow=strict_shadow,
            )
        except Exception as error:
            self._record_ocr_batch_error(f"{type(error).__name__}: {error}")
            return reference_pages or self._reference_pages_across_documents(
                sources,
                layouts,
            )

        if reference_pages is None:
            return candidate_pages
        for position, (candidate, reference) in enumerate(
            zip(candidate_pages, reference_pages)
        ):
            if self._same_page_semantics(candidate, reference):
                continue
            self.last_ocr_batch_audit["rect_fallbacks"] = int(
                self.last_ocr_batch_audit["rect_fallbacks"]
            ) + self._ocr_rect_count(rapid_model, candidate)
            candidate_pages[position] = reference
        return candidate_pages

    @staticmethod
    def _is_rapidocr_engine(engine: Any) -> bool:
        """只接受拥有原 RapidOCR facade 的 Docling OCR engine。"""

        reader = getattr(engine, "reader", None)
        return (
            callable(getattr(engine, "get_ocr_rects", None))
            and callable(getattr(engine, "post_process_cells", None))
            and callable(getattr(reader, "__call__", None))
            and callable(getattr(reader, "preprocess_img", None))
            and callable(getattr(reader, "detect_and_crop", None))
            and callable(getattr(reader, "cls_and_rotate", None))
            and callable(getattr(reader, "recognize_txt", None))
            and callable(getattr(reader, "build_final_output", None))
        )

    def _run_recognition_candidate(
        self,
        rapid_model: Any,
        conv_res_by_page: list[Any],
        pages: list[Any],
        *,
        strict_shadow: bool,
    ) -> list[Any]:
        """保留 RapidOcrModel 的 rect 语义，仅替换 recognizer 调度。

        shadow 模式对每个 rect 调用原 ``RapidOCR.__call__``；accelerated 模式只在
        candidate 失败时延迟调用原 reader。候选路径严格沿用 RapidOCR 的
        ``load_img/preprocess_img/detect_and_crop/cls_and_rotate``，
        仅将 classifier 后的 crop 交给兼容 bucket 的 ``TextRecognizer``。随后仍调用
        原 ``build_final_output``，并按 RapidOcrModel 原样构造 ``TextCell`` 和调用
        ``post_process_cells``。
        """

        import numpy
        from docling.datamodel.settings import settings
        from rapidocr.ch_ppocr_cls import TextClsOutput
        from rapidocr.ch_ppocr_det import TextDetOutput
        from rapidocr.ch_ppocr_rec import TextRecOutput

        reader = rapid_model.reader
        works: list[_RapidOcrRectWork] = []
        jobs: list[OcrCropJob] = []
        page_rects: dict[int, list[Any]] = {}

        for page_index, page in enumerate(pages):
            assert page._backend is not None
            if not page._backend.is_valid():
                continue
            ocr_rects = rapid_model.get_ocr_rects(page)
            page_rects[page_index] = ocr_rects
            for rect_index, ocr_rect in enumerate(ocr_rects):
                # 与 RapidOcrModel.__call__ 一致：零面积 rect 完全跳过。
                if ocr_rect.area() == 0:
                    continue
                high_res_image = page._backend.get_page_image(
                    scale=rapid_model.scale,
                    cropbox=ocr_rect,
                )
                image = numpy.array(high_res_image)
                del high_res_image

                reference = None
                if strict_shadow:
                    reference = reader(
                        image,
                        use_det=rapid_model.options.use_det,
                        use_cls=rapid_model.options.use_cls,
                        use_rec=rapid_model.options.use_rec,
                    )
                work = self._prepare_rapidocr_rect_candidate(
                    reader,
                    rapid_model,
                    image,
                    page_index=page_index,
                    rect_index=rect_index,
                    ocr_rect=ocr_rect,
                    text_det_output=TextDetOutput,
                    text_cls_output=TextClsOutput,
                    text_rec_output=TextRecOutput,
                )
                work.reference = reference
                work_index = len(works)
                works.append(work)
                if work.failed:
                    continue
                for crop_index, crop in enumerate(work.cropped_images):
                    jobs.append(
                        OcrCropJob(
                            page=page_index,
                            rect=rect_index,
                            crop=crop_index,
                            local_order=crop_index,
                            image=crop,
                            metadata={"work_index": work_index},
                        )
                    )

        self.last_ocr_batch_audit["jobs"] = int(
            self.last_ocr_batch_audit["jobs"]
        ) + len(jobs)
        self._run_text_recognition_batch(
            reader,
            jobs,
            works,
            text_rec_output=TextRecOutput,
        )

        selected_by_page: dict[int, list[tuple[Any, Any]]] = {}
        for work in works:
            candidate = work.candidate
            if not work.failed and work.cropped_images:
                try:
                    if len(work.recognitions) != len(work.cropped_images):
                        raise RuntimeError(
                            "recognizer did not return every crop for OCR rect"
                        )
                    candidate = reader.build_final_output(
                        work.original_image,
                        work.det_res,
                        work.cls_res,
                        self._single_rect_recognition_output(
                            work,
                            text_rec_output=TextRecOutput,
                        ),
                        work.cropped_images,
                        work.op_record,
                    )
                except Exception:
                    candidate = None

            if strict_shadow:
                selected, fallback = select_rapidocr_rect_output(
                    work.reference,
                    candidate,
                )
            elif candidate is not None:
                selected, fallback = candidate, False
            else:
                selected = reader(
                    work.original_image,
                    use_det=rapid_model.options.use_det,
                    use_cls=rapid_model.options.use_cls,
                    use_rec=rapid_model.options.use_rec,
                )
                fallback = True
            if fallback:
                self.last_ocr_batch_audit["rect_fallbacks"] = int(
                    self.last_ocr_batch_audit["rect_fallbacks"]
                ) + 1
            selected_by_page.setdefault(work.page_index, []).append(
                (work.ocr_rect, selected)
            )

        for page_index, page in enumerate(pages):
            if page_index not in page_rects:
                continue
            all_ocr_cells = []
            for ocr_rect, result in selected_by_page.get(page_index, []):
                # 下列 TextCell 构造与 Docling RapidOcrModel.__call__ 保持逐字段一致。
                if result is None or result.boxes is None:
                    continue
                result = list(zip(result.boxes.tolist(), result.txts, result.scores))
                from docling_core.types.doc import BoundingBox, CoordOrigin
                from docling_core.types.doc.page import (
                    BoundingRectangle,
                    TextCell,
                )

                cells = [
                    TextCell(
                        index=ix,
                        text=line[1],
                        orig=line[1],
                        confidence=line[2],
                        from_ocr=True,
                        rect=BoundingRectangle.from_bounding_box(
                            BoundingBox.from_tuple(
                                coord=(
                                    (line[0][0][0] / rapid_model.scale)
                                    + ocr_rect.l,
                                    (line[0][0][1] / rapid_model.scale)
                                    + ocr_rect.t,
                                    (line[0][2][0] / rapid_model.scale)
                                    + ocr_rect.l,
                                    (line[0][2][1] / rapid_model.scale)
                                    + ocr_rect.t,
                                ),
                                origin=CoordOrigin.TOPLEFT,
                            )
                        ),
                    )
                    for ix, line in enumerate(result)
                ]
                all_ocr_cells.extend(cells)

            conv_res = conv_res_by_page[page_index]
            rapid_model.post_process_cells(all_ocr_cells, page, conv_res)
            if settings.debug.visualize_ocr:
                rapid_model.draw_ocr_rects_and_cells(
                    conv_res,
                    page,
                    page_rects[page_index],
                )
        return pages

    @staticmethod
    def _prepare_rapidocr_rect_candidate(
        reader: Any,
        rapid_model: Any,
        image: Any,
        *,
        page_index: int,
        rect_index: int,
        ocr_rect: Any,
        text_det_output: Any,
        text_cls_output: Any,
        text_rec_output: Any,
    ) -> _RapidOcrRectWork:
        """逐 rect 复刻 RapidOCR detector/classifier，暂不调用 recognizer。"""

        from rapidocr.main import RapidOCRError

        reader.update_params(
            use_det=rapid_model.options.use_det,
            use_cls=rapid_model.options.use_cls,
            use_rec=rapid_model.options.use_rec,
        )
        original_image = reader.load_img(image)
        prepared_image, op_record = reader.preprocess_img(original_image)
        empty_det = text_det_output()
        empty_cls = text_cls_output()
        empty_rec = text_rec_output()
        try:
            if reader.use_det:
                cropped_images, det_res = reader.detect_and_crop(
                    prepared_image,
                    op_record,
                )
            else:
                cropped_images, det_res = [prepared_image], empty_det
        except RapidOCRError:
            return _RapidOcrRectWork(
                page_index=page_index,
                rect_index=rect_index,
                ocr_rect=ocr_rect,
                original_image=original_image,
                op_record=op_record,
                det_res=empty_det,
                cls_res=empty_cls,
                cropped_images=[],
                reference=None,
                candidate=reader.build_final_output(
                    original_image,
                    empty_det,
                    empty_cls,
                    empty_rec,
                    [],
                    op_record,
                ),
            )
        try:
            if reader.use_cls:
                classified_images, cls_res = reader.cls_and_rotate(cropped_images)
            else:
                classified_images, cls_res = cropped_images, empty_cls
        except RapidOCRError:
            return _RapidOcrRectWork(
                page_index=page_index,
                rect_index=rect_index,
                ocr_rect=ocr_rect,
                original_image=original_image,
                op_record=op_record,
                det_res=det_res,
                cls_res=empty_cls,
                cropped_images=[],
                reference=None,
                candidate=reader.build_final_output(
                    original_image,
                    det_res,
                    empty_cls,
                    empty_rec,
                    [],
                    op_record,
                ),
            )
        if not reader.use_rec:
            return _RapidOcrRectWork(
                page_index=page_index,
                rect_index=rect_index,
                ocr_rect=ocr_rect,
                original_image=original_image,
                op_record=op_record,
                det_res=det_res,
                cls_res=cls_res,
                cropped_images=[],
                reference=None,
                candidate=reader.build_final_output(
                    original_image,
                    det_res,
                    cls_res,
                    empty_rec,
                    cropped_images,
                    op_record,
                ),
            )
        return _RapidOcrRectWork(
            page_index=page_index,
            rect_index=rect_index,
            ocr_rect=ocr_rect,
            original_image=original_image,
            op_record=op_record,
            det_res=det_res,
            cls_res=cls_res,
            cropped_images=list(classified_images),
            reference=None,
        )

    def _run_text_recognition_batch(
        self,
        reader: Any,
        jobs: list[OcrCropJob],
        works: list[_RapidOcrRectWork],
        *,
        text_rec_output: Any,
    ) -> None:
        """按归一化宽度 gather 后调用 RapidOCR facade，再按 lineage scatter。

        adapter 只读取 recognizer 声明的 ``rec_image_shape`` 计算最终 tensor width；
        resize/normalize、排序、session、CTC decode、RTL 恢复和结果对象构造仍全部由
        ``recognize_txt(list[crop])`` 拥有。这个兼容 key 隔离极端长宽比 crop，避免
        RapidOCR 把同一内部 minibatch 的其他 crop pad 到该长尾宽度。
        """

        if not jobs:
            return
        try:
            _, model_height, model_width = reader.text_rec.rec_image_shape[:3]
            base_ratio = model_width / model_height
            buckets: dict[Any, list[OcrCropJob]] = {}
            for job in jobs:
                image_height, image_width = job.image.shape[:2]
                ratio = max(base_ratio, image_width / float(image_height))
                normalized_width = int(model_height * ratio)
                key = (
                    normalized_width,
                    ratio if reader.return_word_box else None,
                )
                buckets.setdefault(key, []).append(job)
        except Exception as error:
            self._record_ocr_batch_error(f"{type(error).__name__}: {error}")
            for job in jobs:
                works[int(job.metadata["work_index"])].failed = True
            return

        for bucket_jobs in buckets.values():
            self.last_ocr_batch_audit["batches"] = int(
                self.last_ocr_batch_audit["batches"]
            ) + (
                len(bucket_jobs) + self.ocr_recognition_batch_size - 1
            ) // self.ocr_recognition_batch_size
            try:
                rec_res = reader.recognize_txt(
                    [job.image for job in bucket_jobs]
                )
                if rec_res.txts is None:
                    raise RuntimeError("RapidOCR recognizer returned no text")
                if len(rec_res.txts) != len(bucket_jobs):
                    raise RuntimeError(
                        "TextRecognizer result count does not match gathered jobs"
                    )
                for job_index, job in enumerate(bucket_jobs):
                    work_index = int(job.metadata["work_index"])
                    works[work_index].recognitions[job.crop] = (
                        self._recognition_item(
                            rec_res,
                            job_index,
                            text_rec_output=text_rec_output,
                        )
                    )
            except Exception as error:
                self._record_ocr_batch_error(
                    f"{type(error).__name__}: {error}"
                )
                for job in bucket_jobs:
                    works[int(job.metadata["work_index"])].failed = True

    @staticmethod
    def _recognition_item(
        rec_res: Any,
        item_index: int,
        *,
        text_rec_output: Any,
    ) -> Any:
        """从 compatible-group TextRecOutput 提取一个 crop 的原字段结果。"""

        word_results = rec_res.word_results
        return text_rec_output(
            imgs=[rec_res.imgs[item_index]]
            if rec_res.imgs is not None
            else None,
            txts=(rec_res.txts[item_index],),
            scores=[rec_res.scores[item_index]],
            word_results=(word_results[item_index],),
            elapse=rec_res.elapse,
            viser=rec_res.viser,
        )

    @staticmethod
    def _single_rect_recognition_output(
        work: _RapidOcrRectWork,
        *,
        text_rec_output: Any,
    ) -> Any:
        """按 detector crop 原顺序把 bucket 结果重新组合为 rect 的 TextRecOutput。"""

        ordered = [
            work.recognitions[index]
            for index in range(len(work.cropped_images))
        ]
        return text_rec_output(
            imgs=[
                item.imgs[0]
                for item in ordered
                if item.imgs is not None
            ],
            txts=tuple(item.txts[0] for item in ordered),
            scores=[item.scores[0] for item in ordered],
            word_results=tuple(item.word_results[0] for item in ordered),
            elapse=sum(
                item.elapse
                for item in ordered
                if isinstance(item.elapse, float)
            ),
            viser=ordered[0].viser if ordered else None,
        )

    def _record_ocr_batch_error(self, summary: str) -> None:
        """累积审计错误，不用异常覆盖严格 reference 输出。"""

        self.last_ocr_batch_audit["batch_errors"] = int(
            self.last_ocr_batch_audit["batch_errors"]
        ) + 1
        self.last_ocr_batch_audit["last_error"] = summary

    @staticmethod
    def _same_page_semantics(left: Any, right: Any) -> bool:
        """比较待输出的 parsed page，而非 Page identity 或 backend 状态。"""

        left_page = left.parsed_page
        right_page = right.parsed_page
        if hasattr(left_page, "model_dump") and hasattr(right_page, "model_dump"):
            try:
                return left_page.model_dump(mode="json") == right_page.model_dump(
                    mode="json"
                )
            except Exception:
                return False
        return left_page == right_page

    @staticmethod
    def _ocr_rect_count(rapid_model: Any, page: Any) -> int:
        """为页级最终回退提供可解释的 rect 审计计数。"""

        try:
            return sum(
                rect.area() != 0
                for rect in rapid_model.get_ocr_rects(page)
            )
        except Exception:
            return 1


class DoclingPostprocessTablesAssemble:
    """复用 layout postprocess、TableFormer 和 PageAssemble 模型。"""

    def __init__(
        self,
        *,
        device: str = "cpu",
        num_threads: int = 4,
        do_table_structure: bool = True,
        table_batch_mode: str = "reference",
        table_batch_max_jobs: int = 16,
    ) -> None:
        """按原生默认 options 初始化核心 stages。

        ``table_batch_mode="reference"`` 完全调用原始 Docling
        TableStructureModel。``"encoder_shadow"`` 将多个 table crop 的 encoder 合批，
        再逐 table 运行原 decoder，并与独立 reference 页作语义比较；任何 table 漂移
        均回退 reference。``"encoder_accelerated"`` 使用同一 encoder 合批路径，但正常
        情况下直接保留候选输出，仅在执行异常时完整回退 reference。当前 TableFormer
        decoder 仍是 batch=1，因此这里只加速 encoder。
        """

        if table_batch_mode not in {
            "reference",
            "encoder_shadow",
            "encoder_accelerated",
        }:
            raise ValueError(
                "table_batch_mode must be reference, encoder_shadow, "
                "or encoder_accelerated"
            )
        if table_batch_max_jobs <= 0:
            raise ValueError("table_batch_max_jobs must be positive")

        from docling.datamodel.accelerator_options import AcceleratorOptions
        from docling.datamodel.pipeline_options import (
            LayoutOptions,
            LayoutPostprocessorOptions,
            TableStructureOptions,
        )
        from docling.models.stages.layout.layout_postprocessing_model import (
            LayoutPostprocessingModel,
        )
        from docling.models.stages.page_assemble.page_assemble_model import (
            PageAssembleModel,
            PageAssembleOptions,
        )
        from docling.models.stages.table_structure.table_structure_model import (
            TableStructureModel,
        )

        layout_options = LayoutOptions()
        self.postprocess = LayoutPostprocessingModel(
            options=LayoutPostprocessorOptions(
                skip_cell_assignment=layout_options.skip_cell_assignment,
                keep_empty_clusters=layout_options.keep_empty_clusters,
                create_orphan_clusters=layout_options.create_orphan_clusters,
                run_postprocessor=True,
            )
        )
        self.table = TableStructureModel(
            enabled=do_table_structure,
            artifacts_path=None,
            options=TableStructureOptions(),
            accelerator_options=AcceleratorOptions(
                device=device,
                num_threads=num_threads,
            ),
        )
        self.assemble = PageAssembleModel(PageAssembleOptions())
        self.render_cache = PdfRenderCache()
        self.table_batch_mode = table_batch_mode
        self.table_batch_max_jobs = table_batch_max_jobs
        self.last_table_batch_audit: dict[str, int] = {
            "jobs": 0,
            "batches": 0,
            "table_fallbacks": 0,
            "batch_errors": 0,
        }
        self.total_table_batch_audit: dict[str, int] = {
            "jobs": 0,
            "batches": 0,
            "table_fallbacks": 0,
            "batch_errors": 0,
        }

    def run(
        self,
        sources: list[DoclingPageSource],
        layouts: list[Any],
        ocr_results: list[DoclingOcrResult],
    ) -> list[DoclingPageAssembly]:
        """执行 page 后处理并返回轻量 assembly values。"""

        self.last_table_batch_audit = {
            "jobs": 0,
            "batches": 0,
            "table_fallbacks": 0,
            "batch_errors": 0,
        }
        if self.table_batch_mode == "encoder_shadow":
            result = self._run_encoder_shadow_across_documents(
                sources,
                layouts,
                ocr_results,
            )
            self._accumulate_table_audit()
            return result
        if self.table_batch_mode == "encoder_accelerated":
            try:
                result = self._run_encoder_accelerated_across_documents(
                    sources,
                    layouts,
                    ocr_results,
                )
            except Exception as error:
                self.last_table_batch_audit["batch_errors"] += 1
                self.last_table_batch_audit["last_error"] = (
                    f"{type(error).__name__}: {error}"
                )
                result = self._run_reference_across_documents(
                    sources,
                    layouts,
                    ocr_results,
                )
            self._accumulate_table_audit()
            return result

        result = self._run_reference_across_documents(
            sources,
            layouts,
            ocr_results,
        )
        self._accumulate_table_audit()
        return result

    def _run_reference_across_documents(
        self,
        sources: list[DoclingPageSource],
        layouts: list[Any],
        ocr_results: list[DoclingOcrResult],
    ) -> list[DoclingPageAssembly]:
        """按 document context 执行未经修改的 TableStructureModel。"""

        pages_by_index: dict[int, Any] = {}
        for path, document_hash, indices in _group_indices_by_document(
            sources
        ):
            pages = self._pages_for_indices(
                indices, sources, layouts, ocr_results
            )
            conv_res = _conversion_result(path, document_hash, pages)
            pages = list(self.postprocess(conv_res, pages))
            pages = list(self.table(conv_res, pages))
            for index, page in zip(indices, pages):
                pages_by_index[index] = page
        return self._assemble_pages_across_documents(
            sources,
            pages_by_index,
        )

    def batch_audit(self) -> dict[str, int | str]:
        """返回小型 Table shadow 审计，不携带 table/job 业务数据。"""

        return dict(self.total_table_batch_audit)

    def _accumulate_table_audit(self) -> None:
        """把当前物理 UDF batch 的 table 审计累加到 actor 生命周期摘要。"""

        for key in ("jobs", "batches", "table_fallbacks", "batch_errors"):
            self.total_table_batch_audit[key] += self.last_table_batch_audit[
                key
            ]

    def _run_encoder_shadow_across_documents(
        self,
        sources: list[DoclingPageSource],
        layouts: list[Any],
        ocr_results: list[DoclingOcrResult],
    ) -> list[DoclingPageAssembly]:
        """跨 document 聚合 TableFormer encoder job，再按原 document 恢复 assembly。

        postprocess/reference table 仍使用每个 document 的最小 context，确保与原 adapter
        一致；只有 candidate 的 encoder precompute 横跨当前 V3 Map physical batch 的所有
        page/table jobs。这是 P0 所需的真正 cross-parent batching 边界。
        """

        if not sources:
            return []
        reference_by_index: dict[int, Any] = {}
        candidate_by_index: dict[int, Any] = {}
        groups = _group_indices_by_document(sources)
        for path, document_hash, indices in groups:
            reference_pages = self._pages_for_indices(
                indices,
                sources,
                layouts,
                ocr_results,
            )
            reference_ctx = _conversion_result(
                path,
                document_hash,
                reference_pages,
            )
            reference_pages = list(
                self.postprocess(reference_ctx, reference_pages)
            )
            reference_pages = list(self.table(reference_ctx, reference_pages))
            for index, page in zip(indices, reference_pages):
                reference_by_index[index] = page

            candidate_pages = self._pages_for_indices(
                indices,
                sources,
                layouts,
                ocr_results,
            )
            candidate_ctx = _conversion_result(
                path,
                document_hash,
                candidate_pages,
            )
            candidate_pages = list(
                self.postprocess(candidate_ctx, candidate_pages)
            )
            for index, page in zip(indices, candidate_pages):
                candidate_by_index[index] = page

        candidate_pages = [
            candidate_by_index[index] for index in range(len(sources))
        ]
        reference_pages = [
            reference_by_index[index] for index in range(len(sources))
        ]
        try:
            candidate_pages = self._run_encoder_batched_table(
                _conversion_result(
                    sources[0].document_path,
                    sources[0].document_hash,
                    candidate_pages,
                ),
                candidate_pages,
            )
        except Exception as error:
            self.last_table_batch_audit["batch_errors"] += 1
            self.last_table_batch_audit["last_error"] = (
                f"{type(error).__name__}: {error}"
            )
            candidate_pages = reference_pages
        else:
            self._restore_reference_table_semantics(
                candidate_pages,
                reference_pages,
            )
        candidate_by_index = {
            index: page for index, page in enumerate(candidate_pages)
        }

        outputs: list[DoclingPageAssembly | None] = [None] * len(sources)
        for path, document_hash, indices in groups:
            pages = [candidate_by_index[index] for index in indices]
            assembled = list(
                self.assemble(
                    _conversion_result(path, document_hash, pages),
                    pages,
                )
            )
            for index, page in zip(indices, assembled):
                outputs[index] = DoclingPageAssembly(
                    document_path=path,
                    document_hash=document_hash,
                    page_no=page.page_no,
                    size=page.size,
                    assembled=copy.deepcopy(page.assembled),
                )
        assert all(output is not None for output in outputs)
        return [output for output in outputs if output is not None]

    def _run_encoder_accelerated_across_documents(
        self,
        sources: list[DoclingPageSource],
        layouts: list[Any],
        ocr_results: list[DoclingOcrResult],
    ) -> list[DoclingPageAssembly]:
        """跨 document 合并 encoder jobs，并直接保留候选 table 输出。"""

        if not sources:
            return []
        pages_by_index: dict[int, Any] = {}
        for path, document_hash, indices in _group_indices_by_document(
            sources
        ):
            pages = self._pages_for_indices(
                indices,
                sources,
                layouts,
                ocr_results,
            )
            context = _conversion_result(path, document_hash, pages)
            pages = list(self.postprocess(context, pages))
            for index, page in zip(indices, pages):
                pages_by_index[index] = page

        pages = [pages_by_index[index] for index in range(len(sources))]
        pages = self._run_encoder_batched_table(
            _conversion_result(
                sources[0].document_path,
                sources[0].document_hash,
                pages,
            ),
            pages,
        )
        return self._assemble_pages_across_documents(
            sources,
            {index: page for index, page in enumerate(pages)},
        )

    def _assemble_pages_across_documents(
        self,
        sources: list[DoclingPageSource],
        pages_by_index: dict[int, Any],
    ) -> list[DoclingPageAssembly]:
        """按原 document 边界组装全局索引中的已处理 Page。"""

        outputs: list[DoclingPageAssembly | None] = [None] * len(sources)
        for path, document_hash, indices in _group_indices_by_document(
            sources
        ):
            pages = [pages_by_index[index] for index in indices]
            assembled = list(
                self.assemble(
                    _conversion_result(path, document_hash, pages),
                    pages,
                )
            )
            for index, page in zip(indices, assembled):
                outputs[index] = DoclingPageAssembly(
                    document_path=path,
                    document_hash=document_hash,
                    page_no=page.page_no,
                    size=page.size,
                    assembled=copy.deepcopy(page.assembled),
                )
        assert all(output is not None for output in outputs)
        return [output for output in outputs if output is not None]

    def _pages_for_indices(
        self,
        indices: list[int],
        sources: list[DoclingPageSource],
        layouts: list[Any],
        ocr_results: list[DoclingOcrResult],
    ) -> list[Any]:
        """从 immutable values 重建一组临时、独占的 Docling Page。"""

        return [
            page_from_source(
                sources[index],
                layout=layouts[index],
                ocr=ocr_results[index],
                render_cache=self.render_cache,
            )
            for index in indices
        ]

    def _run_encoder_batched_table(
        self,
        conv_res: Any,
        pages: list[Any],
    ) -> list[Any]:
        """预计算多个 table crop encoder output，再复用原 batch=1 decoder。

        当前 TableModel04_rs decoder 通过 scalar ``.item()`` 生成每张表的 tag sequence，
        不能直接做 batch>1 decode。本方法只 batch encoder；临时替换 ``_encoder`` 让原
        ``TableStructureModel`` 每次 predict 消费对应的 `[1,...]` encoder output。该
        actor 必须维持 max_concurrency=1，避免 hook 在并发 RPC 间串扰。
        """

        encoder_outputs, batch_count = self._table_encoder_outputs(pages)
        self.last_table_batch_audit = {
            "jobs": len(encoder_outputs),
            "batches": batch_count,
            "table_fallbacks": 0,
            "batch_errors": 0,
        }
        if not encoder_outputs:
            return list(self.table(conv_res, pages))

        model = self.table.tf_predictor.get_model()
        original_encoder = model._encoder
        pending = deque(encoder_outputs)

        class _EncoderReplay:
            """非 Module wrapper，避免 torch Module attribute registration 限制。"""

            def __call__(self, _: Any) -> Any:
                if not pending:
                    raise RuntimeError(
                        "TableFormer encoder call count exceeded prepared jobs"
                    )
                return pending.popleft()

        replay = _EncoderReplay()
        # `_encoder` 原本是 torch.nn.Module，普通 setattr 会拒绝 function/non-Module。
        # 仅在 actor-local、单并发 experimental table batching 内绕过 Module 的注册
        # 逻辑；finally 恢复原 Module。这里不改变模型 state_dict 或参数。
        object.__setattr__(model, "_encoder", replay)
        try:
            result = list(self.table(conv_res, pages))
            if pending:
                raise RuntimeError(
                    "prepared TableFormer encoder outputs were not consumed"
                )
            return result
        finally:
            object.__setattr__(model, "_encoder", original_encoder)

    def _table_encoder_outputs(
        self,
        pages: list[Any],
    ) -> tuple[list[Any], int]:
        """严格复现原 crop/preprocess，再对 TableFormer encoder 做安全 stack。

        这里的 crop 顺序与 ``TableStructureModel.predict_tables`` 的 page/cluster 顺序相同。
        因此临时 encoder hook 中第 N 次调用严格对应第 N 个 table。没有 table 时不调用
        model。该方法只生成 encoder output，不改变 tokens、matching 或 Docling response
        postprocess。
        """

        import numpy
        import torch
        from docling_core.types.doc import DocItemLabel

        predictor = self.table.tf_predictor
        jobs: list[EncoderTableJob] = []
        order = 0
        for page_index, page in enumerate(pages):
            assert page.predictions.layout is not None
            assert page.size is not None
            page_image = numpy.asarray(
                page.get_image(scale=self.table.scale)
            )
            resized, scale_factor = predictor.resize_img(
                page_image,
                height=1024,
            )
            for cluster in page.predictions.layout.clusters:
                if cluster.label not in {
                    DocItemLabel.TABLE,
                    DocItemLabel.DOCUMENT_INDEX,
                }:
                    continue
                box = [
                    round(cluster.bbox.l) * self.table.scale * scale_factor,
                    round(cluster.bbox.t) * self.table.scale * scale_factor,
                    round(cluster.bbox.r) * self.table.scale * scale_factor,
                    round(cluster.bbox.b) * self.table.scale * scale_factor,
                ]
                crop = resized[
                    round(box[1]) : round(box[3]),
                    round(box[0]) : round(box[2]),
                ]
                if crop.size == 0:
                    raise ValueError("TableFormer crop is empty")
                jobs.append(
                    EncoderTableJob(
                        page=page_index,
                        table=int(cluster.id),
                        order=order,
                        payload=predictor._prepare_image(crop),
                        metadata={"batch_key": "tableformer-encoder-v1"},
                    )
                )
                order += 1

        planner = TableBatchPlanner(
            max_batch_size=self.table_batch_max_jobs
        )
        outputs: list[Any] = []
        batches = planner.plan(jobs)
        model = predictor.get_model()
        with torch.no_grad():
            for batch in batches:
                tensors = [job.payload for job in batch.jobs]
                encoder_batch = model._encoder(torch.cat(tensors, dim=0))
                outputs.extend(
                    encoder_batch[index : index + 1]
                    for index in range(len(batch.jobs))
                )
        return outputs, len(batches)

    def _restore_reference_table_semantics(
        self,
        candidate_pages: list[Any],
        reference_pages: list[Any],
    ) -> None:
        """按 table cluster 逐 job 比较 candidate/reference，并替换漂移结果。"""

        for candidate, reference in zip(candidate_pages, reference_pages):
            candidate_prediction = candidate.predictions.tablestructure
            reference_prediction = reference.predictions.tablestructure
            if candidate_prediction is None or reference_prediction is None:
                candidate.predictions.tablestructure = copy.deepcopy(
                    reference_prediction
                )
                self.last_table_batch_audit["table_fallbacks"] += 1
                continue
            candidate_map = candidate_prediction.table_map
            reference_map = reference_prediction.table_map
            if set(candidate_map) != set(reference_map):
                candidate.predictions.tablestructure = copy.deepcopy(
                    reference_prediction
                )
                self.last_table_batch_audit["table_fallbacks"] += len(
                    reference_map
                )
                continue
            for table_id, reference_table in reference_map.items():
                candidate_table = candidate_map[table_id]
                if not self._same_table_semantics(
                    candidate_table,
                    reference_table,
                ):
                    candidate_map[table_id] = copy.deepcopy(reference_table)
                    self.last_table_batch_audit["table_fallbacks"] += 1

    @staticmethod
    def _same_table_semantics(left: Any, right: Any) -> bool:
        """比较 Table Pydantic 语义值，不比较对象 identity 或运行时 metadata。"""

        if hasattr(left, "model_dump") and hasattr(right, "model_dump"):
            return left.model_dump(mode="json") == right.model_dump(
                mode="json"
            )
        return left == right


class ReduceDoclingDocument:
    """按 page ordinal 恢复 assemblies，并复用 Docling reading-order。"""

    def __init__(self) -> None:
        """初始化 Docling ReadingOrderModel。"""

        from docling.models.stages.reading_order.readingorder_model import (
            ReadingOrderModel,
            ReadingOrderOptions,
        )

        self.reading_order = ReadingOrderModel(ReadingOrderOptions())

    def run(
        self,
        groups: list[list[DoclingPageAssembly]],
    ) -> list[dict[str, Any]]:
        """生成与原生可比较的 Markdown 和结构计数。"""

        from docling.datamodel.base_models import AssembledUnit, Page

        outputs = []
        for values in groups:
            ordered = sorted(values, key=lambda value: value.page_no)
            if not ordered:
                raise ValueError("Docling document has no pages")
            expected = list(range(1, len(ordered) + 1))
            if [value.page_no for value in ordered] != expected:
                raise ValueError("Docling page ordinals are not contiguous")
            pages = [
                Page(
                    page_no=value.page_no,
                    size=value.size,
                    assembled=value.assembled,
                )
                for value in ordered
            ]
            first = ordered[0]
            conv_res = _conversion_result(
                first.document_path,
                first.document_hash,
                pages,
            )
            conv_res.assembled = AssembledUnit(
                elements=[
                    element
                    for value in ordered
                    for element in value.assembled.elements
                ],
                headers=[
                    element
                    for value in ordered
                    for element in value.assembled.headers
                ],
                body=[
                    element
                    for value in ordered
                    for element in value.assembled.body
                ],
            )
            conv_res.document = self.reading_order(conv_res)
            markdown = conv_res.document.export_to_markdown()
            outputs.append(
                {
                    "pdf": Path(first.document_path).stem,
                    "pages": len(ordered),
                    "markdown": markdown,
                    "markdown_chars": len(markdown),
                    "texts": len(conv_res.document.texts),
                    "tables": len(conv_res.document.tables),
                    "pictures": len(conv_res.document.pictures),
                }
            )
        return outputs
