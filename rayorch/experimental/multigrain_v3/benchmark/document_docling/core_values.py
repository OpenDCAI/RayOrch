"""Docling core-stage pipeline 的可序列化 value DTO。

这些值只承载业务数据，不包含 Ray handle、V3 lineage 或 Docling native PDF pointers。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from io import BytesIO
from typing import Any


@dataclass(frozen=True, slots=True)
class DoclingPageSource:
    """一个 page grain 的稳定源值。"""

    document_path: str
    document_hash: str
    page_no: int
    size: Any
    segmented_page: Any
    images_by_scale: dict[float, bytes]


@dataclass(frozen=True, slots=True)
class DoclingOcrResult:
    """OCR 后的 segmented page。"""

    segmented_page: Any


@dataclass(frozen=True, slots=True)
class DoclingPostprocessedPage:
    """Table fan-out 和 Page Reduce 共享的 immutable page value。"""

    source: DoclingPageSource
    layout: Any
    ocr: DoclingOcrResult


@dataclass(frozen=True, slots=True)
class DoclingTableJob:
    """一个可被 V3 跨 PDF 重组的 TableFormer 核心模型 job。"""

    page_no: int
    table_cluster: Any
    iocr_page: dict[str, Any]
    table_bbox: tuple[float, float, float, float]
    table_image: Any
    scale_factor: float
    scale: float


@dataclass(frozen=True, slots=True)
class DoclingPageAssembly:
    """Document Reduce 所需的轻量 page assembly。"""

    document_path: str
    document_hash: str
    page_no: int
    size: Any
    assembled: Any


class DetachedDoclingPageBackend:
    """由 Docling core models 使用的可序列化 PdfPageBackend 替代物。"""

    def __init__(
        self,
        source: DoclingPageSource,
        render_cache: "PdfRenderCache | None" = None,
    ) -> None:
        """保存 immutable page source 和可选 actor-local render cache。"""

        self.source = source
        self.render_cache = render_cache

    @property
    def page_no(self) -> int:
        """返回 1-based page number。"""

        return self.source.page_no

    def is_valid(self) -> bool:
        """source 只从有效原生 page 构造。"""

        return True

    def get_segmented_page(self):
        """返回本次临时 Page 独占的 segmented page。"""

        return self.source.segmented_page

    def get_text_cells(self):
        """返回 textline cells。"""

        return self.source.segmented_page.textline_cells

    def get_bitmap_rects(self, scale: float = 1):
        """按原生 backend 语义返回 bitmap boxes。"""

        boxes = []
        for resource in self.source.segmented_page.bitmap_resources:
            box = resource.rect.to_bounding_box().to_top_left_origin(
                self.source.size.height
            )
            if box.area() > 0:
                boxes.append(box.scaled(scale=scale))
        return boxes

    def get_text_in_rect(self, bbox):
        """从 segmented textline cells 中恢复矩形内文本。"""

        text = []
        for cell in self.source.segmented_page.textline_cells:
            cell_bbox = cell.rect.to_bounding_box()
            if cell_bbox.intersection_over_self(bbox) > 0.5:
                text.append(cell.text)
        return " ".join(text)

    def get_page_image(self, scale: float = 1, cropbox=None):
        """解码对应 scale 的 PNG，并按需 resize/crop。"""

        from PIL import Image

        available = self.source.images_by_scale
        if scale in available:
            source_scale = scale
            image = Image.open(BytesIO(available[source_scale])).convert("RGB")
            if cropbox is not None:
                image = image.crop(
                    cropbox.to_top_left_origin(
                        page_height=self.source.size.height
                    )
                    .scaled(scale=scale)
                    .as_tuple()
                )
            return image
        if self.render_cache is not None:
            return self.render_cache.render(
                self.source,
                scale=scale,
                cropbox=cropbox,
            )
        if available:
            source_scale = min(
                available,
                key=lambda candidate: abs(candidate - scale),
            )
            image = Image.open(BytesIO(available[source_scale])).convert("RGB")
            image = image.resize(
                (
                    round(self.source.size.width * scale),
                    round(self.source.size.height * scale),
                )
            )
            if cropbox is not None:
                image = image.crop(
                    cropbox.to_top_left_origin(
                        page_height=self.source.size.height
                    )
                    .scaled(scale=scale)
                    .as_tuple()
                )
            return image
        raise ValueError("page has no image or render cache")

    def get_size(self):
        """返回 PDF page size。"""

        return self.source.size

    def unload(self) -> None:
        """没有 native resource。"""

        return None


class PdfRenderCache:
    """一个 actor 内复用的 pypdfium2 document/image renderer。"""

    def __init__(self) -> None:
        """初始化 document cache。"""

        self.documents: dict[str, Any] = {}

    def render(
        self,
        source: DoclingPageSource,
        *,
        scale: float,
        cropbox: Any | None,
    ):
        """复现 DoclingParsePageBackend 的 sharpened PDFium render。"""

        import pypdfium2 as pdfium
        from docling_core.types.doc import BoundingBox, CoordOrigin
        from docling.utils.locks import pypdfium2_lock

        with pypdfium2_lock:
            document = self.documents.get(source.document_path)
            if document is None:
                document = pdfium.PdfDocument(source.document_path)
                self.documents[source.document_path] = document
            page = document[source.page_no - 1]
            if cropbox is None:
                target = BoundingBox(
                    l=0,
                    r=source.size.width,
                    t=0,
                    b=source.size.height,
                    coord_origin=CoordOrigin.TOPLEFT,
                )
                pad = BoundingBox(
                    l=0,
                    r=0,
                    t=0,
                    b=0,
                    coord_origin=CoordOrigin.BOTTOMLEFT,
                )
            else:
                target = cropbox
                pad = cropbox.to_bottom_left_origin(
                    source.size.height
                ).model_copy()
                pad.r = source.size.width - pad.r
                pad.t = source.size.height - pad.t
            bitmap = page.render(
                scale=scale * 1.5,
                rotation=0,
                crop=pad.as_tuple(),
            )
            image = bitmap.to_pil().copy()
            bitmap.close()
            page.close()
        return image.resize(
            (
                round(target.width * scale),
                round(target.height * scale),
            )
        )


def page_from_source(
    source: DoclingPageSource,
    *,
    layout: Any | None = None,
    ocr: DoclingOcrResult | None = None,
    render_cache: PdfRenderCache | None = None,
):
    """为一个 actor 调用构造临时 Docling Page。"""

    from docling.datamodel.base_models import Page, PagePredictions

    segmented = copy.deepcopy(
        ocr.segmented_page if ocr is not None else source.segmented_page
    )
    page = Page(
        page_no=source.page_no,
        size=source.size,
        parsed_page=segmented,
        predictions=PagePredictions(layout=copy.deepcopy(layout)),
    )
    detached_source = DoclingPageSource(
        document_path=source.document_path,
        document_hash=source.document_hash,
        page_no=source.page_no,
        size=source.size,
        segmented_page=segmented,
        images_by_scale=source.images_by_scale,
    )
    page._backend = DetachedDoclingPageBackend(
        detached_source,
        render_cache,
    )
    return page
