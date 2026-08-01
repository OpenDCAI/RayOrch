"""Docling 文档实验的框架无关数据结构与业务函数。

PDF rendering 与最终 ordered assembly 使用 stdlib/PDFium；页面解析通过延迟 import
Docling，使主环境未安装 Docling 时仍可导入和测试本模块。
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class DoclingPage:
    """一个文档内按 ordinal 编号的 PNG page payload。"""

    pdf_path: str
    page_ordinal: int
    png_bytes: bytes


@dataclass(frozen=True, slots=True)
class DoclingPageResult:
    """Docling 对一个 page 的 Markdown 结果。"""

    page_ordinal: int
    markdown: str


def render_pdf(path: str, *, scale: float = 1.5) -> list[DoclingPage]:
    """使用 PDFium 将 PDF 动态展开为有序 PNG pages。"""

    import pypdfium2 as pdfium

    if scale <= 0:
        raise ValueError("scale must be positive")
    document = pdfium.PdfDocument(path)
    pages = []
    try:
        for ordinal in range(len(document)):
            image = document[ordinal].render(scale=scale).to_pil()
            stream = BytesIO()
            image.save(stream, format="PNG")
            pages.append(
                DoclingPage(
                    pdf_path=path,
                    page_ordinal=ordinal,
                    png_bytes=stream.getvalue(),
                )
            )
    finally:
        document.close()
    return pages


def create_converter(
    *,
    input_format: str,
    device: str = "cpu",
    num_threads: int = 4,
    do_ocr: bool = True,
    do_table_structure: bool = True,
) -> Any:
    """创建配置一致的 Docling image/PDF persistent converter。"""

    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import (
        DocumentConverter,
        ImageFormatOption,
        PdfFormatOption,
    )

    options = PdfPipelineOptions(
        accelerator_options=AcceleratorOptions(
            device=device,
            num_threads=num_threads,
        ),
        do_ocr=do_ocr,
        do_table_structure=do_table_structure,
    )
    if input_format == "image":
        format_value = InputFormat.IMAGE
        format_option = ImageFormatOption(pipeline_options=options)
    elif input_format == "pdf":
        format_value = InputFormat.PDF
        format_option = PdfFormatOption(pipeline_options=options)
    else:
        raise ValueError(f"unsupported Docling input format: {input_format}")
    return DocumentConverter(
        allowed_formats=[format_value],
        format_options={format_value: format_option},
    )


def parse_page(converter: Any, page: DoclingPage) -> DoclingPageResult:
    """把单页 PNG 送入 persistent Docling converter。"""

    from docling_core.types.io import DocumentStream

    result = converter.convert(
        DocumentStream(
            name=f"{Path(page.pdf_path).stem}-p{page.page_ordinal}.png",
            stream=BytesIO(page.png_bytes),
        )
    )
    return DoclingPageResult(
        page_ordinal=page.page_ordinal,
        markdown=result.document.export_to_markdown(),
    )


def parse_pages(
    converter: Any,
    pages: list[DoclingPage],
) -> list[DoclingPageResult]:
    """使用 Docling `convert_all` 处理一个物理 page batch。"""

    from docling_core.types.io import DocumentStream

    streams = [
        DocumentStream(
            name=f"{Path(page.pdf_path).stem}-p{page.page_ordinal}.png",
            stream=BytesIO(page.png_bytes),
        )
        for page in pages
    ]
    results = list(converter.convert_all(streams))
    if len(results) != len(pages):
        raise ValueError("Docling convert_all changed page cardinality")
    return [
        DoclingPageResult(
            page_ordinal=page.page_ordinal,
            markdown=result.document.export_to_markdown(),
        )
        for page, result in zip(pages, results)
    ]


def assemble_document(
    pages: list[DoclingPageResult],
) -> dict[str, Any]:
    """按 page ordinal 组装 Markdown，并拒绝丢页/重复页。"""

    ordered = sorted(pages, key=lambda page: page.page_ordinal)
    ordinals = [page.page_ordinal for page in ordered]
    if ordinals != list(range(len(ordered))):
        raise ValueError(f"invalid page ordinals: {ordinals}")
    markdown = "\n\n<!-- page-break -->\n\n".join(
        page.markdown for page in ordered
    )
    return {
        "pages": len(ordered),
        "page_ordinals": tuple(ordinals),
        "markdown": markdown,
        "markdown_chars": len(markdown),
    }
