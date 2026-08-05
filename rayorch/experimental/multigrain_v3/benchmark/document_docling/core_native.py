"""Docling core-stage 基线的整文档 native runner。

本模块只服务于 ``core_v3`` 的公平实验。它保留 Docling 的整文档
``StandardPdfPipeline``，不经过 V3，也不改变任何 page-stage 的业务模型。

``doc_batch_size`` 和 ``doc_batch_concurrency`` 是 Docling 的进程级 perf settings。
因此一次实验进程内必须串行运行不同 native 配置；本模块在每次运行结束后恢复原设置。
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator


@dataclass(frozen=True, slots=True)
class NativeCoreConfig:
    """一个 native Docling 运行的可复现配置。

    ``doc_batch_*`` 控制多个 document conversion 是否并行；三个 stage batch size 固定
    为与 V3 相同的 heavy-stage cap。这样 ``native_default`` 与 ``native_tuned`` 只在
    document admission/concurrency 上不同。
    """

    device: str = "cpu"
    num_threads: int = 4
    doc_batch_size: int = 1
    doc_batch_concurrency: int = 1
    layout_batch_size: int = 4
    ocr_batch_size: int = 4
    table_batch_size: int = 4

    def __post_init__(self) -> None:
        """拒绝无法解释的 Docling document batching 配置。"""

        if self.num_threads <= 0:
            raise ValueError("num_threads must be positive")
        if self.doc_batch_size <= 0:
            raise ValueError("doc_batch_size must be positive")
        if self.doc_batch_concurrency <= 0:
            raise ValueError("doc_batch_concurrency must be positive")
        if self.doc_batch_size < self.doc_batch_concurrency:
            raise ValueError(
                "doc_batch_size must be >= doc_batch_concurrency"
            )
        if min(
            self.layout_batch_size,
            self.ocr_batch_size,
            self.table_batch_size,
        ) <= 0:
            raise ValueError("native stage batch sizes must be positive")


@dataclass(frozen=True, slots=True)
class NativeCoreRun:
    """一次整文档 Docling baseline 的计时与规范化业务输出。

    ``startup_s`` 是构造 converter 并显式调用私有 ``_get_pipeline()`` 到模型 ready
    的时间。该私有调用仅用于实验把 startup 和 measured conversion 分开；生产代码不依赖
    该方法。``documents`` 中保留 Markdown 和结构计数，以便 runner 在进程内做 parity，
    但 CLI 写出的 JSON 只输出摘要，不写入大业务内容。
    """

    startup_s: float
    measured_s: float
    documents: tuple[dict[str, Any], ...]


@contextmanager
def _docling_perf_settings(
    config: NativeCoreConfig,
) -> Iterator[None]:
    """临时设置并恢复 Docling 全局 document batching 参数。"""

    from docling.datamodel.settings import settings

    previous_batch_size = settings.perf.doc_batch_size
    previous_concurrency = settings.perf.doc_batch_concurrency
    settings.perf.doc_batch_size = config.doc_batch_size
    settings.perf.doc_batch_concurrency = config.doc_batch_concurrency
    try:
        yield
    finally:
        settings.perf.doc_batch_size = previous_batch_size
        settings.perf.doc_batch_concurrency = previous_concurrency


def _normalise_document(conversion: Any) -> dict[str, Any]:
    """抽取 V3 correctness gate 所需的 Docling 文档输出。"""

    document = conversion.document
    markdown = document.export_to_markdown()
    return {
        "pages": len(conversion.pages),
        "markdown": markdown,
        "markdown_chars": len(markdown),
        "texts": len(document.texts),
        "tables": len(document.tables),
        "pictures": len(document.pictures),
    }


def run_native_core(
    paths: list[str],
    *,
    config: NativeCoreConfig,
) -> NativeCoreRun:
    """运行 Docling native PDF pipeline，并分离 startup 与 conversion wall。

    调用方须确保 ``paths`` 有稳定顺序。Docling 的 ``convert_all`` 保持输入顺序，因此
    返回 documents 与 source paths 逐项对应。
    """

    if not paths:
        raise ValueError("paths must not be empty")

    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    with _docling_perf_settings(config):
        options = ThreadedPdfPipelineOptions(
            accelerator_options=AcceleratorOptions(
                device=config.device,
                num_threads=config.num_threads,
            ),
            layout_batch_size=config.layout_batch_size,
            ocr_batch_size=config.ocr_batch_size,
            table_batch_size=config.table_batch_size,
        )

        started = time.perf_counter()
        converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=options,
                )
            },
        )
        # DocumentConverter 的 pipeline cache 是 *converter instance* 级别的。单独构造
        # converter 不加载模型；这里把实际 pipeline readiness 计入 startup，避免将 Python
        # 对象构造噪声与 V3 actor/model readiness 混为一谈。
        converter._get_pipeline(InputFormat.PDF)
        startup_s = time.perf_counter() - started

        measured_started = time.perf_counter()
        conversions = tuple(converter.convert_all(paths))
        measured_s = time.perf_counter() - measured_started

    if len(conversions) != len(paths):
        raise RuntimeError("Docling native changed document cardinality")
    return NativeCoreRun(
        startup_s=startup_s,
        measured_s=measured_s,
        documents=tuple(
            _normalise_document(conversion)
            for conversion in conversions
        ),
    )
