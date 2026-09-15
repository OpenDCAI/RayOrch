"""Declarative RayOrch graph for the MinerU workload."""

from __future__ import annotations

from typing import Any, cast

from rayorch import F, Pipeline, Port, RayModule

from .udfs import MinerUAssembleDoc, MinerUPdfToPages, MinerUVlmOcrPage, PdfMetadata


class MinerUPipeline(Pipeline):
    """Model the real PDF -> page -> OCR -> document MinerU pipeline."""

    def __init__(
        self,
        *,
        output_dir: str,
        model: str,
        render_dpi: int = 200,
        gpu_memory_utilization: float = 0.8,
        **stages: dict[str, Any],
    ) -> None:
        """Build four Calls from ``render/ocr/metadata/assemble`` Ray options."""

        expected = {"render", "ocr", "metadata", "assemble"}
        if set(stages) != expected:
            raise ValueError(
                "MinerUPipeline requires stage options for "
                + ", ".join(sorted(expected))
            )

        self.render = (
            RayModule(MinerUPdfToPages)
            .pre_init(dpi=render_dpi)
            .ray_options(**stages["render"])
        )
        self.ocr = (
            RayModule(MinerUVlmOcrPage)
            .pre_init(
                model=model,
                gpu_memory_utilization=gpu_memory_utilization,
            )
            .ray_options(**stages["ocr"])
        )
        self.metadata = RayModule(PdfMetadata).ray_options(**stages["metadata"])
        self.assemble = (
            RayModule(MinerUAssembleDoc)
            .pre_init(output_dir=output_dir)
            .ray_options(**stages["assemble"])
        )

    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        pdfs: Port,
    ):
        """Declare 1:M page work and ordered M:1 document reconstruction."""

        pages = F.expand(cast(Port, self.render(pdfs)))
        contents = cast(Port, self.ocr(pages))
        stems = cast(Port, self.metadata(pdfs))
        content_groups, ordered_page_groups = F.reduce_aligned(
            contents,
            pages,
            members=contents,
        )
        return self.assemble(content_groups, ordered_page_groups, stems)


__all__ = ["MinerUPipeline"]
