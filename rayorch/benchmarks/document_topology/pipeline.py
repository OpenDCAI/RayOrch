"""Document -> Page -> TableJob -> Page -> Document reference Pipeline."""

from __future__ import annotations

from typing import cast

from rayorch import F, Pipeline, Port, RayModule

from .udfs import (
    ExpandTableJobs,
    LayoutPages,
    OcrPages,
    ParseDocuments,
    PostprocessPages,
    ReduceDocument,
    ReducePage,
    TableCore,
)


class DocumentTopologyPipeline(Pipeline):
    def __init__(
        self,
        *,
        workers: int = 2,
        batch_size: int = 4,
    ) -> None:
        stage = {"replicas": workers, "batch_size": batch_size, "num_cpus": 0}
        self.parse = RayModule(ParseDocuments).ray_options(
            replicas=workers,
            batch_size=1,
            num_cpus=0,
        )
        self.layout = RayModule(LayoutPages).ray_options(**stage)
        self.ocr = RayModule(OcrPages).ray_options(**stage)
        self.postprocess = RayModule(PostprocessPages).ray_options(**stage)
        self.table_prepare = RayModule(ExpandTableJobs).ray_options(**stage)
        self.table_core = RayModule(TableCore).ray_options(**stage)
        self.page_assemble = RayModule(ReducePage).ray_options(**stage)
        self.document_assemble = RayModule(ReduceDocument).ray_options(
            replicas=1,
            batch_size=2,
            num_cpus=0,
        )

    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        documents: Port,
    ) -> Port:
        pages = F.expand(cast(Port, self.parse(documents)))
        layouts = cast(Port, self.layout(pages))
        ocr = cast(Port, self.ocr(pages, layouts))
        postprocessed = cast(Port, self.postprocess(pages, layouts, ocr))

        table_jobs = F.expand(cast(Port, self.table_prepare(postprocessed)))
        tables = cast(Port, self.table_core(table_jobs))
        assembled_pages = cast(
            Port,
            self.page_assemble(F.reduce(tables), postprocessed),
        )
        return cast(Port, self.document_assemble(F.reduce(assembled_pages)))


__all__ = ["DocumentTopologyPipeline"]
