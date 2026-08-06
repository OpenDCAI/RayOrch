"""TableJob-first V3 orchestration for Docling core models."""

from __future__ import annotations

from typing import Any

from ...api import Expand, Map, Pipeline, Reduce
from .core_stages import (
    DoclingLayoutPages,
    DoclingOcrPages,
    ExpandDoclingPages,
    ReduceDoclingDocument,
)
from .core_table_workflow import (
    DoclingPostprocessPages,
    DoclingTableCore,
    ExpandDoclingTableJobs,
    ExpandDoclingTableV2Jobs,
    ReduceDoclingPage,
)
from .tableformer_v1_batch import DoclingTableFormerV1BatchCore
from .tableformer_v2_batch import DoclingTableFormerV2BatchCore


class DoclingTableJobV3Pipeline(Pipeline):
    """Page->TableJob->Page->Document with elastic cross-PDF batching."""

    table_job_stage = ExpandDoclingTableJobs
    table_core_stage = DoclingTableCore
    table_batch_modes = {
        "reference",
        "encoder_shadow",
        "encoder_accelerated",
        "decoder_accelerated",
    }

    def __init__(
        self,
        *,
        layout_replicas: int = 1,
        layout_batch_size: int = 4,
        layout_batch_wait_ms: float = 20.0,
        layout_num_gpus: float = 0.0,
        ocr_replicas: int = 1,
        ocr_batch_size: int = 4,
        ocr_batch_wait_ms: float = 2.0,
        table_replicas: int = 1,
        table_batch_size: int = 4,
        table_core_batch_size: int | None = None,
        table_batch_wait_ms: float = 2.0,
        table_num_gpus: float = 0.0,
        reduce_replicas: int = 1,
        parse_replicas: int = 1,
        parse_batch_size: int = 1,
        parse_batch_wait_ms: float = 20.0,
        image_scales: tuple[float, ...] = (1.0, 2.0, 3.0),
        batch_scope: str = "elastic",
        device: str = "cpu",
        layout_device: str | None = None,
        ocr_device: str | None = None,
        table_device: str | None = None,
        num_threads: int = 4,
        actor_num_cpus: float = 1.0,
        parse_actor_concurrency: int = 1,
        layout_actor_concurrency: int = 1,
        ocr_actor_concurrency: int = 1,
        table_actor_concurrency: int = 1,
        table_batch_mode: str = "reference",
        table_batch_max_jobs: int = 16,
        ocr_batch_mode: str = "reference",
        ocr_recognition_batch_size: int = 6,
        postprocess_replicas: int | None = None,
        table_prepare_replicas: int | None = None,
        page_reduce_replicas: int | None = None,
    ) -> None:
        if ocr_batch_mode not in {
            "reference",
            "recognition_shadow",
            "recognition_accelerated",
        }:
            raise ValueError("unsupported ocr_batch_mode")
        if ocr_recognition_batch_size <= 0:
            raise ValueError("ocr_recognition_batch_size must be positive")
        if ocr_batch_mode != "reference" and ocr_actor_concurrency != 1:
            raise ValueError(
                "ocr_actor_concurrency must be 1 for recognition batching"
            )
        if table_batch_mode not in self.table_batch_modes:
            raise ValueError("unsupported table_batch_mode")
        if table_batch_max_jobs <= 0:
            raise ValueError("table_batch_max_jobs must be positive")
        table_core_batch_size = table_core_batch_size or table_batch_size
        if table_core_batch_size <= 0:
            raise ValueError("table_core_batch_size must be positive")
        if table_batch_mode != "reference" and table_actor_concurrency != 1:
            raise ValueError(
                "table_actor_concurrency must be 1 for table batching"
            )

        layout_device = layout_device or device
        ocr_device = ocr_device or device
        table_device = table_device or device
        postprocess_replicas = postprocess_replicas or reduce_replicas
        table_prepare_replicas = table_prepare_replicas or reduce_replicas
        page_reduce_replicas = page_reduce_replicas or reduce_replicas

        self.expand = (
            Expand(ExpandDoclingPages)
            .pre_init(image_scales=image_scales)
            .ray_options(
                replicas=parse_replicas,
                batch_size=parse_batch_size,
                max_batch_wait_ms=parse_batch_wait_ms,
                num_cpus=1,
                max_concurrency=parse_actor_concurrency,
            )
        )
        self.layout = (
            Map(DoclingLayoutPages)
            .pre_init(device=layout_device, num_threads=num_threads)
            .ray_options(
                replicas=layout_replicas,
                batch_size=layout_batch_size,
                max_batch_wait_ms=layout_batch_wait_ms,
                batch_scope=batch_scope,
                num_cpus=actor_num_cpus,
                num_gpus=layout_num_gpus,
                max_concurrency=layout_actor_concurrency,
            )
        )
        self.ocr = (
            Map(DoclingOcrPages)
            .pre_init(
                device=ocr_device,
                num_threads=num_threads,
                ocr_batch_mode=ocr_batch_mode,
                ocr_recognition_batch_size=ocr_recognition_batch_size,
            )
            .ray_options(
                replicas=ocr_replicas,
                batch_size=ocr_batch_size,
                max_batch_wait_ms=ocr_batch_wait_ms,
                batch_scope=batch_scope,
                num_cpus=actor_num_cpus,
                max_concurrency=ocr_actor_concurrency,
            )
        )
        self.postprocess = Map(DoclingPostprocessPages).ray_options(
            replicas=postprocess_replicas,
            batch_size=table_batch_size,
            max_batch_wait_ms=table_batch_wait_ms,
            batch_scope=batch_scope,
            num_cpus=actor_num_cpus,
            max_concurrency=1,
        )
        self.table_jobs = Expand(self.table_job_stage).ray_options(
            replicas=table_prepare_replicas,
            batch_size=table_batch_size,
            max_batch_wait_ms=table_batch_wait_ms,
            batch_scope=batch_scope,
            num_cpus=actor_num_cpus,
            max_concurrency=1,
        )
        self.table_core = (
            Map(self.table_core_stage)
            .pre_init(
                device=table_device,
                num_threads=num_threads,
                table_batch_mode=table_batch_mode,
                table_batch_max_jobs=table_batch_max_jobs,
            )
            .ray_options(
                replicas=table_replicas,
                batch_size=table_core_batch_size,
                max_batch_wait_ms=table_batch_wait_ms,
                batch_scope=batch_scope,
                num_cpus=actor_num_cpus,
                num_gpus=table_num_gpus,
                max_concurrency=table_actor_concurrency,
            )
        )
        self.page_reduce = Reduce(ReduceDoclingPage).ray_options(
            replicas=page_reduce_replicas,
            batch_size=table_batch_size,
            num_cpus=1,
        )
        self.reduce = Reduce(ReduceDoclingDocument).ray_options(
            replicas=reduce_replicas,
            batch_size=2,
            num_cpus=1,
        )

    def forward(self, documents):
        pages = self.expand(documents)
        layouts = self.layout(pages)
        ocr_results = self.ocr(pages, layouts)
        postprocessed = self.postprocess(pages, layouts, ocr_results)
        table_jobs = self.table_jobs(postprocessed)
        tables = self.table_core(table_jobs)
        assembled_pages = self.page_reduce(
            anchor=postprocessed,
            members=tables,
            pages=postprocessed,
        )
        return self.reduce(anchor=documents, members=assembled_pages)


class DoclingTableFormerV2BatchV3Pipeline(DoclingTableJobV3Pipeline):
    """Independent V3 authoring of RapidOCR-direct + TableFormerV2 batch."""

    table_job_stage = ExpandDoclingTableV2Jobs
    table_core_stage = DoclingTableFormerV2BatchCore
    table_batch_modes = {"v2_batch"}

    def __init__(self, **options: Any) -> None:
        options.setdefault("table_batch_mode", "v2_batch")
        super().__init__(**options)


class DoclingTableFormerV1BatchV3Pipeline(DoclingTableJobV3Pipeline):
    """V1 golden semantics behind an explicit, mutation-free batch kernel."""

    table_job_stage = ExpandDoclingTableJobs
    table_core_stage = DoclingTableFormerV1BatchCore
    table_batch_modes = {"v1_batch"}

    def __init__(self, **options: Any) -> None:
        options.setdefault("table_batch_mode", "v1_batch")
        super().__init__(**options)
