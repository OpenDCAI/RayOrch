"""Docling Document→Page→TableJob→Page→Document on Multigrain v3.6.

The business UDFs are imported unchanged from the V3 benchmark.  This module
only replaces V3 ``Expand/Map/Reduce`` authoring and its executor with
``RayModule + F.*``, the v3.6 compiler, RuntimePlan, MicrobatchEngine and Executor.
"""

from __future__ import annotations

from typing import Any, cast

from ....multigrain_v3.benchmark.document_docling.core_stages import (
    DoclingLayoutPages,
    DoclingOcrPages,
    ExpandDoclingPages,
    ReduceDoclingDocument,
)
from ....multigrain_v3.benchmark.document_docling.core_table_workflow import (
    DoclingPostprocessPages,
    DoclingTableCore,
    ExpandDoclingTableJobs,
    ExpandDoclingTableV2Jobs,
    ReduceDoclingPage,
)
from ....multigrain_v3.benchmark.document_docling.tableformer_v1_batch import (
    DoclingTableFormerV1BatchCore,
)
from ....multigrain_v3.benchmark.document_docling.tableformer_v2_batch import (
    DoclingTableFormerV2BatchCore,
)
from ... import F, Executor, Pipeline, Port, RayModule, RecoveryPolicy
from ... import RunResult


class DoclingTableJobV36Pipeline(Pipeline):
    """Page→TableJob→Page→Document with cross-document batching."""

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
        layout_num_gpus: float = 0.0,
        ocr_replicas: int = 1,
        ocr_batch_size: int = 4,
        table_replicas: int = 1,
        table_batch_size: int = 4,
        table_core_batch_size: int | None = None,
        table_num_gpus: float = 0.0,
        reduce_replicas: int = 1,
        parse_replicas: int = 1,
        parse_batch_size: int = 1,
        image_scales: tuple[float, ...] = (1.0, 2.0, 3.0),
        batch_scope: str = "elastic",
        device: str = "cpu",
        layout_device: str | None = None,
        ocr_device: str | None = None,
        table_device: str | None = None,
        num_threads: int = 4,
        actor_num_cpus: float = 1.0,
        table_batch_mode: str = "reference",
        table_batch_max_jobs: int = 16,
        ocr_batch_mode: str = "reference",
        ocr_recognition_batch_size: int = 6,
        postprocess_replicas: int | None = None,
        table_prepare_replicas: int | None = None,
        page_reduce_replicas: int | None = None,
        infra_retries: int = 1,
    ) -> None:
        """Freeze only execution options that v3.6 actually implements."""

        if batch_scope not in {"elastic", "parent_bound"}:
            raise ValueError("batch_scope must be elastic or parent_bound")
        if ocr_batch_mode not in {
            "reference",
            "recognition_shadow",
            "recognition_accelerated",
        }:
            raise ValueError("unsupported ocr_batch_mode")
        if ocr_recognition_batch_size <= 0:
            raise ValueError("ocr_recognition_batch_size must be positive")
        if table_batch_mode not in self.table_batch_modes:
            raise ValueError("unsupported table_batch_mode")
        if table_batch_max_jobs <= 0:
            raise ValueError("table_batch_max_jobs must be positive")
        if infra_retries < 0:
            raise ValueError("infra_retries must be non-negative")
        recovery = RecoveryPolicy.abort(infra_retries=infra_retries)

        counts = (
            layout_replicas,
            ocr_replicas,
            table_replicas,
            reduce_replicas,
            parse_replicas,
        )
        if min(counts) <= 0:
            raise ValueError("all replica counts must be positive")
        batches = (
            layout_batch_size,
            ocr_batch_size,
            table_batch_size,
            parse_batch_size,
        )
        if min(batches) <= 0:
            raise ValueError("all batch sizes must be positive")
        if actor_num_cpus <= 0:
            raise ValueError("actor_num_cpus must be positive")
        if num_threads <= 0:
            raise ValueError("num_threads must be positive")

        table_core_batch_size = table_core_batch_size or table_batch_size
        if table_core_batch_size <= 0:
            raise ValueError("table_core_batch_size must be positive")

        layout_device = layout_device or device
        ocr_device = ocr_device or device
        table_device = table_device or device
        postprocess_replicas = postprocess_replicas or reduce_replicas
        table_prepare_replicas = table_prepare_replicas or reduce_replicas
        page_reduce_replicas = page_reduce_replicas or reduce_replicas
        if min(
            postprocess_replicas,
            table_prepare_replicas,
            page_reduce_replicas,
        ) <= 0:
            raise ValueError("all structural stage replica counts must be positive")

        self.parse = (
            RayModule(ExpandDoclingPages)
            .pre_init(image_scales=image_scales)
            .ray_options(
                replicas=parse_replicas,
                batch_size=parse_batch_size,
                num_cpus=1,
                recovery=recovery,
            )
        )
        self.layout = (
            RayModule(DoclingLayoutPages)
            .pre_init(device=layout_device, num_threads=num_threads)
            .ray_options(
                replicas=layout_replicas,
                batch_size=layout_batch_size,
                batch_scope=batch_scope,
                num_cpus=actor_num_cpus,
                num_gpus=layout_num_gpus,
                recovery=recovery,
            )
        )
        self.ocr = (
            RayModule(DoclingOcrPages)
            .pre_init(
                device=ocr_device,
                num_threads=num_threads,
                ocr_batch_mode=ocr_batch_mode,
                ocr_recognition_batch_size=ocr_recognition_batch_size,
            )
            .ray_options(
                replicas=ocr_replicas,
                batch_size=ocr_batch_size,
                batch_scope=batch_scope,
                num_cpus=actor_num_cpus,
                recovery=recovery,
            )
        )
        self.postprocess = RayModule(DoclingPostprocessPages).ray_options(
            replicas=postprocess_replicas,
            batch_size=table_batch_size,
            batch_scope=batch_scope,
            num_cpus=actor_num_cpus,
            recovery=recovery,
        )
        self.table_prepare = RayModule(self.table_job_stage).ray_options(
            replicas=table_prepare_replicas,
            batch_size=table_batch_size,
            batch_scope=batch_scope,
            num_cpus=actor_num_cpus,
            recovery=recovery,
        )
        self.table_core = (
            RayModule(self.table_core_stage)
            .pre_init(
                device=table_device,
                num_threads=num_threads,
                table_batch_mode=table_batch_mode,
                table_batch_max_jobs=table_batch_max_jobs,
            )
            .ray_options(
                replicas=table_replicas,
                batch_size=table_core_batch_size,
                batch_scope=batch_scope,
                num_cpus=actor_num_cpus,
                num_gpus=table_num_gpus,
                recovery=recovery,
            )
        )
        self.page_assemble = RayModule(ReduceDoclingPage).ray_options(
            replicas=page_reduce_replicas,
            batch_size=table_batch_size,
            num_cpus=1,
            recovery=recovery,
        )
        self.document_assemble = RayModule(ReduceDoclingDocument).ray_options(
            replicas=reduce_replicas,
            batch_size=2,
            num_cpus=1,
            recovery=recovery,
        )

    def forward(self, documents: Port) -> Port:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Declare both dynamic fan-outs and their ordered reductions."""

        pages = F.expand(cast(Port, self.parse(documents)))
        layouts = cast(Port, self.layout(pages))
        ocr_results = cast(Port, self.ocr(pages, layouts))
        postprocessed = cast(
            Port,
            self.postprocess(pages, layouts, ocr_results),
        )

        table_jobs = F.expand(cast(Port, self.table_prepare(postprocessed)))
        tables = cast(Port, self.table_core(table_jobs))
        table_groups = F.reduce(tables)
        assembled_pages = cast(
            Port,
            self.page_assemble(table_groups, postprocessed),
        )

        page_groups = F.reduce(assembled_pages)
        return cast(Port, self.document_assemble(page_groups))


class DoclingTableFormerV1BatchV36Pipeline(DoclingTableJobV36Pipeline):
    """The promoted mutation-free TableFormer V1 batch kernel on v3.6."""

    table_job_stage = ExpandDoclingTableJobs
    table_core_stage = DoclingTableFormerV1BatchCore
    table_batch_modes = {"v1_batch"}

    def __init__(self, **options: Any) -> None:
        options.setdefault("table_batch_mode", "v1_batch")
        super().__init__(**options)


class DoclingTableFormerV2BatchV36Pipeline(DoclingTableJobV36Pipeline):
    """The opt-in TableFormer V2 batch kernel on v3.6."""

    table_job_stage = ExpandDoclingTableV2Jobs
    table_core_stage = DoclingTableFormerV2BatchCore
    table_batch_modes = {"v2_batch"}

    def __init__(self, **options: Any) -> None:
        options.setdefault("table_batch_mode", "v2_batch")
        super().__init__(**options)


def run_v36(
    paths: list[str],
    *,
    microbatch_size: int = 24,
    max_active_microbatches: int = 4,
    **pipeline_options: Any,
) -> RunResult:
    """Run Docling on an already initialized Ray runtime or a local default."""

    mode = pipeline_options.get("table_batch_mode")
    if mode == "v1_batch":
        pipeline_type = DoclingTableFormerV1BatchV36Pipeline
    elif mode == "v2_batch":
        pipeline_type = DoclingTableFormerV2BatchV36Pipeline
    else:
        pipeline_type = DoclingTableJobV36Pipeline
    with Executor(pipeline_type(**pipeline_options)) as executor:
        return executor.run(
            paths,
            microbatch_size=microbatch_size,
            max_active_microbatches=max_active_microbatches,
        )


__all__ = [
    "DoclingTableFormerV1BatchV36Pipeline",
    "DoclingTableFormerV2BatchV36Pipeline",
    "DoclingTableJobV36Pipeline",
    "run_v36",
]
