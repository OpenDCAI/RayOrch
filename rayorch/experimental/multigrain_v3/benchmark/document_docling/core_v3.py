"""只复用 Docling 核心模型、由 V3 编排 grain 生命周期的 PDF pipeline。"""

from __future__ import annotations

from typing import Any

from ...api import Expand, Map, Pipeline, Reduce
from ...executor import Executor, RunResult
from .core_stages import (
    DoclingLayoutPages,
    DoclingOcrPages,
    DoclingPostprocessTablesAssemble,
    ExpandDoclingPages,
    ReduceDoclingDocument,
)


def four_gpu_split_stage_options() -> dict[str, Any]:
    """返回 Docling split-stage 的静态四卡实验参数。

    根据四卡 pilot 的 Stage busy/span，Layout 部署一个单卡 actor，
    Table/assemble 部署三个单卡 actor，OCR 保持四个 CPU actor；所有 GPU actor
    都限制为单并发，避免同一模型实例被重入。
    调用方可按实验需要覆盖返回字典中的 batch 或设备参数。
    """

    return {
        "layout_replicas": 1,
        "layout_num_gpus": 1.0,
        "layout_actor_concurrency": 1,
        "ocr_replicas": 4,
        "ocr_actor_concurrency": 1,
        "table_replicas": 3,
        "table_num_gpus": 1.0,
        "table_actor_concurrency": 1,
        "reduce_replicas": 4,
    }


class _LegacyPageBoundDoclingCoreV3Pipeline(Pipeline):
    """旧 page-bound 实现，仅留作代码对照，不再由公共入口使用。"""

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
    ) -> None:
        """配置 Docling preprocessing、模型 actors 和 Reduce。

        四卡 split-stage 实验可使用 :func:`four_gpu_split_stage_options`
        作为参数基线：Layout 1×1 GPU，Table/assemble 3×1 GPU。

        OCR recognition batching 和 TableFormer encoder/decoder batching 都会使用
        actor-local 模型状态，因此要求对应 actor ``max_concurrency=1``。
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
        if (
            ocr_batch_mode != "reference"
            and ocr_actor_concurrency != 1
        ):
            raise ValueError(
                "ocr_actor_concurrency must be 1 for recognition batching"
            )
        if table_batch_mode not in {
            "reference",
            "encoder_shadow",
            "encoder_accelerated",
            "decoder_accelerated",
        }:
            raise ValueError(
                "table_batch_mode must be reference, encoder_shadow, "
                "encoder_accelerated, or decoder_accelerated"
            )
        if table_batch_max_jobs <= 0:
            raise ValueError("table_batch_max_jobs must be positive")
        if (
            table_batch_mode != "reference"
            and table_actor_concurrency != 1
        ):
            raise ValueError(
                "table_actor_concurrency must be 1 for encoder batching"
            )

        layout_device = layout_device or device
        ocr_device = ocr_device or device
        table_device = table_device or device
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
            .pre_init(
                device=layout_device,
                num_threads=num_threads,
            )
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
        self.table_assemble = (
            Map(DoclingPostprocessTablesAssemble)
            .pre_init(
                device=table_device,
                num_threads=num_threads,
                table_batch_mode=table_batch_mode,
                table_batch_max_jobs=table_batch_max_jobs,
            )
            .ray_options(
                replicas=table_replicas,
                batch_size=table_batch_size,
                max_batch_wait_ms=table_batch_wait_ms,
                batch_scope=batch_scope,
                num_cpus=actor_num_cpus,
                num_gpus=table_num_gpus,
                max_concurrency=table_actor_concurrency,
            )
        )
        self.reduce = Reduce(ReduceDoclingDocument).ray_options(
            replicas=reduce_replicas,
            batch_size=2,
            num_cpus=1,
        )

    def forward(self, documents):
        """声明 Docling core-stage page pipeline。"""

        pages = self.expand(documents)
        layouts = self.layout(pages)
        ocr_results = self.ocr(pages, layouts)
        assembled_pages = self.table_assemble(
            pages,
            layouts,
            ocr_results,
        )
        return self.reduce(anchor=documents, members=assembled_pages)

# 公共入口保持兼容，但实际 DAG 已切换为 Page -> TableJob -> Page -> Document。
from .core_v3_table import (
    DoclingTableFormerV1BatchV3Pipeline,
    DoclingTableFormerV2BatchV3Pipeline,
    DoclingTableJobV3Pipeline,
)

DoclingCoreV3Pipeline = DoclingTableJobV3Pipeline


def run_v3(
    paths: list[str],
    *,
    microbatch_size: int = 2,
    max_inflight_arenas: int = 2,
    max_pending_per_actor: int = 1,
    actor_max_concurrency: int = 1,
    **pipeline_options: Any,
) -> RunResult:
    """运行 Docling core-stage V3 pipeline。"""

    pipeline_type = {
        "v1_batch": DoclingTableFormerV1BatchV3Pipeline,
        "v2_batch": DoclingTableFormerV2BatchV3Pipeline,
    }.get(
        pipeline_options.get("table_batch_mode"),
        DoclingCoreV3Pipeline,
    )

    return Executor(
        pipeline_type(**pipeline_options),
        microbatch_size=microbatch_size,
        max_inflight_arenas=max_inflight_arenas,
        max_pending_per_actor=max_pending_per_actor,
        actor_max_concurrency=actor_max_concurrency,
    ).run(paths)
