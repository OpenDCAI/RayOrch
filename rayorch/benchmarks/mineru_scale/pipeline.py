"""Declarative RayOrch graph for the scale-oriented MinerU workload."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from rayorch import F, Pipeline, Port, RayModule

from .udfs import (
    MinerUScaleAssembleDoc,
    MinerUScalePdfMetadata,
    MinerUScalePdfToPages,
    MinerUScaleVlmOcrPage,
)


_STAGES = ("render", "ocr", "metadata", "assemble")


class MinerUScalePipeline(Pipeline):
    """Model PDF -> page -> OCR -> ordered document reconstruction at scale."""

    def __init__(
        self,
        *,
        output_dir: str,
        model: str,
        render_replicas: int = 256,
        ocr_replicas: int = 128,
        assemble_replicas: int = 64,
        batch_size: int = 64,
        render_dpi: int = 200,
        gpu_memory_utilization: float = 0.32,
        gpus_per_ocr_actor: float = 0.5,
        stage_options: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        options = normalize_stage_options(stage_options)
        self.render = (
            RayModule(MinerUScalePdfToPages)
            .pre_init(dpi=render_dpi)
            .ray_options(
                **_merge_options(
                    {
                        "replicas": render_replicas,
                        "batch_size": 1,
                        "num_cpus": 1,
                    },
                    options.get("render", {}),
                )
            )
        )
        self.ocr = (
            RayModule(MinerUScaleVlmOcrPage)
            .pre_init(
                model=model,
                gpu_memory_utilization=gpu_memory_utilization,
            )
            .ray_options(
                **_merge_options(
                    {
                        "replicas": ocr_replicas,
                        "batch_size": batch_size,
                        "num_gpus": gpus_per_ocr_actor,
                        "num_cpus": 1,
                    },
                    options.get("ocr", {}),
                )
            )
        )
        self.metadata = RayModule(MinerUScalePdfMetadata).ray_options(
            **_merge_options(
                {"replicas": 1, "batch_size": 32, "num_cpus": 1},
                options.get("metadata", {}),
            )
        )
        self.assemble = (
            RayModule(MinerUScaleAssembleDoc)
            .pre_init(output_dir=output_dir)
            .ray_options(
                **_merge_options(
                    {
                        "replicas": assemble_replicas,
                        "batch_size": 4,
                        "num_cpus": 1,
                    },
                    options.get("assemble", {}),
                )
            )
        )

    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        pdfs: Port,
    ):
        pages = F.expand(cast(Port, self.render(pdfs)))
        contents = cast(Port, self.ocr(pages))
        stems = cast(Port, self.metadata(pdfs))
        content_groups, ordered_page_groups = F.reduce_aligned(
            contents,
            pages,
            members=contents,
        )
        return self.assemble(content_groups, ordered_page_groups, stems)


def normalize_stage_options(
    value: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """Validate and copy advanced per-stage Ray actor overrides."""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("stage_options must be a mapping")
    unknown = sorted(set(value) - set(_STAGES))
    if unknown:
        raise ValueError(
            f"unknown MinerU scale stage {unknown[0]!r}; expected one of "
            + ", ".join(_STAGES)
        )
    normalized = {}
    for stage, overrides in value.items():
        if not isinstance(overrides, Mapping):
            raise TypeError(f"stage_options[{stage!r}] must be a mapping")
        if "num_outputs" in overrides:
            raise ValueError("stage_options cannot override num_outputs")
        normalized[stage] = dict(overrides)
    return normalized


def _merge_options(
    defaults: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    return {**defaults, **overrides}


__all__ = ["MinerUScalePipeline"]
