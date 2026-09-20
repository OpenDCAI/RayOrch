"""Declarative RayOrch graph for the built-in MinerU workload."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from rayorch import F, Pipeline, Port, RayModule

from .udfs import MinerUAssembleDoc, MinerUPdfToPages, MinerUVlmOcrPage, PdfMetadata


_STAGES = ("render", "ocr", "metadata", "assemble")


class MinerUPipeline(Pipeline):
    """Model the real PDF -> page -> OCR -> document MinerU pipeline."""

    def __init__(
        self,
        *,
        output_dir: str,
        model: str,
        num_gpus: int = 1,
        batch_size: int = 64,
        render_dpi: int = 200,
        gpu_memory_utilization: float = 0.8,
        stage_options: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        """Build the four MinerU stages."""

        options = normalize_stage_options(stage_options)
        self.render = (
            RayModule(MinerUPdfToPages)
            .pre_init(dpi=render_dpi)
            .ray_options(
                **_merge_options(
                    {"replicas": num_gpus, "batch_size": 1, "num_cpus": 1},
                    options.get("render", {}),
                )
            )
        )
        self.ocr = (
            RayModule(MinerUVlmOcrPage)
            .pre_init(
                model=model,
                gpu_memory_utilization=gpu_memory_utilization,
            )
            .ray_options(
                **_merge_options(
                    {
                        "replicas": num_gpus,
                        "batch_size": batch_size,
                        "num_gpus": 1.0,
                        "num_cpus": 1,
                    },
                    options.get("ocr", {}),
                )
            )
        )
        self.metadata = RayModule(PdfMetadata).ray_options(
            **_merge_options(
                {"replicas": 1, "batch_size": 32, "num_cpus": 1},
                options.get("metadata", {}),
            )
        )
        self.assemble = (
            RayModule(MinerUAssembleDoc)
            .pre_init(output_dir=output_dir)
            .ray_options(
                **_merge_options(
                    {"replicas": num_gpus, "batch_size": 4, "num_cpus": 1},
                    options.get("assemble", {}),
                )
            )
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


def normalize_stage_options(
    value: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """Validate and copy per-stage Ray execution overrides."""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("stage_options must be a mapping")
    unknown = sorted(set(value) - set(_STAGES))
    if unknown:
        raise ValueError(
            f"unknown MinerU stage {unknown[0]!r}; expected one of {', '.join(_STAGES)}"
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


__all__ = ["MinerUPipeline"]
