"""Declarative YOLO -> SAM image Pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from rayorch import Pipeline, Port, RayModule

from .udfs import (
    DetectObjects,
    LoadImages,
    RenderMasks,
    SaveImages,
    SaveMetadata,
    SegmentImages,
)


_STAGES = ("load", "yolo", "sam", "render", "save", "metadata")


class YoloSamPipeline(Pipeline):
    def __init__(
        self,
        *,
        output_dir: str,
        yolo_model: str,
        sam_checkpoint: str,
        device: str = "cuda",
        yolo_replicas: int = 1,
        sam_replicas: int = 1,
        batch_size: int = 4,
        confidence: float = 0.25,
        sam_model_type: str = "vit_b",
        overlay_alpha: float = 0.35,
        stage_options: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        options = _normalize_options(stage_options)
        gpu = 1.0 if device.startswith("cuda") else 0.0
        self.load = RayModule(LoadImages, num_outputs=2).ray_options(
            **_merge(
                {"replicas": 1, "batch_size": batch_size, "num_cpus": 1},
                options.get("load", {}),
            )
        )
        self.yolo = (
            RayModule(DetectObjects, num_outputs=2)
            .pre_init(
                model=yolo_model,
                device=device,
                confidence=confidence,
            )
            .ray_options(
                **_merge(
                    {
                        "replicas": yolo_replicas,
                        "batch_size": batch_size,
                        "num_cpus": 1,
                        "num_gpus": gpu,
                    },
                    options.get("yolo", {}),
                )
            )
        )
        self.sam = (
            RayModule(SegmentImages, num_outputs=2)
            .pre_init(
                checkpoint=sam_checkpoint,
                model_type=sam_model_type,
                device=device,
            )
            .ray_options(
                **_merge(
                    {
                        "replicas": sam_replicas,
                        "batch_size": batch_size,
                        "num_cpus": 1,
                        "num_gpus": gpu,
                    },
                    options.get("sam", {}),
                )
            )
        )
        self.render = (
            RayModule(RenderMasks, num_outputs=2)
            .pre_init(alpha=overlay_alpha)
            .ray_options(
                **_merge(
                    {"replicas": 1, "batch_size": batch_size, "num_cpus": 1},
                    options.get("render", {}),
                )
            )
        )
        self.save = (
            RayModule(SaveImages)
            .pre_init(output_dir=output_dir)
            .ray_options(
                **_merge(
                    {"replicas": 1, "batch_size": batch_size, "num_cpus": 1},
                    options.get("save", {}),
                )
            )
        )
        self.metadata = (
            RayModule(SaveMetadata)
            .pre_init(output_dir=output_dir)
            .ray_options(
                **_merge(
                    {"replicas": 1, "batch_size": batch_size, "num_cpus": 1},
                    options.get("metadata", {}),
                )
            )
        )

    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        paths: Port,
    ) -> Port:
        images, metadata = self.load(paths)
        detected, metadata = self.yolo(images, metadata)
        masks, metadata = self.sam(detected, metadata)
        overlays, metadata = self.render(detected, masks, metadata)
        saved = cast(Port, self.save(overlays, metadata))
        return cast(Port, self.metadata(saved))


def _normalize_options(
    value: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("stage_options must be a mapping")
    unknown = sorted(set(value) - set(_STAGES))
    if unknown:
        raise ValueError(
            f"unknown YOLO-SAM stage {unknown[0]!r}; "
            f"expected one of {', '.join(_STAGES)}"
        )
    normalized = {}
    for stage, overrides in value.items():
        if not isinstance(overrides, Mapping):
            raise TypeError(f"stage_options[{stage!r}] must be a mapping")
        if "num_outputs" in overrides:
            raise ValueError("stage_options cannot override num_outputs")
        normalized[stage] = dict(overrides)
    return normalized


def _merge(
    defaults: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    return {**defaults, **overrides}


__all__ = ["YoloSamPipeline"]
