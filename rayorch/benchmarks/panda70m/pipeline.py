from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from rayorch import F, Pipeline, Port, RayModule

from .udfs import (
    DecodePandaTeacherFrames,
    ExpandPandaClips,
    PandaFusedTeacher,
    SelectPandaCaption,
    SummarizePandaSource,
    TEACHERS,
)


_STAGES = ("expand", "decode", "teacher", "select", "summarize")


class Panda70MPipeline(Pipeline):
    def __init__(
        self,
        *,
        output_dir: str,
        model: str,
        teacher_batch_size: int = 8,
        teacher_replicas: int = 1,
        gpu_memory_utilization: float = 0.9,
        decode_replicas: int = 2,
        decode_backend: str = "opencv",
        long_edge: int = 448,
        stage_options: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        if teacher_batch_size <= 0:
            raise ValueError("teacher_batch_size must be positive")
        if teacher_replicas <= 0:
            raise ValueError("teacher_replicas must be positive")
        if decode_replicas <= 0:
            raise ValueError("decode_replicas must be positive")
        if decode_backend not in {"opencv", "pyav"}:
            raise ValueError("decode backend must be opencv or pyav")

        options = normalize_stage_options(stage_options)
        self.expand = RayModule(ExpandPandaClips).ray_options(
            **_merge(
                {"replicas": 1, "batch_size": 2, "num_cpus": 1},
                options.get("expand", {}),
            )
        )
        self.decode = (
            RayModule(DecodePandaTeacherFrames)
            .pre_init(long_edge=long_edge, backend=decode_backend)
            .ray_options(
                **_merge(
                    {
                        "replicas": decode_replicas,
                        "batch_size": 1,
                        "num_cpus": 2,
                    },
                    options.get("decode", {}),
                )
            )
        )
        self.teacher = (
            RayModule(PandaFusedTeacher)
            .pre_init(
                model=model,
                gpu_memory_utilization=gpu_memory_utilization,
            )
            .ray_options(
                **_merge(
                    {
                        "replicas": len(TEACHERS) * teacher_replicas,
                        "batch_size": teacher_batch_size,
                        "num_gpus": 1.0,
                        "num_cpus": 1,
                    },
                    options.get("teacher", {}),
                )
            )
        )
        self.select = RayModule(SelectPandaCaption).ray_options(
            **_merge(
                {"replicas": 1, "batch_size": teacher_batch_size, "num_cpus": 1},
                options.get("select", {}),
            )
        )
        self.summarize = (
            RayModule(SummarizePandaSource)
            .pre_init(output_dir=output_dir)
            .ray_options(
                **_merge(
                    {"replicas": 1, "batch_size": 2, "num_cpus": 1},
                    options.get("summarize", {}),
                )
            )
        )

    def forward(self, sources: Port) -> Port:
        clip_groups = cast(Port, self.expand(sources))
        decoded_groups = cast(Port, self.decode(sources, clip_groups))
        decoded = F.expand(decoded_groups)
        candidates = cast(Port, self.teacher(decoded))
        selected = cast(Port, self.select(decoded, candidates))
        selection_groups = F.reduce(selected)
        return cast(Port, self.summarize(sources, selection_groups))


def normalize_stage_options(
    value: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("stage_options must be a mapping")
    unknown = sorted(set(value) - set(_STAGES))
    if unknown:
        raise ValueError(
            f"unknown Panda-70M stage {unknown[0]!r}; "
            f"expected one of {', '.join(_STAGES)}"
        )
    normalized: dict[str, dict[str, Any]] = {}
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


__all__ = ["Panda70MPipeline", "normalize_stage_options"]
