"""Declarative Model A -> refinement prompt -> Model B Pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from rayorch import Pipeline, Port, RayModule

from .udfs import BuildRefinementPrompts, BuildResults, VllmGenerate


_STAGES = ("model_a", "prompt", "model_b", "result")


class DualVllmPipeline(Pipeline):
    def __init__(
        self,
        *,
        model_a: str,
        model_b: str,
        tensor_parallel_size: int = 1,
        batch_size: int = 8,
        gpu_memory_utilization: float = 0.9,
        max_tokens: int = 128,
        temperature: float = 0.0,
        stage_options: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        options = normalize_stage_options(stage_options)
        model_options = {
            "replicas": 1,
            "batch_size": batch_size,
            "num_cpus": 1,
            "num_gpus": float(tensor_parallel_size),
        }
        self.model_a = (
            RayModule(VllmGenerate)
            .pre_init(
                model=model_a,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            .ray_options(**_merge(model_options, options.get("model_a", {})))
        )
        self.prompt = RayModule(BuildRefinementPrompts).ray_options(
            **_merge(
                {"replicas": 1, "batch_size": batch_size, "num_cpus": 1},
                options.get("prompt", {}),
            )
        )
        self.model_b = (
            RayModule(VllmGenerate)
            .pre_init(
                model=model_b,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            .ray_options(**_merge(model_options, options.get("model_b", {})))
        )
        self.result = RayModule(BuildResults).ray_options(
            **_merge(
                {"replicas": 1, "batch_size": batch_size, "num_cpus": 1},
                options.get("result", {}),
            )
        )

    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        prompts: Port,
    ) -> Port:
        first_answers = cast(Port, self.model_a(prompts))
        refinement_prompts = cast(Port, self.prompt(first_answers))
        final_answers = cast(Port, self.model_b(refinement_prompts))
        return cast(Port, self.result(prompts, first_answers, final_answers))


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
            f"unknown dual-vLLM stage {unknown[0]!r}; "
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


__all__ = ["DualVllmPipeline"]
