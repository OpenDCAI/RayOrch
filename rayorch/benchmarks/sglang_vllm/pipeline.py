"""Declarative SGLang -> handoff -> vLLM Pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from rayorch import Pipeline, Port, RayModule

from .udfs import (
    BuildHandoffPrompts,
    BuildResults,
    SglangGenerate,
    VllmGenerate,
)


_STAGES = ("sglang", "handoff", "vllm", "result")


class SglangVllmPipeline(Pipeline):
    def __init__(
        self,
        *,
        model: str,
        sglang_env: str,
        vllm_env: str,
        sglang_tensor_parallel_size: int = 1,
        vllm_tensor_parallel_size: int = 1,
        batch_size: int = 8,
        gpu_memory_utilization: float = 0.9,
        max_tokens: int = 128,
        temperature: float = 0.0,
        stage_options: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        options = normalize_stage_options(stage_options)
        common_init = {
            "model": model,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        self.sglang = (
            RayModule(SglangGenerate)
            .pre_init(
                **common_init,
                tensor_parallel_size=sglang_tensor_parallel_size,
            )
            .ray_options(
                **_merge(
                    {
                        "replicas": 1,
                        "batch_size": batch_size,
                        "num_cpus": 1,
                        "num_gpus": float(sglang_tensor_parallel_size),
                        "runtime_env": {"conda": sglang_env},
                    },
                    options.get("sglang", {}),
                )
            )
        )
        self.handoff = RayModule(BuildHandoffPrompts).ray_options(
            **_merge(
                {"replicas": 1, "batch_size": batch_size, "num_cpus": 1},
                options.get("handoff", {}),
            )
        )
        self.vllm = (
            RayModule(VllmGenerate)
            .pre_init(
                **common_init,
                tensor_parallel_size=vllm_tensor_parallel_size,
            )
            .ray_options(
                **_merge(
                    {
                        "replicas": 1,
                        "batch_size": batch_size,
                        "num_cpus": 1,
                        "num_gpus": float(vllm_tensor_parallel_size),
                        "runtime_env": {"conda": vllm_env},
                    },
                    options.get("vllm", {}),
                )
            )
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
        drafts = cast(Port, self.sglang(prompts))
        handoff_prompts = cast(Port, self.handoff(drafts))
        final_answers = cast(Port, self.vllm(handoff_prompts))
        return cast(Port, self.result(prompts, drafts, final_answers))


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
            f"unknown SGLang-vLLM stage {unknown[0]!r}; "
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


__all__ = ["SglangVllmPipeline"]
