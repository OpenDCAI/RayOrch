"""Small public entrypoint for the SGLang -> vLLM workload."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from rayorch import RunResult
from rayorch.benchmark import (
    BenchmarkReport,
    BenchmarkRun,
    LocalSource,
    benchmark_config,
    run_benchmark,
    submit_benchmark,
)

from .pipeline import SglangVllmPipeline, normalize_stage_options


@dataclass(frozen=True, slots=True)
class SglangVllmBench:
    input_path: str | Path
    output_dir: str | Path
    model: str | Path
    sglang_env: str
    vllm_env: str
    input_limit: int | None = None
    sglang_tensor_parallel_size: int = 1
    vllm_tensor_parallel_size: int = 1
    batch_size: int = 8
    input_batch_size: int = 32
    max_active_input_batches: int = 1
    gpu_memory_utilization: float = 0.9
    max_tokens: int = 128
    temperature: float = 0.0
    stage_options: Mapping[str, Mapping[str, Any]] | None = None

    def __post_init__(self) -> None:
        for name in ("input_path", "output_dir", "model"):
            object.__setattr__(
                self,
                name,
                str(Path(getattr(self, name)).expanduser().resolve()),
            )
        for name in ("sglang_env", "vllm_env"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty Conda environment")
        for name in (
            "sglang_tensor_parallel_size",
            "vllm_tensor_parallel_size",
            "batch_size",
            "input_batch_size",
            "max_active_input_batches",
            "max_tokens",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.input_limit is not None and (
            type(self.input_limit) is not int or self.input_limit <= 0
        ):
            raise ValueError("input_limit must be a positive integer or None")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        object.__setattr__(
            self,
            "stage_options",
            normalize_stage_options(self.stage_options),
        )

    def run(
        self,
        *,
        ray_address: str | None = None,
        ray_init_kwargs: Mapping[str, Any] | None = None,
        profile: bool = True,
        profile_interval_s: float = 1.0,
        run_id: str | None = None,
    ) -> BenchmarkReport:
        prompts = _load_prompts(Path(self.input_path), self.input_limit)
        _validate(self)
        return run_benchmark(
            name="sglang_vllm",
            pipeline=self._pipeline(),
            source_columns=(prompts,),
            config=benchmark_config(self),
            artifact_root=Path(self.output_dir) / ".rayorch-benchmark",
            input_batch_size=self.input_batch_size,
            max_active_input_batches=self.max_active_input_batches,
            ray_address=ray_address,
            ray_init_kwargs=ray_init_kwargs,
            profile=profile,
            profile_interval_s=profile_interval_s,
            run_id=run_id,
            extra_metrics=_metrics,
        )

    def submit(
        self,
        target: str,
        *,
        source: LocalSource | None = None,
        profile: bool = True,
        profile_interval_s: float = 1.0,
        run_id: str | None = None,
    ) -> BenchmarkRun:
        return submit_benchmark(
            benchmark="sglang_vllm",
            config=benchmark_config(self),
            artifact_root=Path(self.output_dir) / ".rayorch-benchmark",
            target=target,
            source=source,
            profile=profile,
            profile_interval_s=profile_interval_s,
            run_id=run_id,
        )

    def _pipeline(self) -> SglangVllmPipeline:
        return SglangVllmPipeline(
            model=str(self.model),
            sglang_env=self.sglang_env,
            vllm_env=self.vllm_env,
            sglang_tensor_parallel_size=self.sglang_tensor_parallel_size,
            vllm_tensor_parallel_size=self.vllm_tensor_parallel_size,
            batch_size=self.batch_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stage_options=self.stage_options,
        )


def _load_prompts(path: Path, limit: int | None) -> tuple[str, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"input_path does not exist: {path}")
    if path.suffix.lower() == ".jsonl":
        prompts = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            value = json.loads(line)
            prompt = value.get("prompt") if isinstance(value, dict) else value
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"invalid prompt at JSONL line {line_number}")
            prompts.append(prompt)
    else:
        prompts = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if limit is not None:
        prompts = prompts[:limit]
    if not prompts:
        raise ValueError(f"no prompts found in {path}")
    return tuple(prompts)


def _validate(benchmark: SglangVllmBench) -> None:
    if not Path(benchmark.model).exists():
        raise FileNotFoundError(f"model does not exist: {benchmark.model}")


def _metrics(result: RunResult) -> dict[str, int]:
    outputs = list(result.outputs)
    return {
        "prompts": len(outputs),
        "sglang_characters": sum(len(item["sglang"]) for item in outputs),
        "vllm_characters": sum(len(item["vllm"]) for item in outputs),
    }


__all__ = ["SglangVllmBench"]
