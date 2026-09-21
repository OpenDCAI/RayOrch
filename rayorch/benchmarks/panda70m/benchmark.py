from __future__ import annotations

import json
import os
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

from .pipeline import Panda70MPipeline, normalize_stage_options
from .udfs import DEFAULT_MODEL


@dataclass(frozen=True, slots=True)
class Panda70MBench:
    manifest: str | Path
    output_dir: str | Path
    model: str = DEFAULT_MODEL
    input_limit: int | None = None
    sample_multiplier: int = 1
    teacher_batch_size: int = 8
    teacher_replicas: int = 1
    decode_replicas: int = 2
    decode_backend: str = "opencv"
    long_edge: int = 448
    input_batch_size: int = 2
    max_active_input_batches: int = 2
    gpu_memory_utilization: float = 0.9
    stage_options: Mapping[str, Mapping[str, Any]] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "manifest",
            str(Path(self.manifest).expanduser().resolve()),
        )
        object.__setattr__(
            self,
            "output_dir",
            str(Path(self.output_dir).expanduser().resolve()),
        )
        for name in (
            "teacher_batch_size",
            "teacher_replicas",
            "decode_replicas",
            "long_edge",
            "input_batch_size",
            "max_active_input_batches",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.input_limit is not None and (
            type(self.input_limit) is not int or self.input_limit <= 0
        ):
            raise ValueError("input_limit must be a positive integer or None")
        if type(self.sample_multiplier) is not int or self.sample_multiplier <= 0:
            raise ValueError("sample_multiplier must be a positive integer")
        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if self.decode_backend not in {"opencv", "pyav"}:
            raise ValueError("decode_backend must be opencv or pyav")
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
        sources = load_panda_sources(
            Path(self.manifest),
            input_limit=self.input_limit,
            sample_multiplier=self.sample_multiplier,
        )
        return run_benchmark(
            name="panda70m",
            pipeline=self._pipeline(),
            source_columns=(sources,),
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
            benchmark="panda70m",
            config=benchmark_config(self),
            artifact_root=Path(self.output_dir) / ".rayorch-benchmark",
            target=target,
            source=source,
            profile=profile,
            profile_interval_s=profile_interval_s,
            run_id=run_id,
        )

    def _pipeline(self) -> Panda70MPipeline:
        return Panda70MPipeline(
            output_dir=str(self.output_dir),
            model=self.model,
            teacher_batch_size=self.teacher_batch_size,
            teacher_replicas=self.teacher_replicas,
            gpu_memory_utilization=self.gpu_memory_utilization,
            decode_replicas=self.decode_replicas,
            decode_backend=self.decode_backend,
            long_edge=self.long_edge,
            stage_options=self.stage_options,
        )


def load_panda_sources(
    manifest: Path,
    *,
    input_limit: int | None = None,
    sample_multiplier: int = 1,
) -> list[dict[str, Any]]:
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest does not exist: {manifest}")
    if sample_multiplier <= 0:
        raise ValueError("sample_multiplier must be positive")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    sources = list(payload.get("sources", ()))
    if input_limit is not None:
        sources = sources[:input_limit]
    if not sources:
        raise ValueError(f"manifest has no sources: {manifest}")

    normalized = []
    for source_order, source in enumerate(sources):
        source_id = str(source.get("source_id", ""))
        clips = sorted(
            source.get("clips", ()),
            key=lambda item: int(item["clip_index"]),
        )
        if not source_id or not clips:
            raise ValueError(f"source at index {source_order} has no complete clips")
        source_clips = []
        for clip in clips:
            path = str(clip.get("path", ""))
            if not path:
                raise ValueError(f"source {source_id} contains a clip without path")
            if not path.startswith("hdfs://") and os.environ.get(
                "RAYORCH_DEFER_LOCAL_PATH_CHECK", ""
            ).lower() not in {"1", "true", "yes"}:
                if not Path(path).is_file():
                    raise FileNotFoundError(path)
            if not clip.get("decodable", True):
                raise ValueError(f"manifest marks clip undecodable: {path}")
            source_clips.append({**clip, "source_id": source_id, "path": path})
        normalized.append(
            {
                "source_id": source_id,
                "source_order": source_order,
                "clips": source_clips,
            }
        )

    if sample_multiplier == 1:
        return normalized
    sampled = []
    for sample_index in range(sample_multiplier):
        for source in normalized:
            sampled_source_id = f"{source['source_id']}__sample_{sample_index:03d}"
            sampled.append(
                {
                    **source,
                    "source_id": sampled_source_id,
                    "source_order": len(sampled),
                    "sample_index": sample_index,
                    "original_source_id": source["source_id"],
                    "clips": [
                        {**clip, "source_id": sampled_source_id}
                        for clip in source["clips"]
                    ],
                }
            )
    return sampled


def _metrics(result: RunResult) -> dict[str, float | int]:
    outputs = list(result.outputs)
    selections = [item for output in outputs for item in output["selections"]]
    return {
        "sources": len(outputs),
        "clips": len(selections),
        "mean_chosen_reference_f1": _mean(
            item["chosen_reference_f1"] for item in selections
        ),
        "mean_oracle_reference_f1": _mean(
            item["oracle_reference_f1"] for item in selections
        ),
    }


def _mean(values: Any) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


__all__ = ["Panda70MBench", "load_panda_sources"]
