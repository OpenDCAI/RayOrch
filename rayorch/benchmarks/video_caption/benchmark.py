"""Public Benchmark for the video-caption reference workload."""

from __future__ import annotations

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

from .pipeline import VideoCaptionTopologyPipeline


@dataclass(frozen=True, slots=True)
class VideoCaptionTopologyBench:
    output_dir: str | Path = "."
    video_count: int = 3
    frames_per_video: int = 4
    workers: int = 2
    batch_size: int = 4
    input_batch_size: int = 1
    max_active_input_batches: int = 3

    def __post_init__(self) -> None:
        for name in (
            "video_count",
            "frames_per_video",
            "workers",
            "batch_size",
            "input_batch_size",
            "max_active_input_batches",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        object.__setattr__(self, "output_dir", str(Path(self.output_dir).resolve()))

    def run(
        self,
        *,
        ray_address: str | None = None,
        ray_init_kwargs: Mapping[str, Any] | None = None,
        profile: bool = True,
        profile_interval_s: float = 1.0,
        run_id: str | None = None,
    ) -> BenchmarkReport:
        videos = tuple(
            (f"video-{index}", self.frames_per_video)
            for index in range(self.video_count)
        )
        return run_benchmark(
            name="video_caption_topology",
            pipeline=VideoCaptionTopologyPipeline(
                workers=self.workers,
                batch_size=self.batch_size,
            ),
            source_columns=(videos,),
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
            benchmark="video_caption_topology",
            config=benchmark_config(self),
            artifact_root=Path(self.output_dir) / ".rayorch-benchmark",
            target=target,
            source=source,
            profile=profile,
            profile_interval_s=profile_interval_s,
            run_id=run_id,
        )


def _metrics(result: RunResult) -> dict[str, int]:
    return {"frames": sum(len(output["frames"]) for output in result.outputs)}


__all__ = ["VideoCaptionTopologyBench"]
