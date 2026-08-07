"""Video→Frame→Feature→Video on Multigrain v3.6.

The decode, transform and summary UDFs are the unchanged V3 workload.  Only
the authoring/compiler/runtime layer changes, which keeps paired comparisons
focused on framework behavior.
"""

from __future__ import annotations

from typing import Any, cast

from ....multigrain_v3.benchmark.video.v3 import (
    DecodeFrames,
    SummarizeVideos,
    TransformFrames,
)
from ... import F, Executor, Pipeline, Port, RayModule, RecoveryPolicy
from ...executor import RunResult


class VideoV36Pipeline(Pipeline):
    """One dynamic frame relation followed by an ordered parent reduction."""

    def __init__(
        self,
        *,
        stride: int,
        max_frames: int | None,
        decode_replicas: int,
        reduce_replicas: int,
        transform_replicas: int,
        transform_batch_size: int,
        batch_scope: str,
        transform_backend: str = "opencv",
        torch_num_threads: int = 1,
        model_path: str | None = None,
        transform_num_gpus: float = 0.0,
        model_repeats: int = 1,
        infra_retries: int = 1,
    ) -> None:
        if stride <= 0:
            raise ValueError("stride must be positive")
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be positive when provided")
        if batch_scope not in {"elastic", "parent_bound"}:
            raise ValueError("batch_scope must be elastic or parent_bound")
        if min(decode_replicas, reduce_replicas, transform_replicas) <= 0:
            raise ValueError("all replica counts must be positive")
        if transform_batch_size <= 0:
            raise ValueError("transform_batch_size must be positive")
        if torch_num_threads <= 0:
            raise ValueError("torch_num_threads must be positive")
        if model_repeats <= 0:
            raise ValueError("model_repeats must be positive")
        if transform_num_gpus < 0:
            raise ValueError("transform_num_gpus must be non-negative")
        if infra_retries < 0:
            raise ValueError("infra_retries must be non-negative")
        recovery = RecoveryPolicy.abort(infra_retries=infra_retries)

        self.decode = (
            RayModule(DecodeFrames)
            .pre_init(stride=stride, max_frames=max_frames)
            .ray_options(
                replicas=decode_replicas,
                batch_size=1,
                num_cpus=1,
                recovery=recovery,
            )
        )
        self.transform = (
            RayModule(TransformFrames)
            .pre_init(
                backend=transform_backend,
                torch_num_threads=torch_num_threads,
                model_path=model_path,
                model_repeats=model_repeats,
            )
            .ray_options(
                replicas=transform_replicas,
                batch_size=transform_batch_size,
                batch_scope=batch_scope,
                num_cpus=max(1, torch_num_threads),
                num_gpus=transform_num_gpus,
                recovery=recovery,
            )
        )
        self.summary = RayModule(SummarizeVideos).ray_options(
            replicas=reduce_replicas,
            batch_size=4,
            num_cpus=1,
            recovery=recovery,
        )

    def forward(self, videos: Port) -> Port:  # pyright: ignore[reportIncompatibleMethodOverride]
        frames = F.expand(cast(Port, self.decode(videos)))
        features = cast(Port, self.transform(frames))
        groups = F.reduce(features)
        return cast(Port, self.summary(groups))


def run_v36(
    paths: list[str],
    *,
    arena_size: int = 2,
    max_in_flight: int = 2,
    **pipeline_options: Any,
) -> RunResult:
    """Run the v3.6 feature pipeline on an initialized or local Ray runtime."""

    with Executor(VideoV36Pipeline(**pipeline_options)) as executor:
        return executor.run(
            paths,
            arena_size=arena_size,
            max_in_flight=max_in_flight,
        )


__all__ = ["VideoV36Pipeline", "run_v36"]
