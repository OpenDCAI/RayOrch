"""SmolVLM frame-caption workload on Multigrain v3.6."""

from __future__ import annotations

from typing import Any, cast

from ....multigrain_v3.benchmark.video.caption_v3 import (
    CaptionFrames,
    SummarizeCaptionVideos,
)
from ....multigrain_v3.benchmark.video.v3 import DecodeFrames
from ... import F, Executor, Pipeline, Port, RayModule, RecoveryPolicy
from ... import RunResult


class VideoCaptionV36Pipeline(Pipeline):
    """Video→Frame→SmolVLM caption→Video with an explicit frame Domain."""

    def __init__(
        self,
        *,
        model_path: str,
        stride: int,
        max_frames: int | None,
        batching_policy: str,
        caption_batch_size: int = 16,
        caption_replicas: int = 1,
        decode_replicas: int = 4,
        reduce_replicas: int = 1,
        max_new_tokens: int = 12,
        infra_retries: int = 1,
    ) -> None:
        if not model_path:
            raise ValueError("model_path must be non-empty")
        if stride <= 0:
            raise ValueError("stride must be positive")
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be positive when provided")
        if batching_policy not in {"any_parent", "single_parent"}:
            raise ValueError("batching_policy must be any_parent or single_parent")
        if min(caption_replicas, decode_replicas, reduce_replicas) <= 0:
            raise ValueError("all replica counts must be positive")
        if caption_batch_size <= 0:
            raise ValueError("caption_batch_size must be positive")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
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
        self.caption = (
            RayModule(CaptionFrames)
            .pre_init(
                model_path=model_path,
                max_new_tokens=max_new_tokens,
            )
            .ray_options(
                replicas=caption_replicas,
                batch_size=caption_batch_size,
                batching_policy=batching_policy,
                num_cpus=1,
                num_gpus=1,
                recovery=recovery,
            )
        )
        self.summary = RayModule(SummarizeCaptionVideos).ray_options(
            replicas=reduce_replicas,
            batch_size=4,
            num_cpus=1,
            recovery=recovery,
        )

    def forward(self, videos: Port) -> Port:  # pyright: ignore[reportIncompatibleMethodOverride]
        frames = F.expand(cast(Port, self.decode(videos)))
        captions = cast(Port, self.caption(frames))
        groups = F.reduce(captions)
        return cast(Port, self.summary(groups))


def run_caption_v36(
    paths: list[str],
    *,
    microbatch_size: int = 16,
    max_active_microbatches: int = 1,
    **pipeline_options: Any,
) -> RunResult:
    """Run the v3.6 caption pipeline."""

    with Executor(VideoCaptionV36Pipeline(**pipeline_options)) as executor:
        return executor.run(
            paths,
            microbatch_size=microbatch_size,
            max_active_microbatches=max_active_microbatches,
        )


__all__ = ["VideoCaptionV36Pipeline", "run_caption_v36"]
