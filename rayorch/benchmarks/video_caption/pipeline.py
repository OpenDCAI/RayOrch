"""Video -> Frame -> Caption -> Video reference Pipeline."""

from __future__ import annotations

from typing import cast

from rayorch import F, Pipeline, Port, RayModule

from .udfs import CaptionFrames, DecodeFrames, SummarizeCaptions


class VideoCaptionTopologyPipeline(Pipeline):
    def __init__(
        self,
        *,
        workers: int = 2,
        batch_size: int = 4,
    ) -> None:
        self.decode = RayModule(DecodeFrames).ray_options(
            replicas=workers,
            batch_size=1,
            num_cpus=0,
        )
        self.caption = RayModule(CaptionFrames).ray_options(
            replicas=workers,
            batch_size=batch_size,
            num_cpus=0,
        )
        self.summary = RayModule(SummarizeCaptions).ray_options(
            replicas=1,
            batch_size=2,
            num_cpus=0,
        )

    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        videos: Port,
    ) -> Port:
        frames = F.expand(cast(Port, self.decode(videos)))
        captions = cast(Port, self.caption(frames))
        return cast(Port, self.summary(F.reduce(captions)))


__all__ = ["VideoCaptionTopologyPipeline"]
