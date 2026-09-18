"""Video -> sibling Audio/Frame domains -> Video reference Pipeline."""

from __future__ import annotations

from typing import cast

from rayorch import F, Pipeline, Port, RayModule

from .udfs import (
    DecodeAudio,
    DecodeFrames,
    MergeModalities,
    ProcessFrames,
    SummarizeAudio,
    SummarizeFrames,
    TranscribeAudio,
)


class VideoMultimodalTopologyPipeline(Pipeline):
    def __init__(
        self,
        *,
        workers: int = 2,
        batch_size: int = 4,
    ) -> None:
        decode = {"replicas": workers, "batch_size": 1, "num_cpus": 0}
        process = {"replicas": workers, "batch_size": batch_size, "num_cpus": 0}
        reduce = {"replicas": 1, "batch_size": 2, "num_cpus": 0}

        self.audio_decode = RayModule(DecodeAudio).ray_options(**decode)
        self.asr = RayModule(TranscribeAudio).ray_options(**process)
        self.audio_summary = RayModule(SummarizeAudio).ray_options(**reduce)
        self.frame_decode = RayModule(DecodeFrames).ray_options(**decode)
        self.vision = RayModule(ProcessFrames).ray_options(**process)
        self.vision_summary = RayModule(SummarizeFrames).ray_options(**reduce)
        self.merge = RayModule(MergeModalities).ray_options(**reduce)

    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        videos: Port,
    ) -> Port:
        audio_chunks = F.expand(cast(Port, self.audio_decode(videos)))
        transcripts = cast(Port, self.asr(audio_chunks))
        audio = cast(Port, self.audio_summary(F.reduce(transcripts)))

        frames = F.expand(cast(Port, self.frame_decode(videos)))
        features = cast(Port, self.vision(frames))
        vision = cast(Port, self.vision_summary(F.reduce(features)))
        return cast(Port, self.merge(audio, vision))


__all__ = ["VideoMultimodalTopologyPipeline"]
