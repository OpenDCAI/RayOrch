"""Whisper-audio plus ViT-frame dual relation on Multigrain v3.6."""

from __future__ import annotations

from typing import Any, cast

from ....multigrain_v3.benchmark.video.multimodal_v3 import (
    DecodeAudioChunks,
    MergeModalities,
    SummarizeAudio,
    SummarizeFrames,
    WhisperChunks,
)
from ....multigrain_v3.benchmark.video.v3 import DecodeFrames, TransformFrames
from ... import F, Executor, Pipeline, Port, RayModule, RecoveryPolicy
from ...executor import RunResult


class VideoMultimodalV36Pipeline(Pipeline):
    """Two sibling child Domains reduced to their shared Video entity."""

    def __init__(
        self,
        *,
        whisper_model_path: str,
        vit_model_path: str,
        batch_scope: str = "elastic",
        audio_chunk_seconds: float = 4.0,
        audio_decode_replicas: int = 2,
        asr_replicas: int = 1,
        asr_batch_size: int = 8,
        asr_num_gpus: float = 0.5,
        frame_stride: int = 20,
        max_frames: int | None = 8,
        frame_decode_replicas: int = 2,
        frame_model_replicas: int = 1,
        frame_batch_size: int = 16,
        frame_num_gpus: float = 0.5,
        reduce_replicas: int = 1,
        infra_retries: int = 1,
    ) -> None:
        if not whisper_model_path or not vit_model_path:
            raise ValueError("both model paths must be non-empty")
        if batch_scope not in {"elastic", "parent_bound"}:
            raise ValueError("batch_scope must be elastic or parent_bound")
        if audio_chunk_seconds <= 0:
            raise ValueError("audio_chunk_seconds must be positive")
        if frame_stride <= 0:
            raise ValueError("frame_stride must be positive")
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be positive when provided")
        if min(
            audio_decode_replicas,
            asr_replicas,
            frame_decode_replicas,
            frame_model_replicas,
            reduce_replicas,
        ) <= 0:
            raise ValueError("all replica counts must be positive")
        if min(asr_batch_size, frame_batch_size) <= 0:
            raise ValueError("all model batch sizes must be positive")
        if min(asr_num_gpus, frame_num_gpus) < 0:
            raise ValueError("model GPU requirements must be non-negative")
        if infra_retries < 0:
            raise ValueError("infra_retries must be non-negative")
        recovery = RecoveryPolicy.abort(infra_retries=infra_retries)

        self.audio_decode = (
            RayModule(DecodeAudioChunks)
            .pre_init(chunk_seconds=audio_chunk_seconds)
            .ray_options(
                replicas=audio_decode_replicas,
                batch_size=1,
                num_cpus=1,
                recovery=recovery,
            )
        )
        self.asr = (
            RayModule(WhisperChunks)
            .pre_init(whisper_model_path)
            .ray_options(
                replicas=asr_replicas,
                batch_size=asr_batch_size,
                batch_scope=batch_scope,
                num_cpus=1,
                num_gpus=asr_num_gpus,
                recovery=recovery,
            )
        )
        self.audio_summary = RayModule(SummarizeAudio).ray_options(
            replicas=reduce_replicas,
            batch_size=4,
            num_cpus=1,
            recovery=recovery,
        )

        self.frame_decode = (
            RayModule(DecodeFrames)
            .pre_init(stride=frame_stride, max_frames=max_frames)
            .ray_options(
                replicas=frame_decode_replicas,
                batch_size=1,
                num_cpus=1,
                recovery=recovery,
            )
        )
        self.frame_model = (
            RayModule(TransformFrames)
            .pre_init(backend="vit", model_path=vit_model_path)
            .ray_options(
                replicas=frame_model_replicas,
                batch_size=frame_batch_size,
                batch_scope=batch_scope,
                num_cpus=1,
                num_gpus=frame_num_gpus,
                recovery=recovery,
            )
        )
        self.frame_summary = RayModule(SummarizeFrames).ray_options(
            replicas=reduce_replicas,
            batch_size=4,
            num_cpus=1,
            recovery=recovery,
        )
        self.merge = RayModule(MergeModalities).ray_options(
            replicas=reduce_replicas,
            batch_size=8,
            num_cpus=1,
            recovery=recovery,
        )

    def forward(self, videos: Port) -> Port:  # pyright: ignore[reportIncompatibleMethodOverride]
        audio_chunks = F.expand(cast(Port, self.audio_decode(videos)))
        transcripts = cast(Port, self.asr(audio_chunks))
        transcript_groups = F.reduce(transcripts)
        audio_summary = cast(Port, self.audio_summary(transcript_groups))

        frames = F.expand(cast(Port, self.frame_decode(videos)))
        features = cast(Port, self.frame_model(frames))
        feature_groups = F.reduce(features)
        frame_summary = cast(Port, self.frame_summary(feature_groups))

        return cast(Port, self.merge(audio_summary, frame_summary))


def run_multimodal_v36(
    paths: list[str],
    *,
    microbatch_size: int = 4,
    max_active_microbatches: int = 2,
    **pipeline_options: Any,
) -> RunResult:
    """Run the dual-relation v3.6 multimodal pipeline."""

    with Executor(VideoMultimodalV36Pipeline(**pipeline_options)) as executor:
        return executor.run(
            paths,
            microbatch_size=microbatch_size,
            max_active_microbatches=max_active_microbatches,
        )


__all__ = ["VideoMultimodalV36Pipeline", "run_multimodal_v36"]
