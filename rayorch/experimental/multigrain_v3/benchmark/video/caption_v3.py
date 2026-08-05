"""SmolVLM frame-caption 的 V3 video pipeline。"""

from __future__ import annotations

from typing import Any

from ...api import Expand, Map, Pipeline, Reduce
from ...executor import Executor, RunResult
from .v3 import DecodeFrames
from .workload import FrameCaptioner, summarize_captions


class CaptionFrames:
    """GPU Map UDF：对跨视频 frame batch 生成 captions。"""

    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 12,
    ) -> None:
        """加载一个 persistent deterministic captioner。"""

        self.captioner = FrameCaptioner(
            model_path,
            max_new_tokens=max_new_tokens,
        )

    def run(self, frames):
        """运行 greedy batched caption。"""

        return self.captioner.caption(frames)


class SummarizeCaptionVideos:
    """Reduce UDF：按 frame ordinal 恢复 captions。"""

    def run(self, groups):
        """逐视频返回 caption summary。"""

        return [summarize_captions(group) for group in groups]


class VideoCaptionV3Pipeline(Pipeline):
    """视频→帧→SmolVLM captions→视频的 V3 Pipeline。"""

    def __init__(
        self,
        *,
        model_path: str,
        stride: int,
        max_frames: int | None,
        batch_scope: str,
        caption_batch_size: int = 16,
        caption_replicas: int = 1,
        decode_replicas: int = 4,
        reduce_replicas: int = 1,
        max_new_tokens: int = 12,
    ) -> None:
        """配置 decode、caption GPU actors 和 ordered Reduce。"""

        self.decode = (
            Expand(DecodeFrames)
            .pre_init(stride=stride, max_frames=max_frames)
            .ray_options(
                replicas=decode_replicas,
                batch_size=1,
                num_cpus=1,
            )
        )
        self.caption = (
            Map(CaptionFrames)
            .pre_init(
                model_path=model_path,
                max_new_tokens=max_new_tokens,
            )
            .ray_options(
                replicas=caption_replicas,
                batch_size=caption_batch_size,
                batch_scope=batch_scope,
                num_cpus=1,
                num_gpus=1,
            )
        )
        self.reduce = Reduce(SummarizeCaptionVideos).ray_options(
            replicas=reduce_replicas,
            batch_size=4,
            num_cpus=1,
        )

    def forward(self, videos):
        """声明 frame caption pipeline。"""

        frames = self.decode(videos)
        captions = self.caption(frames)
        return self.reduce(anchor=videos, members=captions)


def run_v3(
    paths: list[str],
    *,
    microbatch_size: int = 16,
    max_inflight_arenas: int = 1,
    max_pending_per_actor: int = 1,
    actor_max_concurrency: int = 1,
    **pipeline_options: Any,
) -> RunResult:
    """运行 V3 caption pipeline。"""

    return Executor(
        VideoCaptionV3Pipeline(**pipeline_options),
        microbatch_size=microbatch_size,
        max_inflight_arenas=max_inflight_arenas,
        max_pending_per_actor=max_pending_per_actor,
        actor_max_concurrency=actor_max_concurrency,
    ).run(paths)
