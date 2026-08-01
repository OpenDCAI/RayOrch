"""通过 Multigrain V3 执行 Video→Frames→Feature→Video workload。"""

from __future__ import annotations

from typing import Any

from ...api import Expand, Map, Pipeline, Reduce
from ...executor import Executor, RunResult
from .workload import FrameTransformer, decode_video, summarize_video


class DecodeFrames:
    """V3 Expand UDF：每个视频产生动态数量的 frame records。"""

    def __init__(self, stride: int = 1, max_frames: int | None = None) -> None:
        """保存确定性 frame sampling 参数。"""

        self.stride = stride
        self.max_frames = max_frames

    def run(self, paths: list[str]) -> list[list[Any]]:
        """按视频返回独立的 ordered frame lists。"""

        return [
            decode_video(
                path,
                stride=self.stride,
                max_frames=self.max_frames,
            )
            for path in paths
        ]


class TransformFrames:
    """V3 Map UDF：跨视频 batch frame records。"""

    def __init__(
        self,
        backend: str = "opencv",
        torch_num_threads: int = 1,
    ) -> None:
        """初始化共用 frame heavy-stage backend。"""

        self.transformer = FrameTransformer(
            backend,
            torch_num_threads=torch_num_threads,
        )

    def run(self, frames: list[Any]) -> list[Any]:
        """对每个 frame 执行与 Ray Data 完全相同的 transform。"""

        return self.transformer.transform(frames)


class SummarizeVideos:
    """V3 Reduce UDF：恢复每个视频的 ordered frame feature group。"""

    def run(self, groups: list[list[Any]]) -> list[dict[str, Any]]:
        """逐视频生成稳定 summary。"""

        return [summarize_video(group) for group in groups]


class VideoV3Pipeline(Pipeline):
    """最小真实视频 `1:M → M:1` V3 pipeline。"""

    def __init__(
        self,
        *,
        stride: int,
        max_frames: int | None,
        decode_replicas: int,
        transform_replicas: int,
        transform_batch_size: int,
        batch_scope: str,
        transform_backend: str = "opencv",
        torch_num_threads: int = 1,
    ) -> None:
        """配置 decode、frame transform 和 summary stages。"""

        self.decode = (
            Expand(DecodeFrames)
            .pre_init(stride=stride, max_frames=max_frames)
            .ray_options(
                replicas=decode_replicas,
                batch_size=1,
                num_cpus=1,
            )
        )
        self.transform = (
            Map(TransformFrames)
            .pre_init(
                backend=transform_backend,
                torch_num_threads=torch_num_threads,
            )
            .ray_options(
                replicas=transform_replicas,
                batch_size=transform_batch_size,
                batch_scope=batch_scope,
                num_cpus=max(1, torch_num_threads),
            )
        )
        self.summary = Reduce(SummarizeVideos).ray_options(
            replicas=decode_replicas,
            batch_size=4,
            num_cpus=1,
        )

    def forward(self, videos):
        """声明 dynamic frame fan-out 与 ordered video Reduce。"""

        frames = self.decode(videos)
        features = self.transform(frames)
        return self.summary(anchor=videos, members=features)


def run_v3(
    paths: list[str],
    *,
    stride: int = 1,
    max_frames: int | None = None,
    decode_replicas: int = 1,
    transform_replicas: int = 2,
    transform_batch_size: int = 16,
    batch_scope: str = "elastic",
    transform_backend: str = "opencv",
    torch_num_threads: int = 1,
    microbatch_size: int = 2,
    max_inflight_arenas: int = 2,
) -> RunResult:
    """执行 V3 视频 runner，调用方负责初始化本地/集群 Ray。"""

    pipeline = VideoV3Pipeline(
        stride=stride,
        max_frames=max_frames,
        decode_replicas=decode_replicas,
        transform_replicas=transform_replicas,
        transform_batch_size=transform_batch_size,
        batch_scope=batch_scope,
        transform_backend=transform_backend,
        torch_num_threads=torch_num_threads,
    )
    return Executor(
        pipeline,
        microbatch_size=microbatch_size,
        max_inflight_arenas=max_inflight_arenas,
    ).run(paths)
