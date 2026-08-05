"""使用裸 Ray Data 执行与 V3 相同的视频 frame workload。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .workload import (
    FrameFeature,
    FrameRecord,
    FrameTransformer,
    decode_video,
    summarize_video,
)


def _rows(batch: dict[str, Any]) -> list[dict[str, Any]]:
    """把 Ray Data numpy batch 转为 row dictionaries。"""

    if not batch:
        return []
    size = len(next(iter(batch.values())))
    return [
        {name: values[index] for name, values in batch.items()}
        for index in range(size)
    ]


class ExpandVideoFrames:
    """Ray Data flat_map callable：显式增加 video_id/frame_ordinal。"""

    def __init__(self, stride: int = 1, max_frames: int | None = None) -> None:
        """保存与 V3 runner 相同的 sampling 参数。"""

        self.stride = stride
        self.max_frames = max_frames

    def __call__(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        """把一个视频展开为 Arrow tensor frame rows。"""

        frames = decode_video(
            str(row["video_path"]),
            stride=self.stride,
            max_frames=self.max_frames,
        )
        return [
            {
                "video_id": int(row["video_id"]),
                "video_path": str(row["video_path"]),
                "frame_ordinal": ordinal,
                "source_frame_index": frame.source_frame_index,
                "image_bgr": frame.image_bgr,
            }
            for ordinal, frame in enumerate(frames)
        ]


class TransformFrameBatches:
    """Ray Data map_batches callable：执行共用 frame transform。"""

    def __init__(
        self,
        backend: str = "opencv",
        torch_num_threads: int = 1,
        model_path: str | None = None,
        model_repeats: int = 1,
    ) -> None:
        """初始化与 V3 相同的 heavy-stage backend。"""

        self.transformer = FrameTransformer(
            backend,
            torch_num_threads=torch_num_threads,
            model_path=model_path,
            model_repeats=model_repeats,
        )

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        """保留显式 lineage，并返回 scalar feature columns。"""

        import numpy as np

        rows = _rows(batch)
        features = self.transformer.transform(
            [
                FrameRecord(
                    video_path=str(row["video_path"]),
                    source_frame_index=int(row["source_frame_index"]),
                    image_bgr=row["image_bgr"],
                )
                for row in rows
            ]
        )
        return {
            "video_id": batch["video_id"],
            "video_path": batch["video_path"],
            "frame_ordinal": batch["frame_ordinal"],
            "source_frame_index": batch["source_frame_index"],
            "mean_b": np.asarray(
                [feature.mean_bgr[0] for feature in features],
                dtype="float64",
            ),
            "mean_g": np.asarray(
                [feature.mean_bgr[1] for feature in features],
                dtype="float64",
            ),
            "mean_r": np.asarray(
                [feature.mean_bgr[2] for feature in features],
                dtype="float64",
            ),
            "edge_density": np.asarray(
                [feature.edge_density for feature in features],
                dtype="float64",
            ),
            "digest": np.asarray(
                [feature.digest for feature in features],
            ),
        }


class SummarizeVideoGroup:
    """Ray Data map_groups callable：恢复 frame ordinal 并生成 video summary。"""

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        """校验 ordinal 连续性，调用共用 summary 函数。"""

        import numpy as np

        rows = sorted(
            _rows(batch),
            key=lambda row: int(row["frame_ordinal"]),
        )
        ordinals = [int(row["frame_ordinal"]) for row in rows]
        if ordinals != list(range(len(rows))):
            raise ValueError(f"invalid frame ordinals: {ordinals}")
        features = [
            FrameFeature(
                source_frame_index=int(row["source_frame_index"]),
                mean_bgr=(
                    float(row["mean_b"]),
                    float(row["mean_g"]),
                    float(row["mean_r"]),
                ),
                edge_density=float(row["edge_density"]),
                digest=str(row["digest"]),
            )
            for row in rows
        ]
        summary = summarize_video(features)
        return {
            "video_id": np.asarray(
                [int(rows[0]["video_id"])],
                dtype="int64",
            ),
            "video_name": np.asarray(
                [Path(str(rows[0]["video_path"])).name],
            ),
            "frames": np.asarray([int(summary["frames"])], dtype="int64"),
            "source_indices": [list(summary["source_indices"])],
            "digests": [list(summary["digests"])],
            "mean_edge_density": np.asarray(
                [float(summary["mean_edge_density"])],
                dtype="float64",
            ),
        }


def build_dataset(
    paths: list[str],
    *,
    stride: int = 1,
    max_frames: int | None = None,
    decode_replicas: int = 1,
    transform_replicas: int = 2,
    transform_batch_size: int = 16,
    transform_backend: str = "opencv",
    torch_num_threads: int = 1,
    model_path: str | None = None,
    transform_num_gpus: float = 0.0,
    model_repeats: int = 1,
):
    """构造 lazy Ray Data 视频 DAG。"""

    import ray

    source = ray.data.from_items(
        [
            {"video_id": index, "video_path": path}
            for index, path in enumerate(paths)
        ],
        override_num_blocks=max(
            1,
            min(
                len(paths),
                max(decode_replicas, transform_replicas),
            ),
        ),
    )
    frames = source.flat_map(
        ExpandVideoFrames,
        concurrency=decode_replicas,
        num_cpus=1,
        fn_constructor_kwargs={
            "stride": stride,
            "max_frames": max_frames,
        },
    )
    features = frames.map_batches(
        TransformFrameBatches,
        batch_size=transform_batch_size,
        batch_format="numpy",
        concurrency=transform_replicas,
        num_cpus=max(1, torch_num_threads),
        num_gpus=transform_num_gpus,
        fn_constructor_kwargs={
            "backend": transform_backend,
            "torch_num_threads": torch_num_threads,
            "model_path": model_path,
            "model_repeats": model_repeats,
        },
    )
    return features.groupby(
        "video_id",
        num_partitions=max(1, min(len(paths), decode_replicas)),
    ).map_groups(
        SummarizeVideoGroup,
        batch_format="numpy",
        concurrency=decode_replicas,
        num_cpus=1,
    )


def run_ray_data(paths: list[str], **kwargs: Any) -> tuple[dict[str, Any], ...]:
    """执行 Ray Data 视频 runner，并按 source video_id 恢复输出顺序。"""

    rows = build_dataset(paths, **kwargs).take_all()
    rows.sort(key=lambda row: int(row["video_id"]))
    return tuple(
        {
            "frames": int(row["frames"]),
            "source_indices": tuple(
                int(value) for value in row["source_indices"]
            ),
            "digests": tuple(str(value) for value in row["digests"]),
            "mean_edge_density": round(
                float(row["mean_edge_density"]),
                8,
            ),
        }
        for row in rows
    )
