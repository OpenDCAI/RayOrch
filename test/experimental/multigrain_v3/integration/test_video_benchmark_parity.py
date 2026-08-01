"""真实 OpenCV 视频在 V3 与裸 Ray Data runner 间的结果 parity。"""

from __future__ import annotations

import numpy as np
import pytest

from rayorch.experimental.multigrain_v3.benchmark.video.ray_data import (
    run_ray_data,
)
from rayorch.experimental.multigrain_v3.benchmark.video.v3 import run_v3


pytestmark = pytest.mark.usefixtures("ray_cluster")


def _write_video(path, *, frames: int, offset: int) -> None:
    """写一个不同长度的 MJPG 视频，制造动态 fan-out。"""

    import cv2

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (48, 32),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV build cannot create MJPG AVI fixture")
    try:
        for index in range(frames):
            image = np.zeros((32, 48, 3), dtype=np.uint8)
            image[:, :, 0] = (offset + index * 17) % 255
            image[:, index % 48, 1] = 255
            writer.write(image)
    finally:
        writer.release()


def test_v3_and_ray_data_match_on_skewed_video_fanout(tmp_path) -> None:
    """两个不同 frame 数的视频应跨 runner 恢复完全相同的 ordered summary。"""

    short = tmp_path / "short.avi"
    long = tmp_path / "long.avi"
    _write_video(short, frames=5, offset=10)
    _write_video(long, frames=11, offset=80)
    paths = [str(short), str(long)]
    options = {
        "stride": 2,
        "max_frames": None,
        "decode_replicas": 1,
        "transform_replicas": 2,
        "transform_batch_size": 4,
    }

    v3 = run_v3(
        paths,
        **options,
        microbatch_size=1,
        max_inflight_arenas=2,
    )
    ray_data = run_ray_data(paths, **options)

    assert v3.get() == ray_data
    assert [result["frames"] for result in ray_data] == [3, 6]
    assert v3.metrics["grains_per_rpc"] > 1
