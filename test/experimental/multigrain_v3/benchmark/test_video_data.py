"""视频数据适配层的离线、确定性测试。"""

from __future__ import annotations

import numpy as np
import pytest

from rayorch.experimental.multigrain_v3.benchmark.video.data import (
    UCF101_SAMPLE_FILES,
    VideoManifestEntry,
    load_video_manifest,
    probe_video,
    sample_frames,
    video_manifest_entry,
    write_video_manifest,
)


def _write_video(path, *, frames: int = 7) -> None:
    """写一个无需外网的微型 MJPG AVI fixture。"""

    import cv2

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (32, 24),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV build cannot create MJPG AVI fixture")
    try:
        for index in range(frames):
            image = np.full((24, 32, 3), index * 20, dtype=np.uint8)
            writer.write(image)
    finally:
        writer.release()


def test_sample_frames_assigns_contiguous_logical_ordinals(tmp_path) -> None:
    """stride 只改变 source index，logical ordinal 必须从零连续增长。"""

    path = tmp_path / "tiny.avi"
    _write_video(path)

    frames = sample_frames(str(path), stride=2, max_frames=3)

    assert [frame.ordinal for frame in frames] == [0, 1, 2]
    assert [frame.source_frame_index for frame in frames] == [0, 2, 4]
    assert all(frame.image_bgr.shape == (24, 32, 3) for frame in frames)


def test_probe_video_reports_codec_metadata(tmp_path) -> None:
    """probe 应报告后续 fan-out 配置所需的基础 metadata。"""

    path = tmp_path / "tiny.avi"
    _write_video(path, frames=5)

    metadata = probe_video(str(path))

    assert metadata["frame_count"] == 5
    assert metadata["width"] == 32
    assert metadata["height"] == 24
    assert metadata["fps"] > 0


def test_sample_frames_validates_bounds(tmp_path) -> None:
    """无效 stride/max_frames 在打开 codec 前即可明确失败。"""

    with pytest.raises(ValueError, match="stride"):
        sample_frames(str(tmp_path / "missing.avi"), stride=0)
    with pytest.raises(ValueError, match="max_frames"):
        sample_frames(str(tmp_path / "missing.avi"), max_frames=-1)


def test_ucf101_smoke_manifest_is_small_and_explicit() -> None:
    """实验只下载两个独立 AVI，不隐式拉取完整 171MB tarball。"""

    assert UCF101_SAMPLE_FILES == (
        "v_BabyCrawling_g19_c02.avi",
        "v_BasketballDunk_g14_c06.avi",
    )


def test_load_manifest_limit_validates_only_the_stable_prefix(tmp_path) -> None:
    first = tmp_path / "first.avi"
    _write_video(first, frames=2)
    entry = video_manifest_entry(
        str(first),
        dataset="fixture",
        split="test",
        source_id="first",
    )
    missing = VideoManifestEntry(
        path=str(tmp_path / "missing.avi"),
        dataset="fixture",
        split="test",
        source_id="missing",
    )
    manifest = tmp_path / "manifest.json"
    write_video_manifest((entry, missing), str(manifest))

    assert load_video_manifest(str(manifest), limit=1) == (entry,)
    with pytest.raises(ValueError, match="limit"):
        load_video_manifest(str(manifest), limit=0)
