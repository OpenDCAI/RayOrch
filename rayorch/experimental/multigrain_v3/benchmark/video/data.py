"""Hugging Face 视频样本下载与确定性 OpenCV frame sampling。

本模块只负责数据准备，不依赖 Multigrain runtime 或 Ray，因而可先用 CPU smoke 验证
codec、frame ordinal 和 fan-out。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


TINY_VIDEO_REPO = "hf-internal-testing/tiny-video-dataset"
TINY_VIDEO_FILE = "videos/00001.mp4"
UCF101_SUBSET_REPO = "sayakpaul/ucf101-subset"
UCF101_SAMPLE_FILES = (
    "v_BabyCrawling_g19_c02.avi",
    "v_BasketballDunk_g14_c06.avi",
)


@dataclass(frozen=True, slots=True)
class VideoFrame:
    """一个视频内按采样顺序编号的 frame leaf。"""

    video_path: str
    ordinal: int
    source_frame_index: int
    image_bgr: Any


def download_tiny_video(
    cache_dir: str | None = None,
    *,
    revision: str | None = None,
) -> str:
    """下载约 266 KB 的公开 tiny MP4，并返回本地路径。"""

    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=TINY_VIDEO_REPO,
        repo_type="dataset",
        filename=TINY_VIDEO_FILE,
        cache_dir=cache_dir,
        revision=revision,
    )


def download_ucf101_samples(
    cache_dir: str | None = None,
    *,
    revision: str | None = None,
) -> tuple[str, ...]:
    """下载 UCF101 subset 中两个独立 AVI smoke samples。"""

    from huggingface_hub import hf_hub_download

    return tuple(
        hf_hub_download(
            repo_id=UCF101_SUBSET_REPO,
            repo_type="dataset",
            filename=filename,
            cache_dir=cache_dir,
            revision=revision,
        )
        for filename in UCF101_SAMPLE_FILES
    )


def probe_video(path: str) -> dict[str, float | int | str]:
    """读取 codec metadata，确认 OpenCV 能打开候选视频。"""

    import cv2

    capture = cv2.VideoCapture(path)
    try:
        if not capture.isOpened():
            raise ValueError(f"OpenCV cannot open video: {path}")
        return {
            "path": str(Path(path).resolve()),
            "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        capture.release()


def sample_frames(
    path: str,
    *,
    stride: int = 1,
    max_frames: int | None = None,
) -> list[VideoFrame]:
    """按 source frame index 取样，并生成连续 logical ordinal。"""

    import cv2

    if stride <= 0:
        raise ValueError("stride must be positive")
    if max_frames is not None and max_frames < 0:
        raise ValueError("max_frames must be non-negative")
    capture = cv2.VideoCapture(path)
    frames: list[VideoFrame] = []
    source_index = 0
    try:
        if not capture.isOpened():
            raise ValueError(f"OpenCV cannot open video: {path}")
        while max_frames is None or len(frames) < max_frames:
            ok, image = capture.read()
            if not ok:
                break
            if source_index % stride == 0:
                frames.append(
                    VideoFrame(
                        video_path=path,
                        ordinal=len(frames),
                        source_frame_index=source_index,
                        image_bgr=image,
                    )
                )
            source_index += 1
    finally:
        capture.release()
    return frames
