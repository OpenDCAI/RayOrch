"""Hugging Face 视频样本下载与确定性 OpenCV frame sampling。

本模块只负责数据准备，不依赖 Multigrain runtime 或 Ray，因而可先用 CPU smoke 验证
codec、frame ordinal 和 fan-out。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


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


@dataclass(frozen=True, slots=True)
class VideoManifestEntry:
    """一个正式视频 benchmark source 的可审计元数据。

    ``path`` 是实际交给 V3/Ray Data 的本地文件；其余字段固定数据集来源、split、原始
    sample identity 与标签。运行前可由 ``validate_video_manifest`` 重 probe 文件属性，
    防止手工替换/损坏视频后仍沿用旧实验结论。
    """

    path: str
    dataset: str
    split: str
    source_id: str
    label: str | None = None
    duration_s: float | None = None
    bytes: int | None = None

    def to_json(self) -> dict[str, str | float | int | None]:
        """转为稳定、可直接写入 JSONL/JSON array 的 manifest row。"""

        return {
            "path": self.path,
            "dataset": self.dataset,
            "split": self.split,
            "source_id": self.source_id,
            "label": self.label,
            "duration_s": self.duration_s,
            "bytes": self.bytes,
        }


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


def video_manifest_entry(
    path: str,
    *,
    dataset: str,
    split: str,
    source_id: str,
    label: str | None = None,
) -> VideoManifestEntry:
    """probe 一个本地视频并构造带 duration/bytes 的正式 manifest entry。"""

    resolved = Path(path).resolve()
    metadata = probe_video(str(resolved))
    fps = float(metadata["fps"])
    frame_count = int(metadata["frame_count"])
    duration_s = frame_count / fps if fps > 0 else None
    return VideoManifestEntry(
        path=str(resolved),
        dataset=dataset,
        split=split,
        source_id=source_id,
        label=label,
        duration_s=duration_s,
        bytes=resolved.stat().st_size,
    )


def write_video_manifest(
    entries: Iterable[VideoManifestEntry],
    path: str,
) -> None:
    """写入有序 JSON manifest，拒绝空集合和重复 source identity。"""

    import json

    rows = tuple(entries)
    if not rows:
        raise ValueError("video manifest must not be empty")
    identities = [
        (row.dataset, row.split, row.source_id)
        for row in rows
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("video manifest has duplicate source identities")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            [row.to_json() for row in rows],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def load_video_manifest(
    path: str,
    *,
    limit: int | None = None,
) -> tuple[VideoManifestEntry, ...]:
    """读取并重新验证正式 manifest 的本地文件身份和可 decode 属性。

    ``limit`` 在验证前截取稳定前缀，供大型 manifest 的 smoke gate 使用；正式全量
    gate 保持默认 ``None``，逐条验证全部输入。
    """

    import json

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("video manifest must be a non-empty JSON list")
    if limit is not None:
        if limit <= 0:
            raise ValueError("video manifest limit must be positive")
        raw = raw[:limit]
    entries = tuple(
        VideoManifestEntry(
            path=str(item["path"]),
            dataset=str(item["dataset"]),
            split=str(item["split"]),
            source_id=str(item["source_id"]),
            label=(
                str(item["label"])
                if item.get("label") is not None
                else None
            ),
            duration_s=(
                float(item["duration_s"])
                if item.get("duration_s") is not None
                else None
            ),
            bytes=(
                int(item["bytes"])
                if item.get("bytes") is not None
                else None
            ),
        )
        for item in raw
    )
    seen: set[tuple[str, str, str]] = set()
    for entry in entries:
        identity = (entry.dataset, entry.split, entry.source_id)
        if identity in seen:
            raise ValueError("video manifest has duplicate source identities")
        seen.add(identity)
        actual = video_manifest_entry(
            entry.path,
            dataset=entry.dataset,
            split=entry.split,
            source_id=entry.source_id,
            label=entry.label,
        )
        if entry.bytes is not None and entry.bytes != actual.bytes:
            raise ValueError(
                f"video bytes changed for source_id={entry.source_id}"
            )
        if (
            entry.duration_s is not None
            and actual.duration_s is not None
            and abs(entry.duration_s - actual.duration_s) > 0.05
        ):
            raise ValueError(
                f"video duration changed for source_id={entry.source_id}"
            )
    return entries


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
