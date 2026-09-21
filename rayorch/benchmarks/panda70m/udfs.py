from __future__ import annotations

import os
import json
import re
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from typing import Any, Iterator


DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

TEACHERS: tuple[tuple[str, float, str], ...] = (
    (
        "opening_action",
        0.0,
        "Describe the main visible action in this video frame in one concise sentence.",
    ),
    (
        "scene_objects",
        1.0 / 3.0,
        "Describe the people, objects, and scene in this video frame in one precise sentence.",
    ),
    (
        "action_context",
        2.0 / 3.0,
        "Write a concise video caption emphasizing what activity is happening at this moment.",
    ),
    (
        "closing_action",
        1.0,
        "Describe the most important visible subject and action in this video frame in one sentence.",
    ),
)


class VideoCaptionUDF:
    def __init__(
        self,
        *,
        model: str,
        prompts: Mapping[str, str],
        frames_key: str = "frames",
        gpu_memory_utilization: float = 0.9,
        max_tokens: int = 64,
        temperature: float = 0.0,
        trust_remote_code: bool = False,
    ) -> None:
        if not model:
            raise ValueError("model must not be empty")
        if not frames_key:
            raise ValueError("frames_key must not be empty")
        if not prompts:
            raise ValueError("prompts must not be empty")
        if any(not role or not prompt for role, prompt in prompts.items()):
            raise ValueError("prompt roles and text must not be empty")
        if not 0.0 < gpu_memory_utilization <= 1.0:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        from vllm import LLM, SamplingParams

        self.roles = tuple(prompts)
        self.frames_key = frames_key
        self.templates = tuple(
            self._qwen_template(prompt) for prompt in prompts.values()
        )
        self.llm = LLM(
            model=model,
            gpu_memory_utilization=gpu_memory_utilization,
            limit_mm_per_prompt={"image": 1},
            trust_remote_code=trust_remote_code,
        )
        self.sampling = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def run(
        self,
        records: list[Mapping[str, Any]],
    ) -> list[list[dict[str, str]]]:
        requests = self._build_requests(records)
        if not requests:
            return [[] for _ in records]
        outputs = self.llm.generate(
            requests,
            self.sampling,
            use_tqdm=False,
        )
        captions = self._extract_captions(outputs, len(requests))
        return self._group_captions(captions, len(records))

    def _build_requests(
        self,
        records: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        from PIL import Image

        requests: list[dict[str, Any]] = []
        for record in records:
            if self.frames_key not in record:
                raise ValueError(f"video record is missing {self.frames_key!r}")
            frames = record[self.frames_key]
            if len(frames) != len(self.roles):
                raise ValueError(
                    f"{self.frames_key!r} must contain {len(self.roles)} "
                    f"frames, got {len(frames)}"
                )
            requests.extend(
                {
                    "prompt": template,
                    "multi_modal_data": {"image": Image.fromarray(frame)},
                }
                for frame, template in zip(frames, self.templates, strict=True)
            )
        return requests

    @staticmethod
    def _extract_captions(outputs: Sequence[Any], expected: int) -> list[str]:
        if len(outputs) != expected:
            raise ValueError(
                "vLLM returned a different number of outputs than requests"
            )
        return [output.outputs[0].text.strip() for output in outputs]

    def _group_captions(
        self,
        captions: Sequence[str],
        record_count: int,
    ) -> list[list[dict[str, str]]]:
        width = len(self.roles)
        if len(captions) != record_count * width:
            raise ValueError("caption count does not match the input batch")
        return [
            [
                {
                    "role": role,
                    "caption": captions[record_index * width + prompt_index],
                }
                for prompt_index, role in enumerate(self.roles)
            ]
            for record_index in range(record_count)
        ]

    @staticmethod
    def _qwen_template(prompt: str) -> str:
        return (
            "<|im_start|>user\n"
            "<|vision_start|><|image_pad|><|vision_end|>"
            f"{prompt}<|im_end|>\n<|im_start|>assistant\n"
        )


class ExpandPandaClips:
    def run(self, sources: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        groups = []
        for source in sources:
            source_id = str(source.get("source_id", ""))
            clips = source.get("clips")
            if not source_id or not isinstance(clips, list) or not clips:
                raise ValueError("every Panda source must contain source_id and clips")
            groups.append(list(clips))
        return groups


def _resize_long_edge(rgb: Any, long_edge: int) -> Any:
    import cv2

    height, width = rgb.shape[:2]
    scale = long_edge / float(max(height, width))
    if scale >= 1.0:
        return rgb
    new_size = (int(round(width * scale)), int(round(height * scale)))
    return cv2.resize(rgb, new_size, interpolation=cv2.INTER_AREA)


class DecodePandaTeacherFrames:
    def __init__(self, *, long_edge: int = 448, backend: str = "opencv") -> None:
        if backend not in {"opencv", "pyav"}:
            raise ValueError("decode backend must be opencv or pyav")
        if long_edge <= 0:
            raise ValueError("long_edge must be positive")
        self.long_edge = long_edge
        self.backend = backend
        self._hdfs_filesystems: dict[str, Any] = {}

    def _hdfs_location(self, path: str) -> tuple[Any, str]:
        from urllib.parse import unquote, urlsplit

        from pyarrow.fs import FileSystem

        parsed = urlsplit(path)
        authority = f"{parsed.scheme}://{parsed.netloc}"
        filesystem = self._hdfs_filesystems.get(authority)
        if filesystem is None:
            filesystem, filesystem_path = FileSystem.from_uri(path)
            self._hdfs_filesystems[authority] = filesystem
            return filesystem, filesystem_path
        return filesystem, unquote(parsed.path)

    def _copy_hdfs_to_fd(self, path: str, fd: int) -> None:
        filesystem, filesystem_path = self._hdfs_location(path)
        with filesystem.open_input_file(filesystem_path) as source:
            while chunk := source.read(8 * 1024**2):
                view = memoryview(chunk)
                while view:
                    written = os.write(fd, view)
                    if written <= 0:
                        raise OSError("short write while materializing HDFS video")
                    view = view[written:]

    @staticmethod
    def _create_memfd() -> int:
        flags = getattr(os, "MFD_CLOEXEC", 1)
        if hasattr(os, "memfd_create"):
            return os.memfd_create("rayorch-panda-video", flags=flags)
        raise RuntimeError("HDFS OpenCV decode requires Linux memfd_create")

    @contextmanager
    def _seekable_video_path(self, path: str) -> Iterator[str]:
        if not path.startswith("hdfs://"):
            yield path
            return
        fd = self._create_memfd()
        try:
            self._copy_hdfs_to_fd(path, fd)
            os.lseek(fd, 0, os.SEEK_SET)
            yield f"/proc/self/fd/{fd}"
        finally:
            os.close(fd)

    @contextmanager
    def _random_access_video(self, path: str) -> Iterator[Any]:
        if path.startswith("hdfs://"):
            filesystem, filesystem_path = self._hdfs_location(path)
            with filesystem.open_input_file(filesystem_path) as stream:
                yield stream
            return
        with open(path, "rb") as stream:
            yield stream

    @staticmethod
    def _frame_indices(
        total: int,
        clips: Sequence[Mapping[str, Any]],
    ) -> list[tuple[int, ...]]:
        if total <= 0:
            raise ValueError("video has no frames")
        indices = []
        for clip in clips:
            start = float(clip.get("clip_start_fraction", 0.0))
            end = float(clip.get("clip_end_fraction", 1.0))
            if not 0.0 <= start < end <= 1.0:
                raise ValueError("clip fractions must satisfy 0 <= start < end <= 1")
            indices.append(
                tuple(
                    min(
                        total - 1,
                        max(0, round((total - 1) * (start + (end - start) * fraction))),
                    )
                    for _, fraction, _ in TEACHERS
                )
            )
        return indices

    def _decode_opencv_frames(
        self,
        path: str,
        clips: Sequence[Mapping[str, Any]],
    ) -> list[tuple[Any, ...]]:
        import cv2

        with self._seekable_video_path(path) as seekable_path:
            capture = cv2.VideoCapture(seekable_path)
            if not capture.isOpened():
                raise ValueError(f"cannot open video: {path}")
            try:
                total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                frames_by_clip = []
                for clip, indices in zip(
                    clips,
                    self._frame_indices(total, clips),
                    strict=True,
                ):
                    frames = []
                    for frame_index in indices:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                        ok, bgr = capture.read()
                        if not ok:
                            raise ValueError(f"cannot decode frame {frame_index}: {path}")
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        frames.append(_resize_long_edge(rgb, self.long_edge))
                    frames_by_clip.append(tuple(frames))
                return frames_by_clip
            finally:
                capture.release()

    def _decode_pyav_frames(
        self,
        path: str,
        clips: Sequence[Mapping[str, Any]],
    ) -> list[tuple[Any, ...]]:
        from fractions import Fraction

        import av

        with self._random_access_video(path) as stream:
            with av.open(stream, mode="r") as container:
                if not container.streams.video:
                    raise ValueError(f"video has no video stream: {path}")
                video = container.streams.video[0]
                raw_rate = video.average_rate or video.guessed_rate
                if raw_rate is None:
                    raise ValueError(f"video has no usable frame rate: {path}")
                rate = Fraction(raw_rate.numerator, raw_rate.denominator)
                time_base = Fraction(video.time_base.numerator, video.time_base.denominator)
                total = int(video.frames or 0)
                if total <= 0 and video.duration is not None:
                    total = round(video.duration * time_base * rate)
                indices_by_clip = self._frame_indices(total, clips)
                cached: dict[int, Any] = {}
                frames_by_clip = []
                for indices in indices_by_clip:
                    frames = []
                    for frame_index in indices:
                        rgb = cached.get(frame_index)
                        if rgb is None:
                            target_pts = int(Fraction(frame_index, 1) / rate / time_base)
                            container.seek(
                                target_pts,
                                stream=video,
                                backward=True,
                                any_frame=False,
                            )
                            for frame in container.decode(video):
                                if frame.pts is None:
                                    continue
                                decoded_index = round(frame.pts * time_base * rate)
                                if decoded_index >= frame_index:
                                    rgb = _resize_long_edge(
                                        frame.to_ndarray(format="rgb24"),
                                        self.long_edge,
                                    )
                                    break
                            if rgb is None:
                                raise ValueError(f"cannot decode frame {frame_index}: {path}")
                            cached[frame_index] = rgb
                        frames.append(rgb)
                    frames_by_clip.append(tuple(frames))
                return frames_by_clip

    @staticmethod
    def _decoded_item(
        clip: Mapping[str, Any],
        frames: tuple[Any, ...],
    ) -> dict[str, Any]:
        return {
            "source_id": str(clip["source_id"]),
            "clip_index": int(clip["clip_index"]),
            "path": str(clip["path"]),
            "reference_caption": str(clip.get("reference_caption", "")),
            "matching_score": float(clip.get("matching_score", 0.0)),
            "clip_start_s": float(clip.get("clip_start_s", 0.0)),
            "clip_end_s": float(clip.get("clip_end_s", 0.0)),
            "clip_start_fraction": float(clip.get("clip_start_fraction", 0.0)),
            "clip_end_fraction": float(clip.get("clip_end_fraction", 1.0)),
            "frames": frames,
        }

    def run(
        self,
        sources: list[dict[str, Any]],
        clip_groups: list[list[dict[str, Any]]],
    ) -> list[list[dict[str, Any]]]:
        if len(sources) != len(clip_groups):
            raise ValueError("source and clip-group batches must align")
        output_groups = []
        for source, clips in zip(sources, clip_groups, strict=True):
            if not clips:
                raise ValueError(f"source {source['source_id']} has no clips")
            source_ids = {str(clip["source_id"]) for clip in clips}
            paths = {str(clip["path"]) for clip in clips}
            if source_ids != {str(source["source_id"])} or len(paths) != 1:
                raise ValueError("source-level decode received a mixed clip group")
            path = paths.pop()
            if self.backend == "pyav":
                frames_by_clip = self._decode_pyav_frames(path, clips)
            else:
                frames_by_clip = self._decode_opencv_frames(path, clips)
            output_groups.append(
                [
                    self._decoded_item(clip, frames)
                    for clip, frames in zip(clips, frames_by_clip, strict=True)
                ]
            )
        return output_groups


class PandaFusedTeacher(VideoCaptionUDF):
    def __init__(
        self,
        *,
        model: str,
        gpu_memory_utilization: float,
        max_tokens: int = 64,
    ) -> None:
        super().__init__(
            model=model,
            prompts={role: prompt for role, _, prompt in TEACHERS},
            gpu_memory_utilization=gpu_memory_utilization,
            max_tokens=max_tokens,
            trust_remote_code=True,
        )

    def run(self, decoded: list[dict[str, Any]]) -> list[list[dict[str, str]]]:
        requests = self._build_requests(decoded)
        if not requests:
            return [[] for _ in decoded]
        outputs = self.llm.generate(requests, self.sampling, use_tqdm=False)
        return self._group_captions(
            self._extract_captions(outputs, len(requests)),
            len(decoded),
        )


_TOKEN = re.compile(r"[a-z0-9]+")


def _reference_f1(candidate: str, reference: str) -> float:
    left = set(_TOKEN.findall(candidate.lower()))
    right = set(_TOKEN.findall(reference.lower()))
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    if not overlap:
        return 0.0
    precision = overlap / len(left)
    recall = overlap / len(right)
    return 2 * precision * recall / (precision + recall)


def _production_score(caption: str) -> float:
    words = _TOKEN.findall(caption.lower())
    if not words:
        return float("-inf")
    unique_ratio = len(set(words)) / len(words)
    return min(len(words), 24) + 5.0 * unique_ratio - max(0, len(words) - 32) * 1.5


class SelectPandaCaption:
    def run(
        self,
        decoded: list[dict[str, Any]],
        candidates: list[list[dict[str, str]]],
    ) -> list[dict[str, Any]]:
        if len(decoded) != len(candidates):
            raise ValueError("decoded records and teacher outputs must align")
        outputs = []
        for item, raw_candidates in zip(decoded, candidates, strict=True):
            if len(raw_candidates) != len(TEACHERS):
                raise ValueError("each clip must have exactly four candidates")
            candidate_list = [dict(candidate) for candidate in raw_candidates]
            scores = [_production_score(candidate["caption"]) for candidate in candidate_list]
            chosen_index = max(range(len(scores)), key=scores.__getitem__)
            reference = str(item.get("reference_caption", ""))
            reference_scores = [
                _reference_f1(candidate["caption"], reference)
                for candidate in candidate_list
            ]
            outputs.append(
                {
                    "source_id": item["source_id"],
                    "clip_index": item["clip_index"],
                    "path": item["path"],
                    "reference_caption": reference,
                    "panda_matching_score": item.get("matching_score", 0.0),
                    "clip_start_s": item.get("clip_start_s", 0.0),
                    "clip_end_s": item.get("clip_end_s", 0.0),
                    "chosen_index": chosen_index,
                    "chosen_role": candidate_list[chosen_index]["role"],
                    "chosen_caption": candidate_list[chosen_index]["caption"],
                    "chosen_reference_f1": reference_scores[chosen_index],
                    "oracle_reference_f1": max(reference_scores),
                    "candidates": candidate_list,
                }
            )
        return outputs


class SummarizePandaSource:
    def __init__(self, *, output_dir: str) -> None:
        self.output_dir = output_dir
        if output_dir.startswith("hdfs://"):
            raise ValueError("Panda benchmark output_dir must be a local path")

    def run(
        self,
        sources: list[dict[str, Any]],
        grouped_selections: list[list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        from pathlib import Path

        output_root = Path(self.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        if len(sources) != len(grouped_selections):
            raise ValueError("source and selection groups must align")
        outputs = []
        for source, selections in zip(sources, grouped_selections, strict=True):
            ordered = sorted(selections, key=lambda item: int(item["clip_index"]))
            source_id = str(source["source_id"])
            if any(str(item["source_id"]) != source_id for item in ordered):
                raise ValueError("cross-source result reached Panda reduce")
            payload = {
                "source_id": source_id,
                "clips": len(ordered),
                "mean_chosen_reference_f1": _mean(
                    item["chosen_reference_f1"] for item in ordered
                ),
                "mean_oracle_reference_f1": _mean(
                    item["oracle_reference_f1"] for item in ordered
                ),
                "selections": ordered,
            }
            output_path = output_root / f"{source_id}.json"
            output_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            outputs.append({**payload, "output_path": str(output_path.resolve())})
        return outputs


def _mean(values: Sequence[float] | Any) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


__all__ = [
    "DEFAULT_MODEL",
    "DecodePandaTeacherFrames",
    "ExpandPandaClips",
    "PandaFusedTeacher",
    "SelectPandaCaption",
    "SummarizePandaSource",
    "TEACHERS",
    "VideoCaptionUDF",
]
