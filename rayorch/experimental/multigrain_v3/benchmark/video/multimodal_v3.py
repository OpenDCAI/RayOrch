"""Whisper audio chunks + ViT frames 的双分支 V3 video smoke。"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from typing import Any

from ...api import Expand, Map, Pipeline, Reduce
from ...executor import Executor, RunResult
from .v3 import DecodeFrames, TransformFrames


@dataclass(frozen=True, slots=True)
class AudioChunk:
    """一个 video 内按时间窗编号的 mono PCM chunk。"""

    video_path: str
    ordinal: int
    samples: Any
    sample_rate: int


@dataclass(frozen=True, slots=True)
class TranscriptChunk:
    """一个 audio chunk 的 deterministic transcript。"""

    ordinal: int
    text: str


class DecodeAudioChunks:
    """Expand UDF：使用 ffmpeg 抽取音频并按固定秒数切 chunk。"""

    def __init__(
        self,
        chunk_seconds: float = 4.0,
        sample_rate: int = 16000,
    ) -> None:
        """保存固定窗口与采样率。"""

        self.chunk_seconds = chunk_seconds
        self.sample_rate = sample_rate

    def run(self, paths: list[str]) -> list[list[AudioChunk]]:
        """逐 video 返回 ordered PCM chunks。"""

        import numpy as np

        outputs = []
        width = round(self.chunk_seconds * self.sample_rate)
        for path in paths:
            raw = subprocess.check_output(
                [
                    "ffmpeg",
                    "-loglevel",
                    "error",
                    "-i",
                    path,
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    str(self.sample_rate),
                    "-f",
                    "f32le",
                    "-",
                ]
            )
            samples = np.frombuffer(raw, dtype="<f4").copy()
            outputs.append(
                [
                    AudioChunk(
                        path,
                        ordinal,
                        samples[start : start + width],
                        self.sample_rate,
                    )
                    for ordinal, start in enumerate(
                        range(0, len(samples), width)
                    )
                    if len(samples[start : start + width]) > 0
                ]
            )
        return outputs


class WhisperChunks:
    """GPU Map UDF：对 audio chunk batch 执行 Whisper tiny。"""

    def __init__(self, model_path: str) -> None:
        """加载本地 Whisper model/processor。"""

        import torch
        from transformers import AutoProcessor, WhisperForConditionalGeneration

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
        )
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=(
                torch.float16 if torch.cuda.is_available() else torch.float32
            ),
        ).eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.generation_config.language = "en"
        self.model.generation_config.task = "transcribe"

    def run(self, chunks: list[AudioChunk]) -> list[TranscriptChunk]:
        """执行 batched greedy transcription。"""

        if not chunks:
            return []
        batch = self.processor(
            [chunk.samples for chunk in chunks],
            sampling_rate=chunks[0].sample_rate,
            return_tensors="pt",
            return_attention_mask=True,
            padding=True,
        )
        device = next(self.model.parameters()).device
        features = batch.input_features.to(device)
        if device.type == "cuda":
            features = features.half()
        attention = batch.attention_mask.to(device)
        with self.torch.inference_mode():
            ids = self.model.generate(
                features,
                attention_mask=attention,
                max_new_tokens=32,
                do_sample=False,
            )
        texts = self.processor.batch_decode(
            ids,
            skip_special_tokens=True,
        )
        return [
            TranscriptChunk(chunk.ordinal, text.strip())
            for chunk, text in zip(chunks, texts)
        ]


class SummarizeAudio:
    """Reduce UDF：恢复一个 video 的 ordered transcripts。"""

    def run(self, groups):
        """生成 audio branch summary。"""

        return [
            {
                "chunks": len(group),
                "transcript": " ".join(item.text for item in group).strip(),
            }
            for group in groups
        ]


class SummarizeFrames:
    """Reduce UDF：恢复一个 video 的 ordered frame embedding signatures。"""

    def run(self, groups):
        """生成 frame branch summary。"""

        return [
            {
                "frames": len(group),
                "digests": tuple(item.digest for item in group),
            }
            for group in groups
        ]


class MergeModalities:
    """aligned Map：合并同一 source video 的 audio/frame summaries。"""

    def run(self, audio, frames):
        """返回稳定多模态摘要。"""

        return [
            {
                "audio_chunks": a["chunks"],
                "transcript": a["transcript"],
                "frames": f["frames"],
                "frame_digest": hashlib.blake2b(
                    "|".join(f["digests"]).encode(),
                    digest_size=8,
                ).hexdigest(),
            }
            for a, f in zip(audio, frames)
        ]


class VideoMultimodalV3Pipeline(Pipeline):
    """Video audio/frame 双 Expand 分支及 aligned document-level merge。"""

    def __init__(
        self,
        *,
        whisper_model_path: str,
        vit_model_path: str,
        batch_scope: str = "elastic",
    ) -> None:
        """配置双分支 persistent GPU stages。"""

        self.audio_decode = Expand(DecodeAudioChunks).ray_options(
            replicas=2,
            batch_size=1,
            num_cpus=1,
        )
        self.asr = (
            Map(WhisperChunks)
            .pre_init(whisper_model_path)
            .ray_options(
                replicas=1,
                batch_size=8,
                batch_scope=batch_scope,
                num_cpus=1,
                num_gpus=0.5,
            )
        )
        self.audio_reduce = Reduce(SummarizeAudio).ray_options(batch_size=4)

        self.frame_decode = (
            Expand(DecodeFrames)
            .pre_init(stride=20, max_frames=8)
            .ray_options(replicas=2, batch_size=1, num_cpus=1)
        )
        self.frame_model = (
            Map(TransformFrames)
            .pre_init(
                backend="vit",
                model_path=vit_model_path,
            )
            .ray_options(
                replicas=1,
                batch_size=16,
                batch_scope=batch_scope,
                num_cpus=1,
                num_gpus=0.5,
            )
        )
        self.frame_reduce = Reduce(SummarizeFrames).ray_options(batch_size=4)
        self.merge = Map(MergeModalities).ray_options(batch_size=8)

    def forward(self, videos):
        """声明 audio/frame 双分支和同 source Entity aligned merge。"""

        audio_chunks = self.audio_decode(videos)
        transcripts = self.asr(audio_chunks)
        audio_summary = self.audio_reduce(
            anchor=videos,
            members=transcripts,
        )

        frames = self.frame_decode(videos)
        features = self.frame_model(frames)
        frame_summary = self.frame_reduce(
            anchor=videos,
            members=features,
        )
        return self.merge(audio_summary, frame_summary)


def run_v3(
    paths: list[str],
    *,
    microbatch_size: int = 4,
    max_inflight_arenas: int = 2,
    **pipeline_options: Any,
) -> RunResult:
    """运行多模态双分支 pipeline。"""

    return Executor(
        VideoMultimodalV3Pipeline(**pipeline_options),
        microbatch_size=microbatch_size,
        max_inflight_arenas=max_inflight_arenas,
    ).run(paths)
