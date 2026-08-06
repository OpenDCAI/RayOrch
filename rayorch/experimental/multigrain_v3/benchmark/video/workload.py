"""V3 与裸 Ray Data 共用的视频 frame workload 业务函数。

这里的 UDF 不知道 lineage、Ray actor 或物理 batch。两套 runner 复用同一 decode、frame
transform 和 ordered summary，避免因业务实现不同制造虚假性能差异。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from .data import sample_frames


@dataclass(frozen=True, slots=True)
class FrameRecord:
    """送入重型 frame stage 的业务 payload。"""

    video_path: str
    source_frame_index: int
    image_bgr: Any


@dataclass(frozen=True, slots=True)
class FrameFeature:
    """一个 frame transform 的稳定轻量输出。"""

    source_frame_index: int
    mean_bgr: tuple[float, float, float]
    edge_density: float
    digest: str


@dataclass(frozen=True, slots=True)
class FrameCaption:
    """一个 frame 的 deterministic VLM caption。"""

    source_frame_index: int
    text: str


def decode_video(
    path: str,
    *,
    stride: int,
    max_frames: int | None,
) -> list[FrameRecord]:
    """把一个视频动态展开为按 source order 排列的 frame records。"""

    return [
        FrameRecord(
            video_path=frame.video_path,
            source_frame_index=frame.source_frame_index,
            image_bgr=frame.image_bgr,
        )
        for frame in sample_frames(
            path,
            stride=stride,
            max_frames=max_frames,
        )
    ]


def transform_frame(frame: FrameRecord) -> FrameFeature:
    """执行确定性 CPU 图像变换，作为无模型调度 smoke 的共同 heavy stage。"""

    import cv2
    import numpy as np

    image = frame.image_bgr
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    mean = image.reshape(-1, 3).mean(axis=0)
    return FrameFeature(
        source_frame_index=frame.source_frame_index,
        mean_bgr=tuple(round(float(value), 6) for value in mean),
        edge_density=round(float(np.count_nonzero(edges) / edges.size), 8),
        digest=hashlib.blake2b(image.tobytes(), digest_size=8).hexdigest(),
    )


class FrameTransformer:
    """V3 与 Ray Data 共用的可选 frame heavy-stage 实现。"""

    def __init__(
        self,
        backend: str = "opencv",
        *,
        torch_num_threads: int = 1,
        model_path: str | None = None,
        model_repeats: int = 1,
    ) -> None:
        """初始化 OpenCV、ResNet18 或 HuggingFace ViT backend。"""

        self.backend = backend
        self.model = None
        if model_repeats <= 0:
            raise ValueError("model_repeats must be positive")
        self.model_repeats = model_repeats
        if backend == "opencv":
            return
        if backend not in {"resnet18", "vit"}:
            raise ValueError(f"unsupported frame backend: {backend}")
        import torch

        if torch_num_threads <= 0:
            raise ValueError("torch_num_threads must be positive")
        torch.set_num_threads(torch_num_threads)
        self.torch = torch
        if backend == "resnet18":
            from torchvision.models import ResNet18_Weights, resnet18

            self.model = resnet18(
                weights=ResNet18_Weights.IMAGENET1K_V1,
            ).eval()
            self.processor = None
        else:
            if model_path is None:
                raise ValueError("ViT backend requires model_path")
            from transformers import AutoImageProcessor, ViTModel

            self.processor = AutoImageProcessor.from_pretrained(
                model_path,
                local_files_only=True,
            )
            self.model = ViTModel.from_pretrained(
                model_path,
                local_files_only=True,
            ).eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def transform(self, frames: list[FrameRecord]) -> list[FrameFeature]:
        """按 backend 处理一个物理 frame batch。"""

        base = [transform_frame(frame) for frame in frames]
        if self.backend == "opencv" or not frames:
            return base
        assert self.model is not None
        if self.backend == "vit":
            return self._transform_vit(frames, base)
        tensors = [self._resnet_tensor(frame.image_bgr) for frame in frames]
        device = self._model_device()
        with self.torch.inference_mode():
            batch = self.torch.stack(tensors).to(device)
            for _ in range(self.model_repeats):
                logits_tensor = self.model(batch)
            logits = logits_tensor.cpu().numpy()
        return [
            FrameFeature(
                source_frame_index=feature.source_frame_index,
                mean_bgr=feature.mean_bgr,
                edge_density=feature.edge_density,
                digest=hashlib.blake2b(
                    ",".join(
                        str(int(class_id))
                        for class_id in logits[index].argsort()[-5:][::-1]
                    ).encode("ascii"),
                    digest_size=8,
                ).hexdigest(),
            )
            for index, feature in enumerate(base)
        ]

    def _model_device(self):
        """返回模型 device；测试替身没有 parameters 时使用 CPU。"""

        try:
            return next(self.model.parameters()).device
        except AttributeError:
            return self.torch.device("cpu")

    def _transform_vit(
        self,
        frames: list[FrameRecord],
        base: list[FrameFeature],
    ) -> list[FrameFeature]:
        """对一个物理 batch 执行 ViT embedding。"""

        import cv2
        import numpy as np

        assert self.processor is not None and self.model is not None
        images = [
            cv2.cvtColor(frame.image_bgr, cv2.COLOR_BGR2RGB)
            for frame in frames
        ]
        inputs = self.processor(images=images, return_tensors="pt")
        device = self._model_device()
        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }
        with self.torch.inference_mode():
            for _ in range(self.model_repeats):
                output = self.model(**inputs)
            embeddings = output.last_hidden_state[:, 0].cpu().numpy()
        return [
            FrameFeature(
                source_frame_index=feature.source_frame_index,
                mean_bgr=feature.mean_bgr,
                edge_density=feature.edge_density,
                digest=hashlib.blake2b(
                    ",".join(
                        str(int(value))
                        for value in np.argsort(
                            np.abs(embeddings[index])
                        )[-16:][::-1]
                    ).encode("ascii"),
                    digest_size=8,
                ).hexdigest(),
            )
            for index, feature in enumerate(base)
        ]

    def _resnet_tensor(self, image_bgr: Any):
        """把 BGR ndarray 规范化为 ResNet18 的 3×224×224 tensor。"""

        import cv2
        import numpy as np

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        scale = 256.0 / min(height, width)
        resized = cv2.resize(
            rgb,
            (
                max(224, round(width * scale)),
                max(224, round(height * scale)),
            ),
            interpolation=cv2.INTER_LINEAR,
        )
        top = (resized.shape[0] - 224) // 2
        left = (resized.shape[1] - 224) // 2
        crop = resized[top : top + 224, left : left + 224]
        tensor = self.torch.from_numpy(
            np.ascontiguousarray(crop.transpose(2, 0, 1))
        ).float()
        tensor.div_(255.0)
        mean = self.torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
        std = self.torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
        return (tensor - mean) / std


class FrameCaptioner:
    """SmolVLM 等 image-text model 的 persistent batch caption UDF。"""

    def __init__(
        self,
        model_path: str,
        *,
        prompt: str = "Describe the main action in five words.",
        max_new_tokens: int = 12,
    ) -> None:
        """加载 fixed local model/processor，并使用 greedy generation。"""

        import torch
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
        )

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
        )
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("image-text processor must expose a tokenizer")
        # Decoder-only generation must pad variable-length batches on the
        # left.  Right padding makes captions depend on the physical batch
        # assembled by the scheduler even with greedy decoding.
        tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=(
                torch.float16 if torch.cuda.is_available() else torch.float32
            ),
        ).eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.max_new_tokens = max_new_tokens
        self.chat_prompt = self.processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            add_generation_prompt=True,
        )

    def caption(self, frames: list[FrameRecord]) -> list[FrameCaption]:
        """对一个物理 frame batch 执行 greedy caption generation。"""

        import cv2
        from PIL import Image

        if not frames:
            return []
        images = [
            Image.fromarray(
                cv2.cvtColor(frame.image_bgr, cv2.COLOR_BGR2RGB)
            )
            for frame in frames
        ]
        inputs = self.processor(
            text=[self.chat_prompt] * len(frames),
            images=images,
            return_tensors="pt",
            padding=True,
        )
        device = next(self.model.parameters()).device
        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }
        with self.torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        prefix = inputs["input_ids"].shape[1]
        texts = self.processor.batch_decode(
            outputs[:, prefix:],
            skip_special_tokens=True,
        )
        return [
            FrameCaption(
                source_frame_index=frame.source_frame_index,
                text=text.strip(),
            )
            for frame, text in zip(frames, texts)
        ]


def prepare_resnet18_weights() -> None:
    """在 Driver 预下载固定 torchvision ResNet18 权重，避免 actor 下载竞争。"""

    from torchvision.models import ResNet18_Weights

    ResNet18_Weights.IMAGENET1K_V1.get_state_dict(
        progress=True,
        check_hash=True,
    )


def summarize_video(features: list[FrameFeature]) -> dict[str, Any]:
    """把有序 frame features 汇总为可跨 runner 比较的 video result。"""

    source_indices = tuple(feature.source_frame_index for feature in features)
    if source_indices != tuple(sorted(source_indices)):
        raise ValueError("video features are not ordered by source frame index")
    return {
        "frames": len(features),
        "source_indices": source_indices,
        "digests": tuple(feature.digest for feature in features),
        "mean_edge_density": (
            round(
                sum(feature.edge_density for feature in features)
                / len(features),
                8,
            )
            if features
            else 0.0
        ),
    }


def summarize_captions(
    captions: list[FrameCaption],
) -> dict[str, Any]:
    """把有序 frame captions 汇总为稳定 video result。"""

    source_indices = tuple(
        caption.source_frame_index for caption in captions
    )
    if source_indices != tuple(sorted(source_indices)):
        raise ValueError("video captions are not ordered")
    texts = tuple(caption.text for caption in captions)
    return {
        "frames": len(captions),
        "source_indices": source_indices,
        "captions": texts,
        "caption_chars": sum(len(text) for text in texts),
    }
