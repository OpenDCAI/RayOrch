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
    ) -> None:
        """初始化轻量 OpenCV 或预训练 ResNet18 backend。"""

        self.backend = backend
        self.model = None
        if backend == "opencv":
            return
        if backend != "resnet18":
            raise ValueError(f"unsupported frame backend: {backend}")
        import torch
        from torchvision.models import ResNet18_Weights, resnet18

        if torch_num_threads <= 0:
            raise ValueError("torch_num_threads must be positive")
        torch.set_num_threads(torch_num_threads)
        self.torch = torch
        self.model = resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1,
        ).eval()

    def transform(self, frames: list[FrameRecord]) -> list[FrameFeature]:
        """按 backend 处理一个物理 frame batch。"""

        base = [transform_frame(frame) for frame in frames]
        if self.backend == "opencv" or not frames:
            return base
        assert self.model is not None
        tensors = [self._resnet_tensor(frame.image_bgr) for frame in frames]
        with self.torch.inference_mode():
            logits = self.model(self.torch.stack(tensors)).cpu().numpy()
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
